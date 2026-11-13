//! # MultiSimdQuickHeap: A fast SIMD-based priority queue with multi partitioning
//!
//! Implementation of the `MultiSimdQuickHeap`.
//!

use quickheap::ElemT;

use std::fmt::Debug;
use std::marker::PhantomData;

use quickheap::pivots::MedianOfM;
use quickheap::simd;

use simd::SPLIT_PIVOTS_CUR_AND_NEXT;

use crate::ElemConstraint;
use crate::pivot_strategies::EqualMultiMedianOfM;
use crate::pivot_strategies::MultiPivotStrategy;
#[cfg(feature = "layer_stats")]
use crate::stats;

/* Constants for development testing */
const SIMD_INSERT: bool = false;
const LOWEST_LAYER_SIZE: usize = 16;

use quickheap::partitioning::VecPartitioning;

/// The MultiSimdQuickHeap implementation
///
/// - `T`: the element type: `u32`, `i32`, `u64`, or `i64`.
/// - `S`: an optional tag to select the SIMD backend: [`Avx2`], [`Avx512`], or [`Neon`].
pub struct MultiSimdQuickHeap<
    T: ElemT,
    S: simd::SimdElem<T>,
    const B: usize = 4,
    const L: usize = 1024,
> {
    /// A decreasing array of the pivots for all layers.
    /// buckets[i] >= pivots[i] >= buckets[i+1]
    /// Values equal to pivots[i] can be in layer i or i+1.
    /// The first layer does not have a pivot in this array.
    ///
    /// The effective number of layers is always 1 longer than `pivots`.
    ///
    /// This will have enough underlying capacity for out-of-bounds SIMD reads.
    pivots: Vec<T>,
    /// The values in each layer.
    /// pivots[i-1] >= elements of buckets[i] >= pivots[i]
    /// Values equal to pivots[i] can be in layer i or i-1.
    ///
    /// This can be longer than `layer` to reuse allocations.
    buckets: Vec<Vec<T>>,
    size: usize,
    _backend: PhantomData<S>,
}

/// Return a default instance with plenty (128) layers of empty buckets.
impl<T: ElemT + Default, S: simd::SimdElem<T>, const B: usize, const L: usize> Default
    for MultiSimdQuickHeap<T, S, B, L>
{
    fn default() -> Self {
        Self {
            pivots: Vec::with_capacity(128),
            buckets: (0..128).map(|_| vec![]).collect(),
            size: 0,
            _backend: PhantomData,
        }
    }
}

impl<
    T: ElemConstraint + ElemT + Default + Debug,
    S: simd::SimdElem<T>,
    const B: usize,
    const L: usize,
> MultiSimdQuickHeap<T, S, B, L>
{
    const N: usize = LOWEST_LAYER_SIZE;

    /// Return the total capacity over all buckets.
    pub fn capacity(&self) -> usize {
        self.buckets.iter().map(|b| b.capacity()).sum()
    }

    /// Return the number of elements currently in the priority queue.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Return whether the priority queue contains no elements.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Push `t` onto the priority queue.
    pub fn push(&mut self, t: T) {
        #[cfg(feature = "layer_stats")]
        stats::layer_stats::record(self.pivots.len() + 1);

        let target_layer = unsafe { simd::push_position::<T, S>(&self.pivots, t) };

        let layer = &mut self.buckets[target_layer];

        if target_layer == self.pivots.len() && layer.len() < Self::N {
            // Count the number of larger elements in the prefix and insert the new element after them.

            let pos: usize;
            if SIMD_INSERT {
                pos = simd::insert_pos::<T, S>(layer, t);
                debug_assert!(pos == layer.partition_point(|x| *x > t));
            } else {
                pos = layer.partition_point(|x| *x > t);
            }

            layer.insert(pos, t);
        } else {
            layer.push(t);
        }

        self.size += 1;
    }

    /// Pop the smallest element from the priority queue.
    pub fn pop(&mut self) -> Option<T> {
        #[cfg(feature = "layer_stats")]
        stats::layer_stats::record(self.pivots.len() + 1);

        // If the total size is 0, return none
        if self.size == 0 {
            return None;
        }

        // Split the current layer as long as it is too large.
        if self.buckets[self.pivots.len()].len() > Self::N {
            while self.buckets[self.pivots.len()].len() > Self::N {
                self.partition();
            }

            // Sort final layer decreasing.
            let layer = &mut self.buckets[self.pivots.len()];
            layer.sort_unstable_by_key(|&x| std::cmp::Reverse(x));
        }

        // Find and extract the minimum.
        let layer = &mut self.buckets[self.pivots.len()];
        debug_assert!(!layer.is_empty());

        let min = layer.pop().unwrap();

        // Update the active layer.
        if layer.is_empty() && !self.pivots.is_empty() {
            self.pivots.pop();

            // Sort the new final layer decreasing if it's already small.
            if self.buckets[self.pivots.len()].len() <= Self::N {
                let layer = &mut self.buckets[self.pivots.len()];
                layer.sort_unstable_by_key(|&x| std::cmp::Reverse(x));
            }
        }

        self.size -= 1;

        Some(min)
    }

    #[inline(never)]
    fn partition(&mut self) {
        let layer_size = self.buckets[self.pivots.len()].len();

        if layer_size <= L {
            self.dual_partition();
        } else {
            self.multi_partition();
        }
    }

    #[inline(always)]
    fn multi_partition(&mut self) {
        let total_layers = self.pivots.len() + 1;

        // Reserve space for an additional L layers when needed.
        if total_layers + B + S::L >= self.pivots.capacity() {
            self.pivots.reserve(S::L);
        }

        while self.buckets.len() < total_layers + B - 1 {
            self.buckets.push(vec![]);
        }

        // Alias the layers (to be split) and the next layer.
        let layer = self.pivots.len();
        let layers: &mut [Vec<T>] = &mut self.buckets[layer..layer + B];

        let n = layers[0].len();

        // Sample pivots using the pivot strategy
        let (pivots, pivot_pos) = EqualMultiMedianOfM::<2>::pick(&layers[0]);

        // Clear the next layers
        for i in 1..B {
            layers[i].clear();
        }

        let lens =
            VecPartitioning::<T, S>::multi_partition::<B>(layers, &pivots[0..(B - 1)], pivot_pos);

        // Compact: keep non-empty buckets, move empty ones to the group's tail.
        let mut w = 0;
        for k in 0..B {
            if lens[k] == 0 {
                continue;
            }
            if w != k {
                self.buckets.swap(layer + w, layer + k);
            }
            // Separator between the previous kept bucket and this one:
            // the upper bound of bucket k is pivots[k-1].
            if w > 0 {
                self.pivots.push(pivots[k - 1]);
            }
            w += 1;
        }
    }

    #[inline(always)]
    fn dual_partition(&mut self) {
        let total_layers = self.pivots.len() + 1;

        // Reserve space for an additional L layers when needed.
        if total_layers + 2 * S::L >= self.pivots.capacity() {
            self.pivots.reserve(S::L);
        }

        if total_layers == self.buckets.len() {
            self.buckets.push(vec![]);
        }

        // Alias the current layer (to be split) and the next layer.
        let [cur_layer, next_layer] = &mut self.buckets[self.pivots.len()..=self.pivots.len() + 1]
        else {
            unreachable!()
        };

        let n = cur_layer.len();

        // Sample a pivot using the pivot strategy
        let (pivot, pivot_pos) = MedianOfM::<3>::pick(cur_layer);

        self.pivots.push(pivot);

        // Clear the next layer
        next_layer.clear();

        VecPartitioning::<T, S>::partition::<SPLIT_PIVOTS_CUR_AND_NEXT>(
            cur_layer, next_layer, pivot, pivot_pos,
        );

        debug_assert!(next_layer.len() + cur_layer.len() == n);

        // If all elements ended up in the current layer
        if next_layer.is_empty() {
            self.pivots.pop();
        }

        // If we extracted all elements to the next layer
        // because the pivot was the largest one,
        // undo and try again.
        if cur_layer.is_empty() {
            std::mem::swap(cur_layer, next_layer);

            self.pivots.pop();
        }
    }
}
