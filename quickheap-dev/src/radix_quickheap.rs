//! # RadixSimdQuickHeap: A fast SIMD-based priority queue with radix style partitioning
//!
//! Implementation of the `RadixSimdQuickHeap`.
//!

use quickheap::ElemT;

use std::fmt::Debug;
use std::marker::PhantomData;

use quickheap::pivots::MedianOfM;
use quickheap::simd;

use simd::Avx2;
use simd::Avx512;
use simd::Neon;
use simd::SimdElem;

#[cfg(feature = "layer_stats")]
use crate::stats;

/* Constants for development testing */
const SIMD_INSERT: bool = true;
const LOWEST_LAYER_SIZE: usize = 32;

/// The default Simd backend, decided at compile time based on the target architecture.
#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
type Simd = simd::Avx2;
/// The default Simd backend, decided at compile time based on the target architecture.
#[cfg(target_feature = "avx512f")]
type Simd = simd::Avx512;
/// The default Simd backend, decided at compile time based on the target architecture.
#[cfg(target_arch = "aarch64")]
type Simd = simd::Neon;

use quickheap::partitioning::VecPartitioning;

/// The key-only SimdQuickHeap implementation
///
/// - `T`: the element type: `u32`, `i32`, `u64`, or `i64`.
/// - `S`: an optional tag to select the SIMD backend: [`Avx2`], [`Avx512`], or [`Neon`].
pub struct RadixSimdQuickHeap<
    T: ElemT,
    S: simd::SimdElem<T> = Simd,
    const B: usize = 2,
    const L: usize = 512,
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
    for RadixSimdQuickHeap<T, S, B, L>
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

impl<T: ElemT + Default + Debug, S: simd::SimdElem<T>, const B: usize, const L: usize>
    RadixSimdQuickHeap<T, S, B, L>
{
    const N: usize = 16;

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
            self.radix_partition();
        }
    }

    #[inline(always)]
    fn radix_partition(&mut self) {
        let num_buckets = 2usize.pow(B as u32);

        let total_layers = self.pivots.len() + 1;

        // Reserve space for additional layers when needed.
        if total_layers + num_buckets + S::L >= self.pivots.capacity() {
            // TODO: Check this
            self.pivots.reserve(2usize.pow(B as u32) + S::L);
        }

        while self.pivots.len() + 1 + 2usize.pow(B as u32) > self.buckets.len() {
            // TODO: Check this
            self.buckets.push(vec![]);
        }

        // Alias the current layer (to be split) and the next layers.
        let layers = &mut self.buckets[self.pivots.len()..self.pivots.len() + num_buckets];

        let n = layers[0].len();

        // Clear the next layers
        for i in 1..num_buckets {
            layers[i].clear();
        }

        let pivots = VecPartitioning::<T, S>::radix_partition::<B>(layers);

        if pivots.len() == 0 {
            self.dual_partition();
            return;
        }

        for pivot in pivots {
            self.pivots.push(pivot);
        }

        let mut sum = 0;
        for layer in layers {
            sum += layer.len();
        }

        debug_assert!(sum == n);

        let last_layer_idx = self.pivots.len();
        for b in 0..num_buckets {
            let idx = last_layer_idx - b;
            if idx > 0 && self.buckets[idx].is_empty() {
                self.buckets.remove(idx);
                // keep the upper boundary; the lowest layer has no lower pivot
                let p = if idx < self.pivots.len() {
                    idx
                } else {
                    idx - 1
                };
                self.pivots.remove(p);
            }
        }

        /*
        // Remove the empty layers that are created in between
        let last_layer_idx = self.pivots.len();
        for b in 0..num_buckets {
            if self.buckets[last_layer_idx - b].is_empty() && (last_layer_idx - b) > 0 {
                self.buckets.remove(last_layer_idx - b);
                self.pivots.remove(last_layer_idx - 1 - b);
            }
        } */
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

        VecPartitioning::<T, S>::partition(cur_layer, next_layer, pivot, pivot_pos);

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
