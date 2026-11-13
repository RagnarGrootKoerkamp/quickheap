//! # MultiSimdQuickHeap: A fast SIMD-based priority queue with multi partitioning
//!
//! Implementation of the `MultiSimdQuickHeap`.
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

use crate::ElemConstraint;
#[cfg(feature = "layer_stats")]
use crate::stats;

/* Constants for development testing */
const SIMD_INSERT: bool = false;
const LOWEST_LAYER_SIZE: usize = 16;

use quickheap::partitioning::VecPartitioning;

struct Bucket<T> {
    equal: bool,
    data: Vec<T>,
}

/// The EqualSimdQuickHeap implementation
///
/// - `T`: the element type: `u32`, `i32`, `u64`, or `i64`.
/// - `S`: an optional tag to select the SIMD backend: [`Avx2`], [`Avx512`], or [`Neon`].
pub struct EqualSimdQuickHeap<T: ElemT, S: simd::SimdElem<T>> {
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
    buckets: Vec<Bucket<T>>,
    size: usize,
    _backend: PhantomData<S>,
}

/// Return a default instance with plenty (128) layers of empty buckets.
impl<T: ElemT + Default, S: simd::SimdElem<T>> Default for EqualSimdQuickHeap<T, S> {
    fn default() -> Self {
        Self {
            pivots: Vec::with_capacity(128),
            buckets: (0..128)
                .map(|_| Bucket {
                    equal: false,
                    data: vec![],
                })
                .collect(),
            size: 0,
            _backend: PhantomData,
        }
    }
}

impl<T: ElemConstraint + ElemT + Default + Debug, S: simd::SimdElem<T>> EqualSimdQuickHeap<T, S> {
    const N: usize = LOWEST_LAYER_SIZE;

    /// Return the total capacity over all buckets.
    pub fn capacity(&self) -> usize {
        self.buckets.iter().map(|b| b.data.capacity()).sum()
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

        let mut target_layer = unsafe { simd::push_position::<T, S>(&self.pivots, t) };

        if target_layer > 0 && self.buckets[target_layer].equal && t < self.pivots[target_layer - 1]
        {
            // The equal layer only holds values equal to pivots[target_layer - 1].
            if target_layer < self.pivots.len() {
                // Not the top layer: t == pivots[target_layer], so it may go one layer down.
                target_layer += 1;
            } else {
                // Equal layer is the top layer: open a new layer below it.
                if self.buckets.len() == self.pivots.len() + 1 {
                    self.buckets.push(Bucket {
                        equal: false,
                        data: vec![],
                    });
                }
                let new_pivot = self.pivots[target_layer - 1].dec();
                self.pivots.push(new_pivot);

                let new_layer = &mut self.buckets[self.pivots.len()];
                new_layer.equal = false; // reused buckets may be stale
                new_layer.data.clear();
                new_layer.data.push(t);

                self.size += 1;
                return;
            }
        }

        let layer = &mut self.buckets[target_layer];

        if !layer.equal && target_layer == self.pivots.len() && layer.data.len() < Self::N {
            // Count the number of larger elements in the prefix and insert the new element after them.
            let pos;
            if SIMD_INSERT {
                pos = simd::insert_pos::<T, S>(&layer.data, t);
                debug_assert!(pos == layer.data.partition_point(|x| *x > t));
            } else {
                pos = layer.data.partition_point(|x| *x > t);
            }
            layer.data.insert(pos, t);
        } else {
            layer.data.push(t);
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
        if self.buckets[self.pivots.len()].data.len() > Self::N {
            while !self.buckets[self.pivots.len()].equal
                && self.buckets[self.pivots.len()].data.len() > Self::N
            {
                self.partition();
            }

            // Sort final layer decreasing.
            if !self.buckets[self.pivots.len()].equal {
                let layer = &mut self.buckets[self.pivots.len()];
                layer.data.sort_unstable_by_key(|&x| std::cmp::Reverse(x));
            }
        }

        // Find and extract the minimum.
        let layer = &mut self.buckets[self.pivots.len()];
        debug_assert!(!layer.data.is_empty());

        let min = layer.data.pop().unwrap();

        // Update the active layer.
        if layer.data.is_empty() && !self.pivots.is_empty() {
            layer.equal = false;
            self.pivots.pop();

            // Sort the new final layer decreasing if it's already small.
            let buck = &self.buckets[self.pivots.len()];
            if !buck.equal && buck.data.len() <= Self::N {
                let layer = &mut self.buckets[self.pivots.len()];
                layer.data.sort_unstable_by_key(|&x| std::cmp::Reverse(x));
            }
        }

        self.size -= 1;

        Some(min)
    }

    #[inline(never)]
    fn partition(&mut self) {
        let layer = self.pivots.len();
        let total_layers = layer + 1;

        // Reserve space for an additional L layers when needed.
        if total_layers + 2 * S::L >= self.pivots.capacity() {
            self.pivots.reserve(S::L);
        }

        if total_layers == self.buckets.len() {
            self.buckets.push(Bucket {
                equal: false,
                data: vec![],
            });
        }

        // Alias the current layer (to be split) and the next layer.
        let [cur_layer_buck, next_layer_buck] =
            &mut self.buckets[self.pivots.len()..=self.pivots.len() + 1]
        else {
            unreachable!()
        };

        let [cur_layer, next_layer] = [&mut cur_layer_buck.data, &mut next_layer_buck.data];

        let n = cur_layer.len();

        // Sample a pivot using the pivot strategy
        let (pivot, pivot_pos) = MedianOfM::<3>::pick(cur_layer);

        self.pivots.push(pivot);

        // Clear the next layer
        next_layer.clear();

        if layer > 0 && self.pivots[layer] == self.pivots[layer - 1] {
            if pivot == T::minimum() {
                cur_layer_buck.equal = true;
                self.pivots.pop();

                #[cfg(feature = "partition_stats")]
                stats::partition_stats::equal_bucket();
                return;
            }

            #[cfg(feature = "partition_stats")]
            stats::partition_stats::equal_bucket();

            self.pivots[layer] = pivot.dec();
            cur_layer_buck.equal = true; // Implicit equal bucket
            VecPartitioning::<T, S>::partition::<{ simd::ONLY_PIVOTS_TO_CUR }>(
                cur_layer, next_layer, pivot, pivot_pos,
            );
        } else {
            VecPartitioning::<T, S>::partition::<{ simd::SPLIT_PIVOTS_CUR_AND_NEXT }>(
                cur_layer, next_layer, pivot, pivot_pos,
            );
        }

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
