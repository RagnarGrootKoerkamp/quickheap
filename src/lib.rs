//! # SimdQuickHeap: A fast SIMD-based priority queue
//!
//! Just use the [`SimdQuickHeap`] type and it's [`default`](SimdQuickHeap::default), [`push`](SimdQuickHeap::push), and [`pop`](SimdQuickHeap::pop) functions.
//!
//! This is a _min_-queue, so `pop` returns the _smallest_ element in the queue.
//!
//! The [`ConfigurableSimdQuickHeap`] type is mostly for benchmarking only, to test various parameters.
//!
//! By default, it uses AVX2, or AVX-512 when available during compile time.
//! To force one or the other, use `SimdQuickHeap<T, Avx2>` or `SimdQuickHeap<T, Avx512>`.
//!
//! ## Example
//! ```
//! let mut q = quickheap::SimdQuickHeap::<u64>::default();
//! q.push(4);
//! q.push(1);
//! q.push(7);
//! assert_eq!(q.pop(), Some(1));
//! q.push(7);
//! q.push(3);
//! assert_eq!(q.pop(), Some(3));
//! assert_eq!(q.pop(), Some(4));
//! assert_eq!(q.pop(), Some(7));
//! assert_eq!(q.pop(), Some(7));
//! assert_eq!(q.pop(), None);
//! ```

#![feature(linked_list_cursors)]

#[cfg(feature = "c")]
#[doc(hidden)]
pub mod c;

#[doc(hidden)]
pub mod pivot_strategies;

#[doc(hidden)]
pub mod rebalancing_strategies;

use std::fmt::Debug;

mod simd;
#[cfg(test)]
mod test;

pub use simd::{Avx2, Avx512};
use std::marker::PhantomData;

/// Tag to use with [`ConfigurableSimdQuickHeap`] to use AVX-512 if it is available.
#[cfg(not(target_feature = "avx512f"))]
pub type Simd = Avx2;

#[cfg(target_feature = "avx512f")]
pub type Simd = Avx512;

/// Wrapper trait for `Copy + Ord`.
#[doc(hidden)]
pub trait Elem: Copy + Ord {}
impl<T: Copy + Ord> Elem for T {}

/// The SIMD tag ([`Avx2`] or [`Avx512`]) must implement `SimdElem<T>`.
///
/// For now, this means you can only use `u32`, `i32`, `u64`, and `i64`.
pub use simd::SimdElem;

use crate::{buckets::vec_bucket::VecBucket, rebalancing_strategies::NoRebalancing};

use std::ops::Add;
pub trait One {
    fn one() -> Self;
}

impl One for i32 {
    fn one() -> Self {
        1
    }
}
impl One for i64 {
    fn one() -> Self {
        1
    }
}
impl One for u32 {
    fn one() -> Self {
        1
    }
}
impl One for u64 {
    fn one() -> Self {
        1
    }
}

pub mod buckets;

/// The full SimdQuickHeap implementation, with all configuration parameters.
///
/// - `T`: the element type.
/// - `S`: the SIMD tag: [`Avx2`] or [`Avx512`]. Default AVX-512 if available.
/// - `P`: the pivoting strategy; see [`pivot_strategies`]. Default median of 3.
/// - `N`: partition until the bottom layer is <N. Default `16`.
/// - `SORT`: whether to keep the bottom layer sorted. Default `true`.
pub struct ConfigurableSimdQuickHeap<
    T: Elem,
    B: buckets::Bucket<T>,
    S: simd::SimdElem<T> = Simd,
    P: pivot_strategies::PivotStrategy = pivot_strategies::MedianOfM<3>,
    R: rebalancing_strategies::RebalancingStrategy<T> = rebalancing_strategies::NoRebalancing,
    const N: usize = 16,
    const SORT: bool = true,
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
    buckets: Vec<B>,
    equal_buckets: Vec<bool>,

    size: usize,
    #[allow(dead_code)]
    rebal_iteration: usize,
    #[allow(dead_code)]
    _p: PhantomData<P>,
    _r: PhantomData<R>,
    _backend: PhantomData<S>,
}

/// A SIMD-based priority queue. Entrypoint of the crate.
///
/// Returns the *smallest* element first.
///
/// Works for `i32`, `u32`, `i64`, and `u64`.
///
/// Uses AVX-512 instructions when available at compile time.
pub type SimdQuickHeap<T> = ConfigurableSimdQuickHeap<
    T,
    VecBucket<T>,
    Simd,
    pivot_strategies::MedianOfM<3>,
    NoRebalancing,
    16,
    true,
>;

/// Return a default instance with plenty (128) layers of empty buckets.
impl<
    T: Elem + Default + Add<Output = T> + One,
    B: buckets::Bucket<T>,
    S: simd::SimdElem<T>,
    P: pivot_strategies::PivotStrategy,
    R: rebalancing_strategies::RebalancingStrategy<T>,
    const N: usize,
    const SORT: bool,
> Default for ConfigurableSimdQuickHeap<T, B, S, P, R, N, SORT>
{
    fn default() -> Self {
        Self {
            pivots: Vec::with_capacity(128),
            buckets: (0..128).map(|_| B::default()).collect(),
            equal_buckets: (0..128).map(|_| false).collect(),
            size: 0,
            rebal_iteration: 0,
            _p: PhantomData,
            _r: PhantomData,
            _backend: PhantomData,
        }
    }
}

impl<
    T: Elem + Debug + Default + Add<Output = T> + One,
    B: buckets::Bucket<T>,
    S: simd::SimdElem<T>,
    R: rebalancing_strategies::RebalancingStrategy<T>,
    P: pivot_strategies::PivotStrategy,
    const N: usize,
    const SORT: bool,
> ConfigurableSimdQuickHeap<T, B, S, P, R, N, SORT>
{
    /// Return the total capacity over all buckets.
    pub fn capacity(&self) -> usize {
        self.buckets.iter().map(|b| b.capacity()).sum()
    }

    /// Return the number of elements currently in the heap.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Return whether the heap contains no elements.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Push `t` onto the heap.
    pub fn push(&mut self, t: T) {
        #[cfg(feature = "rebalancing")] // TODO: Is this the right position here?
        R::on_push(self.size, &mut self.pivots, &mut self.buckets);

        let target_layer = simd::push_position::<T, S>(&self.pivots, t);
        let layer = &mut self.buckets[target_layer];
        layer.reserve(S::L + 1);
        if SORT && target_layer == self.pivots.len() && layer.len() < N {
            // Count the number of larger elements in the prefix and insert the new element after them.
            let pos = layer.insert_index(t);
            layer.insert(pos, t);
            // TODO: SIMD
        } else {
            layer.push(t);
        }

        self.size += 1;
    }

    /// Pop the smallest element from the queue.
    pub fn pop(&mut self) -> Option<T> {
        #[cfg(feature = "rebalancing")]
        {
            self.rebal_iteration += 1;
        }

        let layer = self.pivots.len();
        // Only the top layer can be empty.
        if layer == 0 && self.buckets[0].is_empty() {
            return None;
        }

        #[cfg(feature = "equal_buckets")]
        {
            if self.equal_buckets[layer] {
                assert!(self.buckets[layer].len() > 0);
                let elem = self.buckets[layer].pop();

                // Update the active layer.
                if self.buckets[layer].is_empty() && self.pivots.len() > 0 {
                    self.equal_buckets[self.pivots.len()] = false;
                    self.pivots.pop();

                    // Sort the new final layer decreasing if it's already small.
                    if !self.equal_buckets[self.pivots.len()]
                        && SORT
                        && self.buckets[self.pivots.len()].len() <= N
                    {
                        let layer = &mut self.buckets[self.pivots.len()];
                        layer.sort_decreasing();
                    }
                }

                return elem;
            }
        }

        // Split the current layer as long as it is too large.
        if self.buckets[self.pivots.len()].len() > N {
            while self.buckets[self.pivots.len()].len() > N {
                self.partition();
            }
            if SORT {
                // Sort final layer decreasing.
                let layer = &mut self.buckets[self.pivots.len()];
                layer.sort_decreasing();
            }
        }
        // Find and extract the minimum.
        let layer = &mut self.buckets[self.pivots.len()];
        let min = if SORT {
            layer.pop().unwrap()
        } else {
            let min_pos = simd::position_min_bucket::<T, S, B>(layer);
            layer.remove(min_pos)
        };

        // Update the active layer.
        if layer.is_empty() && self.pivots.len() > 0 {
            self.pivots.pop();

            // Sort the new final layer decreasing if it's already small.
            if !self.equal_buckets[self.pivots.len()]
                && SORT
                && self.buckets[self.pivots.len()].len() <= N
            {
                let layer = &mut self.buckets[self.pivots.len()];
                layer.sort_decreasing();
            }
        }

        self.size -= 1;
        Some(min)
    }

    #[inline(never)]
    fn partition(&mut self) {
        let layer = self.pivots.len();

        debug_assert!(!self.equal_buckets[layer]);

        #[cfg(feature = "equal_buckets")]
        {
            if layer_len > 1000 {
                use crate::buckets::equal_buckets::EqualBucketSamplingTest;
                use crate::buckets::equal_buckets::EqualBucketTest;

                let (test, elem) =
                    EqualBucketSamplingTest::<8>::check::<T, B>(&self.buckets[layer]);

                if test {
                    self.equal_partition(elem);
                    return;
                }
            }
        }

        // Reserve space for an additional L layers when needed.
        if layer + 2 * S::L >= self.pivots.capacity() {
            self.pivots.reserve(S::L);
        }
        if layer + 1 == self.buckets.len() {
            self.equal_buckets.push(false);
            self.buckets.push(B::default());
        }
        // Alias the current layer (to be split) and the next layer.
        let [cur_layer, next_layer] = &mut self.buckets[layer..=layer + 1] else {
            unreachable!()
        };
        let n = cur_layer.len();

        // Sample a pivot using the pivot strategy
        let (pivot, pivot_pos) = P::pick_bucket(cur_layer);

        self.pivots.push(pivot);

        // Reserve space in the next layer,
        // and make sure the current layer can hold a spare SIMD register.
        cur_layer.reserve(S::L);
        next_layer.clear();
        next_layer.reserve(n + S::L);

        unsafe { cur_layer.set_len(n + S::L) };
        unsafe { next_layer.set_len(n + S::L) };

        let n2 = n.next_multiple_of(S::L).saturating_sub(S::L);

        // Partition a list into two using SIMD.
        let mut cur_len = 0;
        let mut next_len = 0;
        let half = (pivot_pos + 1).min(n2).next_multiple_of(S::L);
        let threshold = S::splat(pivot);
        for i in (0..half).step_by(S::L) {
            unsafe {
                S::partition_fast_bucket::<B, true>(
                    S::simd_from_slice(cur_layer.get_unchecked(i, S::L).as_slice()),
                    threshold,
                    cur_layer,
                    &mut cur_len,
                    next_layer,
                    &mut next_len,
                );
            }
        }

        for i in (half..n2).step_by(S::L) {
            unsafe {
                S::partition_fast_bucket::<B, false>(
                    S::simd_from_slice(cur_layer.get_unchecked(i, S::L).as_slice()),
                    threshold,
                    cur_layer,
                    &mut cur_len,
                    next_layer,
                    &mut next_len,
                );
            }
        }

        if n2 < n {
            let threshold = if pivot_pos >= n2 {
                S::splat(S::wrapping_add_one(pivot))
            } else {
                S::splat(pivot)
            };
            unsafe {
                S::partition_slow_bucket(
                    S::simd_from_slice(cur_layer.get_unchecked(n2, S::L).as_slice()),
                    S::splat(S::from_usize(n - n2)),
                    threshold,
                    cur_layer,
                    &mut cur_len,
                    next_layer,
                    &mut next_len,
                );
            }
        }

        cur_layer.flush(cur_len);
        next_layer.flush(next_len);

        debug_assert!(next_len > 0);

        unsafe {
            cur_layer.set_len(cur_len);
            next_layer.set_len(next_len);
        }

        // If we extracted all elements to the next layer
        // because the pivot was the largest one,
        // undo and try again.
        if cur_len == 0 {
            std::mem::swap(cur_layer, next_layer);
            self.pivots.pop().unwrap();
        }

        #[cfg(feature = "rebalancing")]
        {
            if self.rebal_iteration < R::MAX_REBAL_ITERATIONS {
                return;
            }
            self.rebal_iteration = 0;
            R::on_pop(self.size, &mut self.pivots, &mut self.buckets);
        }
    }

    #[inline(never)]
    fn equal_partition(&mut self, pivot: T) {
        let layer = self.pivots.len();

        // Reserve space for an additional L layers when needed.
        if layer + 2 * S::L >= self.pivots.capacity() {
            self.pivots.reserve(S::L);
        }

        if layer + 1 == self.buckets.len() {
            self.buckets.push(B::default());
            self.buckets.push(B::default());
            self.equal_buckets.push(false);
            self.equal_buckets.push(false);
        }

        // Alias the current layer (to be split) and the next layers.
        let [cur_layer, equal_layer, next_layer] = &mut self.buckets[layer..=layer + 2] else {
            unreachable!()
        };
        let n = cur_layer.len();

        self.pivots.push(pivot + T::one());
        self.pivots.push(pivot);

        // Reserve space in the next layers,
        // and make sure the current layer can hold a spare SIMD register.
        cur_layer.reserve(S::L);
        equal_layer.clear();
        equal_layer.reserve(n + S::L);
        next_layer.clear();
        next_layer.reserve(n + S::L);

        unsafe { cur_layer.set_len(n + S::L) };
        unsafe { equal_layer.set_len(n + S::L) };
        unsafe { next_layer.set_len(n + S::L) };

        let mut cur_len = 0;
        let mut equal_len = 0;
        let mut next_len = 0;

        let n2 = n.next_multiple_of(S::L).saturating_sub(S::L);

        // Partition a list into three (smaller, equal, greater) using SIMD.
        let threshold = S::splat(pivot);

        for i in (0..n2).step_by(S::L) {
            unsafe {
                S::partition_equal_bucket(
                    S::simd_from_slice(cur_layer.get_unchecked(i, S::L).as_slice()),
                    S::splat(S::from_usize(S::L)),
                    threshold,
                    cur_layer,
                    &mut cur_len,
                    equal_layer,
                    &mut equal_len,
                    next_layer,
                    &mut next_len,
                );
            }
        }

        if n2 < n {
            unsafe {
                S::partition_equal_bucket(
                    S::simd_from_slice(cur_layer.get_unchecked(n2, S::L).as_slice()),
                    S::splat(S::from_usize(n - n2)),
                    threshold,
                    cur_layer,
                    &mut cur_len,
                    equal_layer,
                    &mut equal_len,
                    next_layer,
                    &mut next_len,
                );
            }
        }

        cur_layer.flush(cur_len);
        equal_layer.flush(equal_len);
        next_layer.flush(next_len);

        debug_assert!(equal_len > 0);

        unsafe {
            cur_layer.set_len(cur_len);
            equal_layer.set_len(equal_len);
            next_layer.set_len(next_len);
        }

        self.equal_buckets[layer + 1] = true;

        // If we extracted all elements to the next layer
        // because the equal element was the largest one
        if cur_len == 0 {
            assert!(false);
            std::mem::swap(cur_layer, equal_layer);
            std::mem::swap(cur_layer, next_layer);
            self.pivots.swap(layer + 1, layer + 2);
            self.equal_buckets.swap(layer + 1, layer + 2);
            self.pivots.pop().unwrap();
        }
    }
}
