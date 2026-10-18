#![allow(unused)]
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

#[doc(hidden)]
pub mod pivot_strategies;

#[doc(hidden)]
pub mod rebalancing_strategies;

#[doc(hidden)]
pub mod buckets;

mod simd_buckets;

use std::fmt::Debug;

#[cfg(test)]
mod test;

#[cfg(target_feature = "avx2")]
pub use quickheap::simd::Avx2;
#[cfg(target_feature = "avx512f")]
pub use quickheap::simd::Avx512;
#[cfg(target_arch = "aarch64")]
pub use quickheap::simd::Neon;

pub mod stats;

use std::marker::PhantomData;

/// Tag to use with [`ConfigurableSimdQuickHeap`]: picks the widest available backend
/// (AVX-512 > AVX2 > NEON).
#[cfg(target_feature = "avx512f")]
pub type Simd = Avx512;

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
pub type Simd = Avx2;

#[cfg(target_arch = "aarch64")]
pub type Simd = Neon;

/// The SIMD tag ([`Avx2`] or [`Avx512`]) must implement `SimdElem<T>`.
///
/// For now, this means you can only use `u32`, `i32`, `u64`, and `i64`.
pub use quickheap::simd::SimdElem;

use crate::pivot_strategies::PivotStrategy;
use quickheap::partitioning::{MultiPartition, Partition};
use quickheap::simd;

use crate::{buckets::block_arena::BlockArena, pivot_strategies::RandomPivot};

use std::ops::Sub;

/// Wrapper around `Copy + Ord`.
pub trait Elem: Copy + Ord {}
impl<T: Copy + Ord> Elem for T {}

pub trait ElemConstraint {
    fn one() -> Self;
    fn dec(&self) -> Self;
    fn minimum() -> Self;
    fn maximum() -> Self;
    fn to_usize(&self) -> usize;
}

impl ElemConstraint for i32 {
    fn one() -> Self {
        1
    }

    fn minimum() -> Self {
        i32::MIN
    }

    fn maximum() -> Self {
        i32::MAX
    }

    fn dec(&self) -> Self {
        self.saturating_sub(1)
    }

    fn to_usize(&self) -> usize {
        *self as usize
    }
}

impl ElemConstraint for i64 {
    fn one() -> Self {
        1
    }

    fn minimum() -> Self {
        i64::MIN
    }

    fn maximum() -> Self {
        i64::MAX
    }

    fn dec(&self) -> Self {
        self.saturating_sub(1)
    }

    fn to_usize(&self) -> usize {
        *self as usize
    }
}

impl ElemConstraint for u32 {
    fn one() -> Self {
        1
    }

    fn minimum() -> Self {
        0
    }

    fn maximum() -> Self {
        u32::MAX
    }

    fn dec(&self) -> Self {
        self.saturating_sub(1)
    }

    fn to_usize(&self) -> usize {
        *self as usize
    }
}

impl ElemConstraint for u64 {
    fn one() -> Self {
        1
    }

    fn minimum() -> Self {
        0
    }

    fn maximum() -> Self {
        u64::MAX
    }

    fn dec(&self) -> Self {
        self.saturating_sub(1)
    }

    fn to_usize(&self) -> usize {
        *self as usize
    }
}

/// The full SimdQuickHeap implementation, with all configuration parameters.
///
/// - `T`: the element type.
/// - `S`: the SIMD tag: [`Avx2`] or [`Avx512`]. Default AVX-512 if available.
/// - `P`: the pivoting strategy; see [`pivot_strategies`]. Default median of 3.
/// - `N`: partition until the bottom layer is <N. Default `16`.
/// - `SORT`: whether to keep the bottom layer sorted. Default `true`.
pub struct ConfigurableSimdQuickHeap<
    T: Elem,
    B: buckets::Bucket<T, K, CAP>,
    PART: Partition<T, B>,
    S: simd::SimdElem<T> = Simd,
    PIVOT: pivot_strategies::PivotStrategy = pivot_strategies::MedianOfM<3>,
    R: rebalancing_strategies::RebalancingStrategy<T, B, K, CAP> = rebalancing_strategies::NoRebalancing,
    const N: usize = 16,
    const K: usize = 128,
    const CAP: usize = 128,
    const SORT: bool = true,
    const EQUAL: bool = false,
    const P: usize = 2,
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
    free_arena: *mut BlockArena<T, K, CAP>,

    size: usize,
    // TODO: Store subtree sizes
    // subtree_sizes: Vec<usize>,

    global_deletions: usize,
    #[allow(dead_code)]
    rebal_push_iteration: usize,
    #[allow(dead_code)]
    rebal_pop_iteration: usize,
    #[allow(dead_code)]
    _p: PhantomData<PIVOT>,
    _r: PhantomData<R>,
    _part: PhantomData<PART>,
    _backend: PhantomData<S>,
}

/// Return a default instance with plenty (128) layers of empty buckets.
impl<
    T: Elem + Debug + Default + Sub<Output = T> + ElemConstraint,
    B: buckets::Bucket<T, K, CAP>,
    PART: Partition<T, B>,
    S: simd::SimdElem<T>,
    PIVOT: pivot_strategies::PivotStrategy,
    R: rebalancing_strategies::RebalancingStrategy<T, B, K, CAP>,
    const N: usize,
    const K: usize,
    const CAP: usize,
    const SORT: bool,
    const EQUAL: bool,
    const P: usize,
> Default for ConfigurableSimdQuickHeap<T, B, PART, S, PIVOT, R, N, K, CAP, SORT, EQUAL, P>
{
    fn default() -> Self {
        let free_arena = Box::into_raw(Box::from(BlockArena::<T, K, CAP>::new()));
        Self {
            pivots: Vec::with_capacity(128),
            buckets: (0..128).map(|_| B::default(free_arena)).collect(),
            equal_buckets: (0..128).map(|_| false).collect(),
            size: 0,
            global_deletions: 0,
            rebal_pop_iteration: 0,
            rebal_push_iteration: 0,
            free_arena,
            _p: PhantomData,
            _part: PhantomData,
            _r: PhantomData,
            _backend: PhantomData,
        }
    }
}

impl<
    T: Elem + Debug + Default + Sub<Output = T> + ElemConstraint,
    B: buckets::Bucket<T, K, CAP>,
    PART: Partition<T, B> + MultiPartition<T, B>,
    S: simd::SimdElem<T>,
    R: rebalancing_strategies::RebalancingStrategy<T, B, K, CAP>,
    PIVOT: pivot_strategies::PivotStrategy,
    const N: usize,
    const K: usize,
    const CAP: usize,
    const SORT: bool,
    const EQUAL: bool,
    const P: usize,
> ConfigurableSimdQuickHeap<T, B, PART, S, PIVOT, R, N, K, CAP, SORT, EQUAL, P>
{
    /// Initializes the heap from a given list of layers
    pub fn from_vecs(layers: Vec<Vec<T>>) -> Self {
        let mut size: usize = 0;
        let mut buckets = vec![];
        let num_buckets = layers.len();
        let mut pivots: Vec<T> = vec![];

        let free_arena = Box::into_raw(Box::from(BlockArena::<T, K, CAP>::new()));

        for layer in &layers {
            assert!(layer.len() > 0);
            let mut b = B::default(free_arena);
            let max = layer.iter().min().unwrap();

            for e in layer {
                b.push(*e);
                size += 1;
            }

            buckets.push(b);
            pivots.push(*max);
        }

        pivots.pop();

        Self {
            pivots,
            buckets,
            equal_buckets: (0..num_buckets).map(|_| false).collect(),
            size: size,
            global_deletions: 0,
            rebal_pop_iteration: 0,
            rebal_push_iteration: 0,
            free_arena,
            _p: PhantomData,
            _part: PhantomData,
            _r: PhantomData,
            _backend: PhantomData,
        }
    }

    /// Return the total capacity over all buckets.
    pub fn capacity(&self) -> usize {
        unsafe {
            // For the blocked data structures we want the
            // total sizes of all allocated blocks
            if (*self.free_arena).capacity() != 0 {
                return (*self.free_arena).capacity();
            }
        }

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
        #[cfg(feature = "layer_stats")]
        stats::layer_stats::record(self.pivots.len() + 1);

        self.rebal_push_iteration += 1;
        let target_layer = simd::push_position::<T, S>(&self.pivots, t);

        if EQUAL
            && target_layer > 0
            && target_layer == self.pivots.len()
            && self.equal_buckets[target_layer]
            && t < self.pivots[target_layer - 1]
        {
            // Push to top layer, create a new layer to leave the equal layer intact
            if self.buckets.len() == (self.pivots.len() + 1) {
                self.buckets.push(B::default(self.free_arena));
                self.equal_buckets.push(false);
            }

            self.pivots.push(self.pivots[target_layer - 1].dec());
            self.buckets[self.pivots.len()].push(t);

            self.size += 1;
            return;
        }

        let layer = &mut self.buckets[target_layer];

        if !self.equal_buckets[target_layer]
            && SORT
            && target_layer == self.pivots.len()
            && layer.len() < N
        {
            // Count the number of larger elements in the prefix and insert the new element after them.
            let pos = layer.insert_index(t);
            layer.insert(pos, t);
            // TODO: SIMD
        } else {
            layer.push(t);
        }

        self.size += 1;

        if self.rebal_push_iteration >= R::MAX_REBAL_ITERATIONS {
            R::on_push(
                self.size,
                self.global_deletions,
                target_layer,
                &mut self.pivots,
                &mut self.buckets,
                self.free_arena,
            );
            self.rebal_push_iteration = 0;
        }
    }

    fn pull(&mut self, layer: usize) {
        // Semantic: Pull elements into layer

        // If the bucket to pull from is also empty, recurse
        if self.buckets[layer - 1].len() == 0 {
            self.pull(layer - 1);
        }

        let total_layers = self.pivots.len() + 1;
        let logical_layer = total_layers - layer - 1;

        let shift = 120.min(logical_layer + 3);
        let max_small_pull = ((2 as u128) << shift) as u128; // TODO: Maybe a problem for big number of layers

        if (self.buckets[layer - 1].len() as u128) < max_small_pull {
            // TODO: Check spec with paper
            // If the bucket to pull from is small -> swap
            debug_assert!(self.buckets[layer].len() == 0);
            self.buckets.swap(layer, layer - 1);

            if layer >= 2 {
                self.pivots[layer - 1] = self.pivots[layer - 2];
            }
        } else {
            // If the bucket to pull from too big -> partition
            while self.buckets[layer].is_empty() {
                self.partition(layer - 1);
            }
        }

        debug_assert!(self.buckets[layer].len() > 0);
    }

    /// Pop the smallest element from the queue.
    pub fn pop(&mut self) -> Option<T> {
        #[cfg(feature = "layer_stats")]
        stats::layer_stats::record(self.pivots.len() + 1);

        self.rebal_pop_iteration += 1;
        self.global_deletions += 1;

        // Only the top layer can be empty.
        if self.size == 0 {
            return None;
        }

        if R::ALLOW_EMPTY_LAYERS {
            if self.buckets[self.pivots.len()].is_empty() {
                self.pull(self.pivots.len());
                // Clean up lowest layers
                while self.pivots.len() > 0 {
                    if self.buckets[0].is_empty() {
                        self.buckets.remove(0);
                        self.pivots.remove(0);
                    } else {
                        break;
                    }
                }
            }

            debug_assert!(self.buckets[self.pivots.len()].len() > 0);

            while self.buckets[self.pivots.len()].len() > K {
                self.partition(self.pivots.len());
            }

            debug_assert!(self.buckets[self.pivots.len()].len() > 0);

            self.buckets[self.pivots.len()].sort_decreasing();

            self.size -= 1;
            return self.buckets[self.pivots.len()].pop();
        }

        // Split the current layer as long as it is too large.
        if self.buckets[self.pivots.len()].len() > N {
            while !self.equal_buckets[self.pivots.len()]
                && self.buckets[self.pivots.len()].len() > N
            {
                self.partition(self.pivots.len());
            }
            if !self.equal_buckets[self.pivots.len()] && SORT {
                // Sort final layer decreasing.
                let layer = &mut self.buckets[self.pivots.len()];
                layer.sort_decreasing();
            }
        }
        // Find and extract the minimum.
        let layer = &mut self.buckets[self.pivots.len()];
        debug_assert!(!layer.is_empty());
        let min = if SORT || self.equal_buckets[self.pivots.len()] {
            layer.pop().unwrap()
        } else {
            let min_pos = simd_buckets::position_min_bucket::<T, S, B, K, CAP>(layer);
            layer.remove(min_pos)
        };

        // Update the active layer.
        if !R::ALLOW_EMPTY_LAYERS && layer.is_empty() && self.pivots.len() > 0 {
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

        self.size -= 1;

        if self.rebal_pop_iteration <= R::MAX_REBAL_ITERATIONS {
            return Some(min);
        }
        self.rebal_pop_iteration = 0;
        R::on_pop(
            self.size,
            &mut self.global_deletions,
            &mut self.pivots,
            &mut self.buckets,
        );

        Some(min)
    }

    #[inline(never)]
    fn partition(&mut self, layer: usize) {
        if P == 2 {
            self.dual_partition(layer);
            return;
        }

        if self.buckets[layer].len() <= 128 {
            self.dual_partition(layer);
        } else {
            self.multi_partition();
        }
    }

    #[inline(never)]
    fn dual_partition(&mut self, layer: usize) {
        let total_layers = self.pivots.len() + 1;

        #[cfg(feature = "partition_stats")]
        stats::partition_stats::partition(self.buckets[layer].len());

        // Reserve space for an additional L layers when needed.
        if total_layers + 2 + S::L >= self.pivots.capacity() {
            self.pivots.reserve(S::L);
        }

        if layer == (total_layers - 1) && total_layers == self.buckets.len() {
            self.equal_buckets.push(false);
            self.buckets.push(B::default(self.free_arena));
        }

        // Alias the current layer (to be split) and the next layer.
        let [cur_layer, next_layer] = &mut self.buckets[layer..=layer + 1] else {
            unreachable!()
        };

        let n = cur_layer.len();

        // Sample a pivot using the pivot strategy
        let (pivot, pivot_pos) = PIVOT::pick_bucket(cur_layer);

        let old_pivot: T;
        let replace: bool;
        if layer < self.pivots.len() {
            replace = true;
            old_pivot = self.pivots.remove(layer);
            self.pivots.insert(layer, pivot);
        } else {
            replace = false;
            self.pivots.push(pivot);
            old_pivot = pivot;
        }

        // Clear the next layer
        debug_assert!(self.equal_buckets[layer + 1] == false);
        debug_assert!(self.equal_buckets[layer] == false);
        next_layer.clear();

        if EQUAL && (layer > 0) && self.pivots[layer] == self.pivots[layer - 1] {
            if pivot == T::minimum() {
                self.equal_buckets[layer] = true;

                if replace {
                    self.pivots[layer] = old_pivot;
                } else {
                    self.pivots.pop();
                }

                #[cfg(feature = "partition_stats")]
                stats::partition_stats::equal_bucket();
                return;
            }

            #[cfg(feature = "partition_stats")]
            stats::partition_stats::equal_bucket();
            self.pivots[layer] = pivot.dec();
            self.equal_buckets[layer] = true; // Implicit equal bucket
            PART::partition::<{ simd::ONLY_PIVOTS_TO_CUR }>(
                cur_layer, next_layer, pivot, pivot_pos,
            );
        } else {
            PART::partition::<{ simd::SPLIT_PIVOTS_CUR_AND_NEXT }>(
                cur_layer, next_layer, pivot, pivot_pos,
            );
        }

        debug_assert!(next_layer.len() + cur_layer.len() == n);

        // If all elements ended up in the current layer
        // because the pivot was the smallest one
        if next_layer.len() == 0 {
            #[cfg(feature = "partition_stats")]
            stats::partition_stats::redo_next_layer();
            if replace {
                self.pivots[layer] = old_pivot;
            } else {
                debug_assert!(layer == self.pivots.len() - 1);
                self.pivots.pop();
            }
        }

        // If we extracted all elements to the next layer
        // because the pivot was the largest one,
        // undo and try again.
        if cur_layer.len() == 0 {
            #[cfg(feature = "partition_stats")]
            stats::partition_stats::redo_cur_layer();

            std::mem::swap(cur_layer, next_layer);
            if replace {
                self.equal_buckets[layer] = false;
                self.pivots[layer] = old_pivot;
            } else {
                debug_assert!(layer == self.pivots.len() - 1);
                self.equal_buckets[layer] = false;
                self.pivots.pop();
            }
        }
    }

    #[inline(never)]
    fn multi_partition(&mut self) {
        let total_layers = self.pivots.len() + 1;

        // #[cfg(feature = "partition_stats")]
        // stats::partition_stats::partition(self.buckets[layer].len());

        // Reserve space for an additional L layers when needed.
        if total_layers + P + S::L >= self.pivots.capacity() {
            self.pivots.reserve(S::L);
        }

        while self.buckets.len() < total_layers + P - 1 {
            self.equal_buckets.push(false);
            self.buckets.push(B::default(self.free_arena));
        }

        // Alias the layers (to be split) and the next layer.
        let layer = self.pivots.len();
        let layers: &mut [B] = &mut self.buckets[layer..layer + P];

        let n = layers[0].len();

        // Sample pivots using the pivot strategy
        let num_pivots = 2usize.pow(P as u32 - 1) - 1;
        let mut pivots: [T; P] = [T::minimum(); P];
        let mut pivot_pos: [usize; P] = [0; P];

        let mut pivot_samples = vec![];

        for _ in 0..num_pivots {
            let p = RandomPivot::pick_bucket(&layers[0]);
            pivot_samples.push(p);
        }
        pivot_samples.sort_unstable_by_key(|&p| p.0);

        for i in 0..(P - 1) {
            let idx = ((num_pivots + 1) / 2usize.pow(i as u32 + 1)) - 1;
            let piv = pivot_samples[idx];
            pivots[i] = piv.0;
            pivot_pos[i] = piv.1;
            self.pivots.push(piv.0);
        }

        // Clear the next layers
        for i in 1..P {
            layers[i].clear();
        }

        PART::multi_partition::<P>(layers, &pivots[0..(P - 1)], pivot_pos);

        let mut sum = 0;
        for i in 0..P {
            sum += layers[i].len();
        }

        debug_assert!(sum == n);

        // Remove the empty layers that are created in between
        let last_layer_idx = self.pivots.len();
        for b in 0..P {
            if self.buckets[last_layer_idx - b].is_empty() && (last_layer_idx - b) > 0 {
                self.buckets.remove(last_layer_idx - b);
                self.pivots.remove(last_layer_idx - 1 - b);
            }
        }
    }

    pub fn introspect(&self) {
        println!("#buckets: {} #elements: {}", self.buckets.len(), self.size);

        println!("Pivots: {:?}", self.pivots);

        for i in 0..self.pivots.len() {
            println!(
                "Size: {}, Pivot: {:?}",
                self.buckets[i].len(),
                self.pivots[i]
            );
        }

        println!("Size: {}", self.buckets[self.pivots.len()].len());
    }
}
