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

#[cfg(feature = "c")]
#[doc(hidden)]
pub mod c;

#[doc(hidden)]
pub mod pivot_strategies;

#[doc(hidden)]
pub mod rebalancing_strategies;

use std::{fmt::Debug, ptr};

mod simd;
#[cfg(test)]
mod test;

#[cfg(target_feature = "avx2")]
pub use simd::Avx2;
#[cfg(target_feature = "avx512f")]
pub use simd::Avx512;
#[cfg(target_arch = "aarch64")]
pub use simd::Neon;

use std::marker::PhantomData;

/// Tag to use with [`ConfigurableSimdQuickHeap`]: picks the widest available backend
/// (AVX-512 > AVX2 > NEON).
#[cfg(target_feature = "avx512f")]
pub type Simd = Avx512;

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
pub type Simd = Avx2;

#[cfg(target_arch = "aarch64")]
pub type Simd = Neon;

/// Wrapper trait for `Copy + Ord`.
#[doc(hidden)]
pub trait Elem: Copy + Ord {}
impl<T: Copy + Ord> Elem for T {}

/// The SIMD tag ([`Avx2`] or [`Avx512`]) must implement `SimdElem<T>`.
///
/// For now, this means you can only use `u32`, `i32`, `u64`, and `i64`.
pub use simd::SimdElem;

use crate::{
    buckets::{
        Bucket,
        block_arena::BlockArena,
        partitioning::{FlatPartitioning, Partition},
        vec_bucket::VecBucket,
    },
    rebalancing_strategies::{NoRebalancing, RebalancingStrategy},
};

use std::ops::Sub;

pub trait EqualBucketConstraints {
    fn one() -> Self;
    fn minimum() -> Self;
}

impl EqualBucketConstraints for i32 {
    fn one() -> Self {
        1
    }

    fn minimum() -> Self {
        i32::MIN
    }
}

impl EqualBucketConstraints for i64 {
    fn one() -> Self {
        1
    }

    fn minimum() -> Self {
        i64::MIN
    }
}

impl EqualBucketConstraints for u32 {
    fn one() -> Self {
        1
    }

    fn minimum() -> Self {
        0
    }
}

impl EqualBucketConstraints for u64 {
    fn one() -> Self {
        1
    }

    fn minimum() -> Self {
        0
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
    B: buckets::Bucket<T, K, CAP>,
    PART: Partition<T, B>,
    S: simd::SimdElem<T> = Simd,
    P: pivot_strategies::PivotStrategy = pivot_strategies::MedianOfM<3>,
    R: rebalancing_strategies::RebalancingStrategy<T, B, K, CAP> = rebalancing_strategies::NoRebalancing,
    const N: usize = 16,
    const K: usize = 128,
    const CAP: usize = 128,
    const SORT: bool = true,
    const EQUAL: bool = false,
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
    global_deletions: usize,
    #[allow(dead_code)]
    rebal_iteration: usize,
    #[allow(dead_code)]
    _p: PhantomData<P>,
    _r: PhantomData<R>,
    _part: PhantomData<PART>,
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
    VecBucket<T, 128, 128>,
    FlatPartitioning<Simd, 128, 128>,
    Simd,
    pivot_strategies::MedianOfM<3>,
    NoRebalancing,
    16,
    128,
    128,
    true,
    false,
>;

/// Return a default instance with plenty (128) layers of empty buckets.
impl<
    T: Elem + Debug + Default + Sub<Output = T> + EqualBucketConstraints,
    B: buckets::Bucket<T, K, CAP>,
    PART: Partition<T, B>,
    S: simd::SimdElem<T>,
    P: pivot_strategies::PivotStrategy,
    R: rebalancing_strategies::RebalancingStrategy<T, B, K, CAP>,
    const N: usize,
    const K: usize,
    const CAP: usize,
    const SORT: bool,
    const EQUAL: bool,
> Default for ConfigurableSimdQuickHeap<T, B, PART, S, P, R, N, K, CAP, SORT, EQUAL>
{
    fn default() -> Self {
        let free_arena = Box::into_raw(Box::from(BlockArena::<T, K, CAP>::new()));
        Self {
            pivots: Vec::with_capacity(128),
            buckets: (0..128).map(|_| B::default(free_arena)).collect(),
            equal_buckets: (0..128).map(|_| false).collect(),
            size: 0,
            global_deletions: 0,
            rebal_iteration: 0,
            free_arena,
            _p: PhantomData,
            _part: PhantomData,
            _r: PhantomData,
            _backend: PhantomData,
        }
    }
}

impl<
    T: Elem + Debug + Default + Sub<Output = T> + EqualBucketConstraints,
    B: buckets::Bucket<T, K, CAP>,
    PART: Partition<T, B>,
    S: simd::SimdElem<T>,
    R: rebalancing_strategies::RebalancingStrategy<T, B, K, CAP>,
    P: pivot_strategies::PivotStrategy,
    const N: usize,
    const K: usize,
    const CAP: usize,
    const SORT: bool,
    const EQUAL: bool,
> ConfigurableSimdQuickHeap<T, B, PART, S, P, R, N, K, CAP, SORT, EQUAL>
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
            rebal_iteration: 0,
            free_arena,
            _p: PhantomData,
            _part: PhantomData,
            _r: PhantomData,
            _backend: PhantomData,
        }
    }

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
        self.rebal_iteration += 1;
        let target_layer = simd::push_position::<T, S>(&self.pivots, t);

        let layer = &mut self.buckets[target_layer];

        if SORT && target_layer == self.pivots.len() && layer.len() < N {
            // Count the number of larger elements in the prefix and insert the new element after them.
            let pos = layer.insert_index(t);
            layer.insert(pos, t);
            // TODO: SIMD
        } else {
            layer.push(t);
        }

        self.size += 1;

        if self.rebal_iteration >= R::MAX_REBAL_ITERATIONS {
            R::on_push(
                self.size,
                self.global_deletions,
                target_layer,
                &mut self.pivots,
                &mut self.buckets,
                self.free_arena,
            );
            self.rebal_iteration = 0;
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
        self.rebal_iteration += 1;
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
            if SORT {
                // Sort final layer decreasing.
                let layer = &mut self.buckets[self.pivots.len()];
                layer.sort_decreasing();
            }
        }
        // Find and extract the minimum.
        let layer = &mut self.buckets[self.pivots.len()];
        debug_assert!(!layer.is_empty());
        let min = if SORT {
            layer.pop().unwrap()
        } else {
            let min_pos = simd::position_min_bucket::<T, S, B, K, CAP>(layer);
            layer.remove(min_pos)
        };

        // Update the active layer.
        if !R::ALLOW_EMPTY_LAYERS && layer.is_empty() && self.pivots.len() > 0 {
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

        if self.rebal_iteration <= R::MAX_REBAL_ITERATIONS {
            return Some(min);
        }
        self.rebal_iteration = 0;
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
        let total_layers = self.pivots.len() + 1;

        // Reserve space for an additional L layers when needed.
        if total_layers + 2 * S::L >= self.pivots.capacity() {
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
        let (pivot, _) = P::pick_bucket(cur_layer);

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

        // Check if the pivot is the same than the previous one
        if (layer > 0) && pivot == self.pivots[layer - 1] {
            // TODO: Implicit equal partitioning
        }

        // Clear the next layer
        next_layer.clear();

        PART::partition(cur_layer, next_layer, pivot);

        debug_assert!(next_layer.len() + cur_layer.len() == n);

        // If all elements ended up in the current layer
        if next_layer.len() == 0 {
            if replace {
                self.pivots[layer] = old_pivot;
            } else {
                self.pivots.pop();
            }
        }

        // If we extracted all elements to the next layer
        // because the pivot was the largest one,
        // undo and try again.
        if cur_layer.len() == 0 {
            std::mem::swap(cur_layer, next_layer);

            if replace {
                self.pivots[layer] = old_pivot;
            } else {
                self.pivots.pop();
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
