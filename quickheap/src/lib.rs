//! # SimdQuickHeap: A fast SIMD-based priority queue
//!
//! Implementation of the `SimdQuickHeap`.
//! See also the [repo](https://github.com/ragnargrootkoerkamp/quickheap) and [the paper](https://doi.org/10.4230/LIPIcs.ESA.2026.16).
//!
//! The [`SimdQuickHeap`] is the main entrypoint and supports `u32`, `i32`, `u64`, and `i64` keys.
//!
//! If your data is larger, the [`KvSimdQuickHeap`] is slightly less efficient but supports key-value pairs `(u32,u32)`, `(i32,i32)`, `(u64, u64)`, and `(i64,i64)`.
//!
//! Create a new instance using [`SimdQuickHeap::default`].
//! [`SimdQuickHeap::push`] pushes a new value to the queue, and [`SimdQuickHeap::pop`] returns the *smallest* element in the priority queue.
//!
//! ## Example
//! ```
//! let mut q = quickheap::SimdQuickHeap::<u64>::default();
//! q.push(4);
//! q.push(1);
//! q.push(1); // Duplicate elements allowed!
//! q.push(7);
//! assert_eq!(q.pop(), Some(1)); // Returns the smallest element.
//! q.push(7);
//! q.push(3);
//! assert_eq!(q.pop(), Some(1));
//! assert_eq!(q.pop(), Some(3));
//! assert_eq!(q.pop(), Some(4));
//! assert_eq!(q.pop(), Some(7));
//! assert_eq!(q.pop(), Some(7));
//! assert_eq!(q.pop(), None);
//! q.push(2); // Pushing new minimal values is allowed.
//! ```
//!
//! With key/value pairs:
//! ```
//! let mut q = quickheap::KvSimdQuickHeap::<u64>::default();
//! q.push(1, 123);
//! q.push(2, 0);
//! assert_eq!(q.pop(), Some((1, 123)));
//! assert_eq!(q.pop(), Some((2, 0)));
//! ```
//!
//! ## SIMD
//!
//! The Simd backend is automatically chosen based on the target architecture during compilation
//! when compiling with `RUSTFLAGS="-C target-cpu=native"`.
//! If needed, `SimdQuickHeap::<T, Avx2>` explicitly selects AVX2 instead of AVX-512.

use std::fmt::Debug;
use std::marker::PhantomData;

#[cfg(feature = "c")]
#[doc(hidden)]
pub mod c;
#[doc(hidden)]
pub mod decrease_key;
#[doc(hidden)]
pub mod partitioning;
#[doc(hidden)]
pub mod pivots;
#[doc(hidden)]
pub mod simd;
#[doc(hidden)]
pub mod util;

#[cfg(test)]
mod test;

pub use simd::SimdElem;

use num_traits::PrimInt;
use std::ops::{BitAndAssign, BitOrAssign};

pub use simd::Avx2;
pub use simd::Avx512;
pub use simd::Neon;

pub trait ElemT: PartialEq + PrimInt + BitAndAssign + BitOrAssign {}
impl<T: PartialEq + PrimInt + BitAndAssign + BitOrAssign> ElemT for T {}

/* Constants for development testing */
const SIMD_INSERT: bool = false; // TODO: Test again
const LOWEST_LAYER_SIZE: usize = 16;

/// The default Simd backend, decided at compile time based on the target architecture.
#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
pub type Simd = simd::Avx2;
/// The default Simd backend, decided at compile time based on the target architecture.
#[cfg(target_feature = "avx512f")]
pub type Simd = simd::Avx512;
/// The default Simd backend, decided at compile time based on the target architecture.
#[cfg(target_arch = "aarch64")]
pub type Simd = simd::Neon;

use crate::decrease_key::DecreaseKeyStrategy;
use crate::decrease_key::NoDecreaseKey;
use crate::{partitioning::VecPartitioning, pivots::MedianOfM};

/// The key-only SimdQuickHeap implementation
///
/// - `T`: the element type: `u32`, `i32`, `u64`, or `i64`.
/// - `S`: an optional tag to select the SIMD backend: [`Avx2`], [`Avx512`], or [`Neon`].
pub struct SimdQuickHeap<T: ElemT, S: simd::SimdElem<T> = Simd> {
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
impl<T: ElemT + Default, S: simd::SimdElem<T>> Default for SimdQuickHeap<T, S> {
    fn default() -> Self {
        Self {
            pivots: Vec::with_capacity(128),
            buckets: (0..128).map(|_| vec![]).collect(),
            size: 0,
            _backend: PhantomData,
        }
    }
}

impl<T: ElemT + Default, S: simd::SimdElem<T>> SimdQuickHeap<T, S> {
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

/// The key-value pair SimdQuickHeap implementation
///
/// - `T`: the element type.
/// - `S`: an optional backend tag: [`Avx2`], [`Avx512`], or [`Neon`].
/// - `DK`: the decrease-key strategy.
///
/// The heap data lives in [`KvCore`], separate from the strategy `DK`,
/// so that the strategy can mutate the core while borrowing itself mutably.
pub struct KvSimdQuickHeap<
    T: ElemT + Debug,
    S: simd::SimdElem<T> = Simd,
    DK: DecreaseKeyStrategy<T, S> = NoDecreaseKey,
> {
    core: KvCore<T, S>,
    decrease_key: DK,
}

impl<T: ElemT + Debug + Default, S: simd::SimdElem<T>, DK: Default + DecreaseKeyStrategy<T, S>>
    Default for KvSimdQuickHeap<T, S, DK>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ElemT + Debug + Default, S: simd::SimdElem<T>, DK: Default + DecreaseKeyStrategy<T, S>>
    KvSimdQuickHeap<T, S, DK>
{
    pub fn new() -> Self {
        Self {
            core: KvCore::new(),
            decrease_key: DK::default(),
        }
    }

    /// Return the total capacity over all buckets.
    pub fn capacity(&self) -> usize {
        self.core.capacity()
    }

    /// Return the number of elements currently in the heap.
    pub fn len(&self) -> usize {
        self.core.len()
    }

    /// Return whether the heap contains no elements.
    pub fn is_empty(&self) -> bool {
        self.core.is_empty()
    }

    /// Push `t` onto the heap.
    pub fn push(&mut self, key: T, value: T) {
        self.core.push(key, value);
    }

    /// Method to decrease the key of an already inserted value
    pub fn decrease_key(&mut self, old_key: T, new_key: T, val: T) {
        self.decrease_key
            .decrease_key(&mut self.core, old_key, new_key, val);
    }

    /// Peek at the smallest element from the queue.
    pub fn peek(&mut self) -> Option<(T, T)> {
        self.core.peek(&mut self.decrease_key)
    }

    /// Pop the smallest element from the queue.
    pub fn pop(&mut self) -> Option<(T, T)> {
        loop {
            if self.core.size == 0 {
                return None;
            }
            self.core.ensure_small(&mut self.decrease_key);
            let l = self.core.pivots.len();
            self.decrease_key.before_pop(&mut self.core);
            if self.core.pivots.len() == l && !self.core.keys[l].is_empty() {
                break;
            }
        }

        let r = self.core.remove_smallest_elem();
        if let Some((k, v)) = r {
            self.decrease_key.after_pop(k, v);
        }
        self.core.size -= 1;
        r
    }
}

/// The data of the key-value pair SimdQuickHeap, without any decrease-key strategy.
///
/// - `T`: the element type.
/// - `S`: an optional backend tag: [`Avx2`], [`Avx512`], or [`Neon`].
pub struct KvCore<T: Copy + Ord + Debug, S: simd::SimdElem<T> = Simd> {
    /// A decreasing array of the pivots for all layers.
    /// buckets[i] >= pivots[i] >= buckets[i+1]
    /// Values equal to pivots[i] can be in layer i or i+1.
    /// The first layer does not have a pivot in this array.
    ///
    /// The effective number of layers is always 1 longer than `pivots`.
    ///
    /// This will have enough underlying capacity for out-of-bounds SIMD reads.
    pivots: Vec<T>,
    pivot_vals: Vec<T>,
    /// The values in each layer.
    /// pivots[i-1] >= elements of buckets[i] >= pivots[i]
    /// Values equal to pivots[i] can be in layer i or i-1.
    ///
    /// This can be longer than `layer` to reuse allocations.
    keys: Vec<Vec<T>>,
    values: Vec<Vec<T>>,
    size: usize,
    _backend: PhantomData<S>,
}

impl<T: ElemT + Debug, S: simd::SimdElem<T>> KvCore<T, S> {
    const N: usize = LOWEST_LAYER_SIZE;

    pub fn new() -> Self {
        Self {
            pivots: Vec::with_capacity(128),
            pivot_vals: Vec::with_capacity(128),
            keys: (0..128).map(|_| vec![]).collect(),
            values: (0..128).map(|_| vec![]).collect(),
            size: 0,
            _backend: PhantomData,
        }
    }

    /// Return the total capacity over all buckets.
    pub fn capacity(&self) -> usize {
        let key_cap: usize = self.keys.iter().map(|b| b.capacity()).sum();
        let val_cap: usize = self.values.iter().map(|b| b.capacity()).sum();
        key_cap + val_cap
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
    pub fn push(&mut self, key: T, value: T) {
        self.insert_elem(key, value);
        self.size += 1;
    }

    /// Remove top layer and sort if necessary
    pub(crate) fn pop_layer(&mut self) {
        self.pivots.pop();
        self.pivot_vals.pop();
        let l = self.pivots.len();
        if self.keys[l].len() <= Self::N {
            Self::sort_key_and_values(&mut self.keys[l], &mut self.values[l]);
        }
    }

    #[inline(always)]
    fn layer_of(&self, key: T, val: T) -> usize {
        // pivots with p > key (SIMD, as before)
        let mut l = unsafe { simd::push_position::<T, S>(&self.pivots, key) };
        // ties on the key: pair (key,val) < (pivot,pivot_val) -> lower layer
        while l < self.pivots.len() && self.pivots[l] == key && val < self.pivot_vals[l] {
            l += 1;
        }
        l
    }

    /// Method to insert a helper element without increasing the heaps size
    fn insert_elem(&mut self, key: T, value: T) {
        // let target_layer = unsafe { simd::push_position::<T, S>(&self.pivots, key) };
        let target_layer = self.layer_of(key, value);

        let key_layer = &mut self.keys[target_layer];
        let value_layer = &mut self.values[target_layer];

        if target_layer == self.pivots.len() && key_layer.len() < Self::N {
            // Count the number of larger elements in the prefix and insert the new element after them.

            let pos: usize;
            if SIMD_INSERT {
                pos = simd::insert_pos::<T, S>(key_layer, key);
                debug_assert!(pos == key_layer.partition_point(|x| *x > key));
            } else {
                pos = key_layer.partition_point(|x| *x > key);
            }

            key_layer.insert(pos, key);
            value_layer.insert(pos, value);
        } else {
            key_layer.push(key);
            value_layer.push(value);
        }
    }

    /// Peek at the smallest element from the queue.
    pub fn peek<DK: Default + DecreaseKeyStrategy<T, S>>(
        &mut self,
        strat: &mut DK,
    ) -> Option<(T, T)> {
        // If the total size is 0, return none
        if self.size == 0 {
            return None;
        }

        self.ensure_small(strat);

        // Find and extract the minimum.
        let key_layer = &mut self.keys[self.pivots.len()];
        let val_layer = &mut self.values[self.pivots.len()];
        debug_assert!(!key_layer.is_empty());

        let idx = key_layer.len() - 1;
        let key = key_layer[idx];
        let val = val_layer[idx];

        Some((key, val))
    }

    #[inline(always)]
    fn ensure_small<DK: Default + DecreaseKeyStrategy<T, S>>(&mut self, strat: &mut DK) {
        // Split the current layer as long as it is too large.
        if self.keys[self.pivots.len()].len() > Self::N {
            while self.keys[self.pivots.len()].len() > Self::N {
                self.partition(strat);
            }

            // Sort final layer decreasing.
            let key_layer = &mut self.keys[self.pivots.len()];
            let val_layer = &mut self.values[self.pivots.len()];
            KvCore::<T, S>::sort_key_and_values(key_layer, val_layer);
        }
    }

    fn remove_smallest_elem(&mut self) -> Option<(T, T)> {
        // Find and extract the minimum.
        let key_layer = &mut self.keys[self.pivots.len()];
        let val_layer = &mut self.values[self.pivots.len()];
        debug_assert!(!key_layer.is_empty());

        let key = key_layer.pop().unwrap();
        let val = val_layer.pop().unwrap();

        // Update the active layer.
        if key_layer.is_empty() && !self.pivots.is_empty() {
            debug_assert!(val_layer.is_empty());

            self.pivots.pop();
            self.pivot_vals.pop();

            // Sort the new final layer decreasing if it's already small.
            if self.keys[self.pivots.len()].len() <= Self::N {
                let key_layer = &mut self.keys[self.pivots.len()];
                let val_layer = &mut self.values[self.pivots.len()];
                Self::sort_key_and_values(key_layer, val_layer);
            }
        }

        Some((key, val))
    }

    #[inline(never)]
    fn partition<DK: Default + DecreaseKeyStrategy<T, S>>(&mut self, strat: &mut DK) {
        strat.before_partition(self);
        if self.keys[self.pivots.len()].len() <= Self::N {
            return;
        }

        let total_layers = self.pivots.len() + 1;

        // Reserve space for an additional L layers when needed.
        if total_layers + 2 * S::L >= self.pivots.capacity() {
            self.pivots.reserve(S::L);
        }

        if total_layers == self.keys.len() {
            debug_assert!(total_layers == self.values.len());
            self.keys.push(vec![]);
            self.values.push(vec![]);
            strat.layer_push();
        }

        let layer = self.pivots.len();

        // Alias the current layer (to be split) and the next layer.
        let [cur_key_layer, next_key_layer] =
            &mut self.keys[self.pivots.len()..=self.pivots.len() + 1]
        else {
            unreachable!()
        };

        let [cur_val_layer, next_val_layer] =
            &mut self.values[self.pivots.len()..=self.pivots.len() + 1]
        else {
            unreachable!()
        };

        let n = cur_key_layer.len();
        debug_assert!(cur_val_layer.len() == n);

        // Sample a pivot using the pivot strategy
        let (pivot, pivot_pos) = MedianOfM::<3>::pick(cur_key_layer);
        let pivot_val = cur_val_layer[pivot_pos];

        self.pivots.push(pivot);
        self.pivot_vals.push(pivot_val);

        // Clear the next layer
        next_key_layer.clear();
        next_val_layer.clear();

        VecPartitioning::<T, S>::partition_kv(
            cur_key_layer,
            cur_val_layer,
            next_key_layer,
            next_val_layer,
            pivot,
            pivot_val,
            pivot_pos,
        );

        debug_assert!(cur_key_layer.len() + next_key_layer.len() == n);
        debug_assert!(cur_val_layer.len() + next_val_layer.len() == n);

        // If all elements ended up in the current layer
        if next_key_layer.is_empty() {
            debug_assert!(next_val_layer.is_empty());
            self.pivots.pop();
            self.pivot_vals.pop();
            return;
        }

        // If we extracted all elements to the next layer
        // because the pivot was the largest one,
        // undo and try again.
        if cur_key_layer.is_empty() {
            debug_assert!(cur_val_layer.is_empty());
            std::mem::swap(cur_key_layer, next_key_layer);
            std::mem::swap(cur_val_layer, next_val_layer);

            self.pivots.pop();
            self.pivot_vals.pop();
            return;
        }

        strat.after_partition(layer, pivot, pivot_val, pivot_pos);
    }

    fn sort_key_and_values(keys: &mut [T], vals: &mut [T]) {
        let n = keys.len();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_unstable_by_key(|&i| std::cmp::Reverse((keys[i], vals[i])));

        for i in 0..n {
            if idx[i] == i {
                continue;
            }
            let (tk, tv) = (keys[i], vals[i]);
            let mut cur = i;
            loop {
                let next = idx[cur];
                idx[cur] = cur; // mark as done
                if next == i {
                    keys[cur] = tk;
                    vals[cur] = tv;
                    break;
                }
                keys[cur] = keys[next];
                vals[cur] = vals[next];
                cur = next;
            }
        }
    }
}
