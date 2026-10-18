//! # SimdQuickHeap: A fast SIMD-based priority queue
//!
//! Implementation of the `SimdQuickHeap`.
//! See also the [repo](https://github.com/ragnargrootkoerkamp/quickheap) and [the paper](https://doi.org/10.4230/LIPIcs.ESA.2026.16).
//!
//! The [`SimdQuickHeap`] is the main entrypoint and supports `u32`, `i32`, `u64`, and `i64` keys.
//!
//! If your data is larger, the [`KVSimdQuickHeap`] is slightly less efficient buts supports key-value pairs `(u32,u32)`, `(i32,i32)`, `(u64, u64)`, and `(i64,i64)`.
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
//! ## SIMD
//!
//! The Simd backend is automatically chosen based on the target architecture during compilation.
//! If needed, `SimdQuickHeap::<T, AVX2>` can be used to explicitly use the AVX2 rather than AVX512 implementation.

use std::fmt::Debug;
use std::marker::PhantomData;

#[cfg(feature = "c")]
#[doc(hidden)]
pub mod c;
#[doc(hidden)]
pub mod partitioning;
#[doc(hidden)]
pub mod pivots;
#[doc(hidden)]
pub mod simd;

#[cfg(test)]
mod test;

#[doc(hidden)]
pub use simd::SimdElem;

pub use simd::Avx2;
pub use simd::Avx512;
pub use simd::Neon;

/// The default Simd backend, decided at compile time based on the target architecture.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
pub type Simd = simd::Avx2;
/// The default Simd backend, decided at compile time based on the target architecture.
#[cfg(target_feature = "avx512f")]
pub type Simd = simd::Avx512;
/// The default Simd backend, decided at compile time based on the target architecture.
#[cfg(target_arch = "aarch64")]
pub type Simd = simd::Neon;

/// Wrapper trait for `Copy + Ord`.
pub trait Elem: Copy + Ord {}
impl<T: Copy + Ord> Elem for T {}

use crate::{partitioning::VecPartitioning, pivots::MedianOfM};

/// The key-only SimdQuickHeap implementation
///
/// - `T`: the element type.
/// - `S`: the SIMD tag: [`Avx2`] or [`Avx512`]. Default AVX-512 if available.
pub struct SimdQuickHeap<T: Elem, S: simd::SimdElem<T> = Simd> {
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
impl<T: Elem + Default, S: simd::SimdElem<T>> Default for SimdQuickHeap<T, S> {
    fn default() -> Self {
        Self {
            pivots: Vec::with_capacity(128),
            buckets: (0..128).map(|_| vec![]).collect(),
            size: 0,
            _backend: PhantomData,
        }
    }
}

impl<T: Elem + Default, S: simd::SimdElem<T>> SimdQuickHeap<T, S> {
    const N: usize = 16;

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
        let target_layer = simd::push_position::<T, S>(&self.pivots, t);

        let layer = &mut self.buckets[target_layer];

        if target_layer == self.pivots.len() && layer.len() < Self::N {
            // Count the number of larger elements in the prefix and insert the new element after them.
            let pos = layer.partition_point(|x| *x > t);
            layer.insert(pos, t);
            // TODO: SIMD
        } else {
            layer.push(t);
        }

        self.size += 1;
    }

    /// Pop the smallest element from the queue.
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
        if layer.is_empty() && self.pivots.len() > 0 {
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
        if next_layer.len() == 0 {
            self.pivots.pop();
        }

        // If we extracted all elements to the next layer
        // because the pivot was the largest one,
        // undo and try again.
        if cur_layer.len() == 0 {
            std::mem::swap(cur_layer, next_layer);

            self.pivots.pop();
        }
    }
}

pub trait IndexElem {
    fn to_usize(&self) -> usize;
}

macro_rules! impl_index_elem {
    ($t: ty) => {
        impl IndexElem for $t {
            fn to_usize(&self) -> usize {
                *self as usize
            }
        }
    };
}

impl_index_elem!(i32);
impl_index_elem!(u32);
impl_index_elem!(i64);
impl_index_elem!(u64);

/// The key-value pair SimdQuickHeap implementation
///
/// - `T`: the element type.
/// - `S`: the SIMD tag: [`Avx2`] or [`Avx512`]. Default AVX-512 if available.
pub struct KVSimdQuickHeap<T: Elem, S: simd::SimdElem<T> = Simd> {
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
    keys: Vec<Vec<T>>,
    values: Vec<Vec<T>>,
    popped_values: Vec<bool>,
    size: usize,
    _backend: PhantomData<S>,
}

impl<T: Elem + Debug + Default + IndexElem, S: simd::SimdElem<T>> KVSimdQuickHeap<T, S> {
    const N: usize = 16;

    pub fn new() -> Self {
        Self {
            pivots: Vec::with_capacity(128),
            keys: (0..128).map(|_| vec![]).collect(),
            values: (0..128).map(|_| vec![]).collect(),
            popped_values: vec![false; n],
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
        let target_layer = simd::push_position::<T, S>(&self.pivots, key);

        let key_layer = &mut self.keys[target_layer];
        let value_layer = &mut self.values[target_layer];

        if target_layer == self.pivots.len() && key_layer.len() < Self::N {
            // Count the number of larger elements in the prefix and insert the new element after them.
            let pos = key_layer.partition_point(|x| *x > key); // TODO: Is this correct??
            key_layer.insert(pos, key);
            value_layer.insert(pos, value);
            // TODO: SIMD
        } else {
            key_layer.push(key);
            value_layer.push(value);
        }

        self.size += 1;
    }

    /// Pop the smallest element from the queue.
    pub fn pop(&mut self) -> Option<(T, T)> {
        // If the total size is 0, return none
        if self.size == 0 {
            return None;
        }

        // Split the current layer as long as it is too large.
        if self.keys[self.pivots.len()].len() > Self::N {
            while self.keys[self.pivots.len()].len() > Self::N {
                self.partition();
            }

            // Sort final layer decreasing.
            let key_layer = &mut self.keys[self.pivots.len()];
            let val_layer = &mut self.values[self.pivots.len()];
            Self::sort_key_and_values(key_layer, val_layer);
        }

        // Find and extract the minimum.
        let key_layer = &mut self.keys[self.pivots.len()];
        let val_layer = &mut self.values[self.pivots.len()];
        debug_assert!(!key_layer.is_empty());

        let key = key_layer.pop().unwrap();
        let val = val_layer.pop().unwrap();

        // Update the active layer.
        if key_layer.is_empty() && self.pivots.len() > 0 {
            debug_assert!(val_layer.is_empty());

            self.pivots.pop();

            // Sort the new final layer decreasing if it's already small.
            if self.keys[self.pivots.len()].len() <= Self::N {
                let key_layer = &mut self.keys[self.pivots.len()];
                let val_layer = &mut self.values[self.pivots.len()];
                Self::sort_key_and_values(key_layer, val_layer);
            }
        }

        if self.popped_values[val.to_usize()] {
            return self.pop();
        }

        self.size -= 1;
        self.popped_values[val.to_usize()] = true;

        Some((key, val))
    }

    #[inline(never)]
    fn partition(&mut self) {
        let total_layers = self.pivots.len() + 1;

        // Reserve space for an additional L layers when needed.
        if total_layers + 2 * S::L >= self.pivots.capacity() {
            self.pivots.reserve(S::L);
        }

        if total_layers == self.keys.len() {
            debug_assert!(total_layers == self.values.len());
            self.keys.push(vec![]);
            self.values.push(vec![]);
        }

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

        self.pivots.push(pivot);

        // Clear the next layer
        next_key_layer.clear();
        next_val_layer.clear();

        VecPartitioning::<T, S>::partition_kv(
            cur_key_layer,
            cur_val_layer,
            next_key_layer,
            next_val_layer,
            pivot,
            pivot_pos,
        );

        debug_assert!(cur_key_layer.len() + next_key_layer.len() == n);
        debug_assert!(cur_val_layer.len() + next_val_layer.len() == n);

        // If all elements ended up in the current layer
        if next_key_layer.len() == 0 {
            debug_assert!(next_val_layer.len() == 0);
            self.pivots.pop();
        }

        // If we extracted all elements to the next layer
        // because the pivot was the largest one,
        // undo and try again.
        if cur_key_layer.len() == 0 {
            debug_assert!(cur_val_layer.len() == 0);
            std::mem::swap(cur_key_layer, next_key_layer);
            std::mem::swap(cur_val_layer, next_val_layer);

            self.pivots.pop();
        }
    }

    fn sort_key_and_values(keys: &mut Vec<T>, vals: &mut Vec<T>) {
        let n = keys.len();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_unstable_by_key(|&i| std::cmp::Reverse(keys[i]));

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
