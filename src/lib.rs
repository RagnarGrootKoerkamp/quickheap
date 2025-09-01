#![feature(iter_partition_in_place, portable_simd, slice_as_array)]
#![allow(unused)]
mod impls;
mod simd;
use std::{array::from_fn, cmp::Reverse, iter::repeat_n, simd::u32x8};

use itertools::Itertools;
use rand::seq::SliceRandom;

#[cfg(not(feature = "u64"))]
pub const T_U32: bool = true;
#[cfg(not(feature = "u64"))]
pub type T = u32;
#[cfg(not(feature = "u64"))]
const L: usize = 8;
#[cfg(not(feature = "u64"))]
type S = std::simd::u32x8;

#[cfg(feature = "u64")]
pub const T_U32: bool = false;
#[cfg(feature = "u64")]
pub type T = u64;
#[cfg(feature = "u64")]
const L: usize = 4;
#[cfg(feature = "u64")]
type S = std::simd::u64x4;

pub trait Heap {
    fn default() -> Self;
    fn push(&mut self, t: T);
    fn pop(&mut self) -> Option<T>;
}

pub struct QuickHeap<const N: usize, const M: usize> {
    /// The number of layers in the tree.
    layer: usize,
    /// A decreasing array of the pivots for all layers.
    /// pivots[0] = u32::MAX
    pivots: Vec<T>,
    /// The values in each layer.
    /// pivots[i+1] <= elements of buckets[i] <= pivots[i]
    /// Values equal to pivots[i] can be in layer i or i-1.
    buckets: Vec<Vec<T>>,
}

impl<const N: usize, const M: usize> Heap for QuickHeap<N, M> {
    fn default() -> Self {
        let mut pivots = vec![0; 128];
        pivots[0] = T::MAX;
        // pivots[1] = u32::MAX;
        Self {
            // layer: 1,
            layer: 0,
            pivots,
            buckets: vec![vec![]; 128],
        }
    }
    #[inline(never)]
    fn push(&mut self, t: T) {
        let target_layer = simd::push_position(&self.pivots, self.layer, t);

        self.buckets[target_layer].reserve(L + 1);
        self.buckets[target_layer].push(t);
    }
    #[inline(never)]
    fn pop(&mut self) -> Option<T> {
        // Only the top layer can be empty.
        if self.buckets[self.layer].len() == 0 {
            return None;
        }
        // Split the current layer as long as it is too large.
        while self.buckets[self.layer].len() > N {
            self.partition();
        }
        // Find and extract the minimum.
        let layer = &mut self.buckets[self.layer];
        let min_pos = simd::position_min(layer);
        let min = layer.swap_remove(min_pos);

        // Update the active layer.
        if layer.is_empty() && self.layer > 0 {
            self.pivots[self.layer] = 0;
            self.layer -= 1;
        }
        Some(min)
    }
}

impl<const N: usize, const M: usize> QuickHeap<N, M> {
    #[inline(never)]
    fn partition(&mut self) {
        // eprintln!(
        //     "Partitioning layer {} of size {}",
        //     self.layer,
        //     self.buckets[self.layer].len()
        // );

        // Reserve space for an additional L layers when needed.
        if self.layer + 2 == self.pivots.len() {
            self.pivots.extend(repeat_n(0, L));
            self.buckets.extend(repeat_n(vec![], L));
        }
        // Alias the current layer (to be split) and the next layer.
        let [cur_layer, next_layer] = &mut self.buckets[self.layer..=self.layer + 1] else {
            unreachable!()
        };
        let n = cur_layer.len();

        // Select 3 random pivots, and compute their median.
        let mut pivots: [(T, usize); M] = from_fn(|_| {
            let pos = rand::random_range(0..n);
            (cur_layer[pos], pos)
        });
        pivots.sort();
        // Pivots are exclusive.
        let pivot = pivots[M / 2].0;
        let pivot_pos = pivots[M / 2].1;
        // eprintln!("Pivot: {pivot} at {pivot_pos}");
        // +1, so we can use 0 to indicate empty layers.
        self.pivots[self.layer + 1] = pivot + 1;

        // Reserve space in the next layer,
        // and make sure the current layer can hold a spare SIMD register.
        cur_layer.reserve(L);
        next_layer.clear();
        next_layer.reserve(n + L);

        unsafe { cur_layer.set_len(n + L) };
        unsafe { next_layer.set_len(n + L) };

        // Partition a list into two using SIMD.
        let mut cur_len = 0;
        let mut next_len = 0;
        let half = (pivot_pos + 1).next_multiple_of(L);
        for i in (0..half).step_by(L) {
            let vals = *cur_layer[i..i + L].as_array().unwrap();
            simd::partition(
                S::from_array(vals),
                n - i,
                pivot + 1,
                cur_layer,
                &mut cur_len,
                next_layer,
                &mut next_len,
            );
        }
        for i in (half..n).step_by(L) {
            let vals = *cur_layer[i..i + L].as_array().unwrap();
            simd::partition(
                S::from_array(vals),
                n - i,
                pivot,
                cur_layer,
                &mut cur_len,
                next_layer,
                &mut next_len,
            );
        }
        // eprintln!("cur len: {cur_len}, next len: {next_len}");
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
            return;
        }

        // Increment the active layer.
        self.layer += 1;
    }
}
