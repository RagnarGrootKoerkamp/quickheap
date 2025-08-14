#![feature(iter_partition_in_place, portable_simd, slice_as_array, array_chunks)]
#![allow(unused)]
mod bench;
mod impls;
mod simd;
#[cfg(test)]
mod test;
use std::{array::from_fn, cmp::Reverse, iter::repeat_n, simd::u32x8};

use itertools::Itertools;
use rand::seq::SliceRandom;

#[cfg(not(feature = "u64"))]
const T_U32: bool = true;
#[cfg(not(feature = "u64"))]
type T = u32;
#[cfg(not(feature = "u64"))]
const L: usize = 8;
#[cfg(not(feature = "u64"))]
type S = std::simd::u32x8;
#[cfg(feature = "u64")]
const T_U32: bool = false;
#[cfg(feature = "u64")]
type T = u64;
#[cfg(feature = "u64")]
const L: usize = 4;
#[cfg(feature = "u64")]
type S = std::simd::u64x4;

trait Heap {
    fn default() -> Self;
    fn push(&mut self, t: T);
    fn pop(&mut self) -> Option<T>;
}

struct QuickHeap<const N: usize, const M: usize> {
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
    #[inline(always)]
    fn push(&mut self, t: T) {
        let target_layer = simd::push_position(&self.pivots, self.layer, t);

        self.buckets[target_layer].reserve(L + 1);
        self.buckets[target_layer].push(t);
    }
    #[inline(always)]
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
        next_layer.resize(n + L, 0);
        cur_layer.resize(n + L, 0);

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

        cur_layer.resize(cur_len, 0);
        next_layer.resize(next_len, 0);

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

fn main() {
    use impls::*;

    eprintln!("QUICKHEAP");
    // bench::bench::<QuickHeap<8, 3>>(false);
    bench::bench::<QuickHeap<16, 3>>(false);
    // bench::bench::<QuickHeap<32, 3>>(false);
    // bench::bench::<QuickHeap<64, 3>>(false);

    // bench::bench::<QuickHeap<16, 1>>(false); // actually slightly faster usually ??
    // bench::bench::<QuickHeap<16, 5>>(false);

    eprintln!("BASELINE");
    bench::bench::<BinaryHeap<Reverse<T>>>(false);

    eprintln!("DARY");
    // bench::bench::<dary_heap::DaryHeap<Reverse<T>, 2>>(false);
    // bench::bench::<dary_heap::DaryHeap<Reverse<T>, 4>>(false);
    bench::bench::<dary_heap::DaryHeap<Reverse<T>, 8>>(false);
    // bench::bench::<DaryHeap<(), T, 2>>(false);
    bench::bench::<DaryHeap<(), T, 4>>(false);
    // bench::bench::<DaryHeap<(), T, 8>>(false);

    eprintln!("RADIX");
    bench::bench::<RadixHeapMap<Reverse<T>, ()>>(true);

    //
    // eprintln!("BTREES");
    // bench::bench::<BTreeSet<T>>(false);
    // bench::bench::<BTreeSet<Reverse<T>>>(false);
    // bench::bench::<indexset::BTreeSet<T>>(false);
    // bench::bench::<indexset::BTreeSet<Reverse<T>>>(false);

    // eprintln!("FANCY");
    // bench::bench::<PairingHeap<(), T>>(false);
    // bench::bench::<FibonacciHeap>(false); // too slow
}
