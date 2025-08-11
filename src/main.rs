#![feature(iter_partition_in_place, portable_simd, slice_as_array)]
#![allow(unused)]
mod bench;
mod impls;
mod simd;
#[cfg(test)]
mod test;
use std::{array::from_fn, cmp::Reverse, iter::repeat_n, simd::u32x8};

use itertools::Itertools;
use rand::seq::SliceRandom;

type T = u32;

trait Heap {
    fn default() -> Self;
    fn push(&mut self, t: T);
    fn pop(&mut self) -> Option<T>;
}

struct BucketHeap<const N: usize, const M: usize> {
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

impl<const N: usize, const M: usize> Heap for BucketHeap<N, M> {
    fn default() -> Self {
        let mut pivots = vec![0; 128];
        pivots[0] = u32::MAX;
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
        let mut target_layer = 0;
        for &th in &self.pivots[1..1 + self.layer.next_multiple_of(8)] {
            if t < th {
                target_layer += 1;
            }
        }
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
        let min_pos = layer.iter().position_min().unwrap();
        let min = layer.swap_remove(min_pos);

        // Update the active layer.
        if layer.is_empty() && self.layer > 0 {
            self.pivots[self.layer] = 0;
            self.layer -= 1;
        }
        Some(min)
    }
}

impl<const N: usize, const M: usize> BucketHeap<N, M> {
    fn partition(&mut self) {
        // eprintln!(
        //     "Partitioning layer {} of size {}",
        //     self.layer,
        //     self.buckets[self.layer].len()
        // );

        // Reserve space for an additional 8 layers when needed.
        if self.layer + 2 == self.pivots.len() {
            self.pivots.extend(repeat_n(0, 8));
            self.buckets.extend(repeat_n(vec![], 8));
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
        next_layer.resize(n + 8, 0);
        cur_layer.resize(n + 8, 0);

        // Partition a list into two using SIMD.
        let mut cur_len = 0;
        let mut next_len = 0;
        let half = (pivot_pos + 1).next_multiple_of(8);
        for i in (0..half).step_by(8) {
            let vals = *cur_layer[i..i + 8].as_array().unwrap();
            simd::partition(
                u32x8::from_array(vals),
                n - i,
                pivot + 1,
                cur_layer,
                &mut cur_len,
                next_layer,
                &mut next_len,
            );
        }
        for i in (half..n).step_by(8) {
            let vals = *cur_layer[i..i + 8].as_array().unwrap();
            simd::partition(
                u32x8::from_array(vals),
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

    for incr in [false, true] {
        eprintln!("BUCKETHEAP");
        // bench::bench::<BucketHeap<8, 3>>(incr);
        bench::bench::<BucketHeap<16, 3>>(incr);
        bench::bench::<BucketHeap<32, 3>>(incr);
        // bench::bench::<BucketHeap<64, 3>>(incr);

        bench::bench::<BucketHeap<16, 1>>(incr);
        bench::bench::<BucketHeap<16, 5>>(incr);

        eprintln!("BASELINE");
        bench::bench::<BinaryHeap<Reverse<T>>>(incr);

        eprintln!("DARY");
        bench::bench::<dary_heap::DaryHeap<T, 2>>(incr);
        bench::bench::<dary_heap::DaryHeap<T, 4>>(incr);
        bench::bench::<dary_heap::DaryHeap<T, 8>>(incr);
        bench::bench::<DaryHeap<(), T, 2>>(incr);
        bench::bench::<DaryHeap<(), T, 4>>(incr);
        bench::bench::<DaryHeap<(), T, 8>>(incr);

        // if !incr {
        //     eprintln!("BTREES");
        //     bench::bench::<BTreeSet<T>>(incr);
        //     bench::bench::<BTreeSet<Reverse<T>>>(incr);
        //     bench::bench::<indexset::BTreeSet<T>>(incr);
        //     bench::bench::<indexset::BTreeSet<Reverse<T>>>(incr);
        // }

        // eprintln!("FANCY");
        // bench::bench::<PairingHeap<(), T>>(incr);
        // bench::bench::<FibonacciHeap>(incr); // too slow

        if incr {
            eprintln!("RADIX");
            bench::bench::<RadixHeapMap<Reverse<T>, ()>>(incr);
        }
    }
}
