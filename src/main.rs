#![feature(iter_partition_in_place, portable_simd, slice_as_array)]
#![allow(unused)]
mod bench;
mod impls;
mod simd;
#[cfg(test)]
mod test;
use std::{
    cmp::Reverse,
    collections::{BTreeSet, BinaryHeap},
    iter::repeat_n,
    simd::u32x8,
};

use fibonacci_heap::FibonacciHeap;
use orx_priority_queue::DaryHeap;
use radix_heap::RadixHeapMap;
use rand::seq::SliceRandom;

type T = u32;

trait Heap {
    fn default() -> Self;
    fn push(&mut self, t: T);
    fn pop(&mut self) -> Option<T>;
}

impl Heap for BinaryHeap<T> {
    fn default() -> Self {
        BinaryHeap::with_capacity(1 << 20)
    }
    #[inline(always)]
    fn push(&mut self, t: T) {
        self.push(t);
    }
    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        self.pop()
    }
}

impl Heap for BinaryHeap<Reverse<T>> {
    fn default() -> Self {
        BinaryHeap::with_capacity(1 << 20)
    }
    #[inline(always)]
    fn push(&mut self, t: T) {
        self.push(Reverse(t));
    }
    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        Some(self.pop()?.0)
    }
}

impl Heap for BTreeSet<T> {
    fn default() -> Self {
        <BTreeSet<T> as Default>::default()
    }
    #[inline(always)]
    fn push(&mut self, t: T) {
        self.insert(t);
    }
    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        self.first().copied()
    }
}

struct RevBTreeSet<T: Ord> {
    h: BTreeSet<Reverse<T>>,
}
impl Heap for RevBTreeSet<T> {
    fn default() -> Self {
        Self {
            h: <BTreeSet<Reverse<T>> as Default>::default(),
        }
    }
    #[inline(always)]
    fn push(&mut self, t: T) {
        self.h.insert(Reverse(t));
    }
    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        Some(self.h.last()?.0)
    }
}

struct BucketHeap<const N: usize, const M: usize> {
    layer: usize,
    // bucket i has values <= thresholds[i]
    // decreasing array
    thresholds: Vec<T>,
    buckets: Vec<Vec<T>>,
}

impl<const N: usize, const M: usize> Heap for BucketHeap<N, M> {
    fn default() -> Self {
        let mut thresholds = vec![0; 129];
        thresholds[0] = u32::MAX;
        thresholds[1] = u32::MAX;
        Self {
            layer: 1,
            thresholds,
            buckets: vec![vec![]; 129],
        }
    }
    #[inline(always)]
    fn push(&mut self, t: T) {
        let mut target_layer = 0;
        for &th in &self.thresholds[1..=self.layer.next_multiple_of(8)] {
            if t < th {
                target_layer += 1;
            }
        }
        self.buckets[target_layer].push(t);
    }
    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        if self.layer == 0 && self.buckets[self.layer].len() == 0 {
            return None;
        }
        // eprintln!("pop from {}", self.layer);
        while self.buckets[self.layer].len() > N {
            if self.layer + 2 == self.thresholds.len() {
                self.thresholds.extend(repeat_n(0, 8));
                self.buckets.extend(repeat_n(vec![], 8));
            }
            let [cur_layer, next_layer] = &mut self.buckets[self.layer..=self.layer + 1] else {
                unreachable!()
            };
            let cur_len = cur_layer.len();
            let mut ts: [T; M] = [(); M].map(|_| cur_layer[rand::random_range(0..cur_len)]);
            ts.sort();
            let new_threshold = ts[M / 2] + 1;
            self.thresholds[self.layer + 1] = new_threshold;

            next_layer.resize(cur_len + 8, 0);
            cur_layer.resize(cur_len + 8, 0);
            let mut cur_layer_idx = 0;
            let mut next_layer_idx = 0;
            for i in (0..cur_len).step_by(8) {
                let vals = *cur_layer[i..i + 8].as_array().unwrap();
                simd::partition(
                    u32x8::from_array(vals),
                    cur_len - i,
                    new_threshold,
                    cur_layer,
                    &mut cur_layer_idx,
                    next_layer,
                    &mut next_layer_idx,
                );
            }
            cur_layer.resize(cur_layer_idx, 0);
            next_layer.resize(next_layer_idx, 0);

            if cur_layer_idx == 0 {
                std::mem::swap(cur_layer, next_layer);
                continue;
            }

            self.layer += 1;
        }
        let layer = &mut self.buckets[self.layer];
        if N == 1 {
            let ans = layer.pop();
            self.layer -= 1;
            self.thresholds[self.layer + 1] = 0;
            ans
        } else {
            let (pos, ans) = layer
                .iter()
                .copied()
                .enumerate()
                .min_by_key(|(_idx, x)| *x)
                .unwrap();
            layer.swap_remove(pos);
            self.layer -= (layer.is_empty() && self.layer > 1) as usize;
            self.thresholds[self.layer + 1] = 0;
            Some(ans)
        }
    }
}

fn main() {
    eprintln!("BUCKETHEAP");
    bench::bench::<BucketHeap<32, 3>>();

    eprintln!("BASELINE");
    bench::bench::<BinaryHeap<T>>();

    // return;

    eprintln!("DARY");
    bench::bench::<dary_heap::DaryHeap<T, 2>>();
    bench::bench::<dary_heap::DaryHeap<T, 4>>();
    bench::bench::<dary_heap::DaryHeap<T, 8>>();
    bench::bench::<DaryHeap<(), T, 2>>();
    bench::bench::<DaryHeap<(), T, 4>>();
    bench::bench::<DaryHeap<(), T, 8>>();

    // eprintln!("BTREES");
    // bench::bench::<BTreeSet<T>>();
    // bench::bench::<indexset::BTreeSet<T>>();

    // SLOWER VARIANTS
    // bench::bench::<BinaryHeap<Reverse<T>>>();
    // bench::bench::<RevBTreeSet<T>>();
    // bench::bench::<BucketHeap<8, 1>>();
    // bench::bench::<BucketHeap<8, 3>>();
    // bench::bench::<BucketHeap<8, 5>>();
    // bench::bench::<BucketHeap<32, 1>>();
    // bench::bench::<BucketHeap<32, 5>>();

    // BAD
    // bench::bench::<RadixHeapMap<T, ()>>();
    // bench::bench::<pheap::PairingHeap<(), T>>();
    // bench::bench::<FibonacciHeap>();
}
