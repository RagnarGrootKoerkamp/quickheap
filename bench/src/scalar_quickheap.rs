use crate::workloads::{COMPARISONS, CountComparisons, CountingHeap, Elem};

use super::Heap;
use std::array::from_fn;

#[derive(PartialEq, Eq, std::marker::ConstParamTy)]
pub enum Search {
    LinearScan,
    BinarySearch,
}

/// A non-SIMD implementation that works for any type.
/// M: Use median of M pivots.
/// S: Strategy for finding the target bucket in push.
pub struct ScalarQuickHeap<
    T: Ord,
    const M: usize,
    const PERFECT: bool,
    const S: Search = { Search::BinarySearch },
> {
    /// A decreasing array of the pivots for all layers.
    /// buckets[i] >= pivots[i] >= buckets[i+1]
    /// Values equal to pivots[i] can be in layer i or i+1.
    /// The first layer does not have a pivot in this array.
    pub pivots: Vec<T>,
    /// The values in each layer.
    /// pivots[i-1] >= elements of buckets[i] >= pivots[i]
    pub buckets: Vec<Vec<T>>,
}

impl<T: Elem, const M: usize, const PERFECT: bool, const S: Search> Heap<T>
    for ScalarQuickHeap<T, M, PERFECT, S>
{
    type CountedHeap = CountingHeap<T, ScalarQuickHeap<CountComparisons<T>, M, PERFECT, S>>;

    fn default() -> Self {
        Self {
            pivots: vec![],
            buckets: vec![vec![]],
        }
    }
    fn push(&mut self, t: T) {
        // Push on the last layer with a pivot >= t,
        // i.e. the index of first pivot < t.
        let target_layer = match S {
            Search::BinarySearch => self
                .pivots
                .binary_search_by(|p| {
                    if *p < t {
                        std::cmp::Ordering::Greater
                    } else {
                        std::cmp::Ordering::Less
                    }
                })
                .unwrap_err(),
            Search::LinearScan => self
                .pivots
                .iter()
                .position(|p| *p < t)
                .unwrap_or(self.pivots.len()),
        };

        self.buckets[target_layer].push(t);
    }
    fn pop(&mut self) -> Option<T> {
        // Only the top layer can be empty.
        if self.buckets.len() == 1 && self.buckets[0].is_empty() {
            return None;
        }
        // Split the current layer as long as it is too large.
        while self.buckets.last().unwrap().len() > 1 {
            self.partition();
        }
        // Find and extract the minimum.
        if self.buckets.len() > 1 {
            let mut layer = self.buckets.pop().unwrap();
            self.pivots.pop().unwrap();
            let min = layer.pop().unwrap();
            assert!(layer.is_empty());
            Some(min)
        } else {
            // Preserve the first bucket.
            let layer = &mut self.buckets[0];
            let min = layer.pop().unwrap();
            assert!(layer.is_empty());
            Some(min)
        }
    }
}

impl<T: Ord + Copy, const M: usize, const PERFECT: bool, const S: Search>
    ScalarQuickHeap<T, M, PERFECT, S>
{
    #[inline(never)]
    fn partition(&mut self) {
        // Alias the current layer (to be split) and the next layer.
        let cur_layer = self.buckets.last_mut().unwrap();
        let mut next_layer = vec![];
        let n = cur_layer.len();

        let pivot;
        let pivot_pos;
        if PERFECT {
            let old_comparison_count = COMPARISONS.with(|c| c.get());
            cur_layer.sort();
            pivot_pos = (n - 1) / 2;
            pivot = cur_layer[pivot_pos];
            COMPARISONS.with(|c| c.set(old_comparison_count));
        } else {
            // Select M random pivots, and compute their median.
            let mut pivots: [(T, usize); M] = from_fn(|_| {
                let pos = rand::random_range(0..n);
                (cur_layer[pos], pos)
            });
            pivots.sort();
            // Pivots are inclusive.
            pivot = pivots[M / 2].0;
            pivot_pos = pivots[M / 2].1;
        }

        next_layer.clear();

        let buf = std::mem::take(cur_layer);

        let (first_half, second_half) = buf.split_at(pivot_pos + 1);

        for &x in first_half {
            if x <= pivot {
                next_layer.push(x);
            } else {
                cur_layer.push(x);
            }
        }

        for &x in second_half {
            if x < pivot {
                next_layer.push(x);
            } else {
                cur_layer.push(x);
            }
        }

        // If we extracted all elements to the next layer
        // because the pivot was the largest one,
        // undo and try again.
        if cur_layer.is_empty() {
            std::mem::swap(cur_layer, &mut next_layer);
            return;
        }
        self.pivots.push(pivot);
        self.buckets.push(next_layer);
    }
}
