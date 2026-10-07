use crate::{
    EqualBucketConstraints,
    buckets::{Bucket, block_arena::BlockArena},
};
use fastrand;
use std::ops::Sub;

pub trait RebalancingStrategy<T: Ord, B: Bucket<T, K, CAP>, const K: usize, const CAP: usize> {
    const MAX_REBAL_ITERATIONS: usize;
    const ALLOW_EMPTY_LAYERS: bool = false;
    fn on_pop(size: usize, global_deletions: &mut usize, pivots: &mut Vec<T>, buckets: &mut Vec<B>);
    fn on_push(
        total_size: usize,
        global_deletions: usize,
        layer: usize,
        pivots: &mut Vec<T>,
        buckets: &mut Vec<B>,
        free_arena: *mut BlockArena<T, K, CAP>,
    );
}

pub struct NoRebalancing;
impl<T: Ord, B: Bucket<T, K, CAP>, const K: usize, const CAP: usize>
    RebalancingStrategy<T, B, K, CAP> for NoRebalancing
{
    const MAX_REBAL_ITERATIONS: usize = usize::MAX;
    fn on_pop(_: usize, _: &mut usize, _: &mut Vec<T>, _: &mut Vec<B>) {}
    fn on_push(
        _: usize,
        _: usize,
        _: usize,
        _: &mut Vec<T>,
        _: &mut Vec<B>,
        _: *mut BlockArena<T, K, CAP>,
    ) {
    }
}

pub struct NaiveLogRebalancing<const THRESH: usize, const IT: usize>;
impl<
    T: Copy + Ord,
    B: Bucket<T, K, CAP>,
    const K: usize,
    const CAP: usize,
    const THRESH: usize,
    const IT: usize,
> RebalancingStrategy<T, B, K, CAP> for NaiveLogRebalancing<THRESH, IT>
{
    const MAX_REBAL_ITERATIONS: usize = IT;

    #[inline(always)]
    fn on_pop(size: usize, _: &mut usize, pivots: &mut Vec<T>, buckets: &mut Vec<B>) {
        debug_assert!(size > 0);
        let max = THRESH * size.ilog2() as usize;

        if pivots.len() > max {
            let mut flat = buckets.swap_remove(pivots.len());
            // Merge all layers together
            for bucket in buckets.drain(0..pivots.len()) {
                debug_assert!(bucket.len() > 0);
                flat.concat(bucket);
            }
            pivots.clear();

            if buckets.len() == 0 {
                buckets.push(flat);
            } else {
                buckets[0] = flat;
            }

            debug_assert!(pivots.is_empty());
        }
    }

    fn on_push(
        _: usize,
        _: usize,
        _: usize,
        _: &mut Vec<T>,
        _: &mut Vec<B>,
        _: *mut BlockArena<T, K, CAP>,
    ) {
    }
}

pub struct PivotForgetting<const F: usize, const IT: usize>;
impl<
    T: Copy + Ord,
    B: Bucket<T, K, CAP>,
    const F: usize,
    const IT: usize,
    const K: usize,
    const CAP: usize,
> RebalancingStrategy<T, B, K, CAP> for PivotForgetting<F, IT>
{
    const MAX_REBAL_ITERATIONS: usize = IT;

    #[inline(always)]
    fn on_pop(_: usize, _: &mut usize, pivots: &mut Vec<T>, buckets: &mut Vec<B>) {
        let mut total: usize = 0;
        let mut layer: usize = pivots.len();
        loop {
            if layer > 0 && buckets[layer].len() + buckets[layer - 1].len() < F * total {
                // Merge bucket with next one, forget the pivot of the layer
                if buckets[layer].len() > buckets[layer - 1].len() {
                    let old_bucket = buckets.remove(layer - 1);
                    buckets[layer - 1].concat(old_bucket);
                } else {
                    let old_bucket = buckets.remove(layer);
                    buckets[layer - 1].concat(old_bucket);
                }
                pivots.remove(layer - 1);
            } else if layer == 0 {
                break;
            } else {
                total += buckets[layer].len();
            }
            layer -= 1;
        }
    }

    fn on_push(
        _: usize,
        _: usize,
        _: usize,
        _: &mut Vec<T>,
        _: &mut Vec<B>,
        _: *mut BlockArena<T, K, CAP>,
    ) {
    }
}

pub struct ExponentialUpperBoundRebalancing {}
impl<
    T: Ord + EqualBucketConstraints + Sub<Output = T>,
    B: Bucket<T, K, CAP>,
    const K: usize,
    const CAP: usize,
> RebalancingStrategy<T, B, K, CAP> for ExponentialUpperBoundRebalancing
{
    const MAX_REBAL_ITERATIONS: usize = 1;
    const ALLOW_EMPTY_LAYERS: bool = true;

    fn on_pop(_: usize, _: &mut usize, _: &mut Vec<T>, _: &mut Vec<B>) {}

    #[inline(always)]
    fn on_push(
        _: usize,
        _: usize,
        layer: usize,
        pivots: &mut Vec<T>,
        buckets: &mut Vec<B>,
        free_arena: *mut BlockArena<T, K, CAP>,
    ) {
        let total_layers = pivots.len() + 1;
        let logical_layer = total_layers - layer - 1;
        // Exponential upper bound of layer

        // Our invariant: Layer 0 has up to size N
        //                Layer 1 has up to size 3 * N * 2^1
        //                Layer i has up to size 3 * M * 2^i

        let shift = 120.min(logical_layer + 4);
        let max_layer_size = 3 * ((2 as u128) << (shift));
        let layer_size = buckets[layer].len() as u128;

        if layer_size < max_layer_size {
            // Layer is small enough, no rebalancing necessary
            return;
        }

        // Current bucket received its 3 * 2^ith element

        debug_assert!(buckets[layer].len() > 0);
        let layer_min = buckets[layer].min().0;

        // If it is already the last layer, insert a new one
        if layer == 0 {
            buckets.insert(0, B::default(free_arena));
            buckets.swap(0, 1);
            pivots.insert(0, layer_min);
            return;
        }

        buckets.push(B::default(free_arena));
        let bucket_to_push = buckets.swap_remove(layer);

        debug_assert!(layer > 0);

        pivots[layer - 1] = layer_min;
        debug_assert!(bucket_to_push.len() > 0);

        ExponentialUpperBoundRebalancing::push_layer(layer - 1, bucket_to_push, pivots, buckets);
    }
}

impl ExponentialUpperBoundRebalancing {
    #[inline(always)]
    fn push_layer<T: Ord, B: Bucket<T, K, CAP>, const K: usize, const CAP: usize>(
        layer: usize,
        mut bucket_to_push: B,
        pivots: &mut Vec<T>,
        buckets: &mut Vec<B>,
    ) {
        debug_assert!(bucket_to_push.len() > 0);
        let bucket_max = bucket_to_push.max().0;
        let bucket_min = bucket_to_push.min().0;

        let total_layers = pivots.len();
        let locial_layer = total_layers - layer;

        let mut cur_bucket = bucket_to_push;
        pivots[layer] = bucket_min;

        // If next layer is small enough to be appended
        if buckets[layer].len() < (2 << (locial_layer + 4)) {
            buckets[layer].concat(cur_bucket);
            return;
        }

        std::mem::swap(&mut cur_bucket, &mut buckets[layer]);
        if layer > 0 {
            ExponentialUpperBoundRebalancing::push_layer(layer - 1, cur_bucket, pivots, buckets);
        } else {
            buckets.insert(0, cur_bucket);
            pivots.insert(0, bucket_max);
        }
    }
}

#[inline]
fn one_in(s: usize) -> bool {
    fastrand::usize(0..s) == 0
}

pub struct RandomizedRebalancing {}
impl<
    T: Ord + EqualBucketConstraints + Sub<Output = T>,
    B: Bucket<T, K, CAP>,
    const K: usize,
    const CAP: usize,
> RebalancingStrategy<T, B, K, CAP> for RandomizedRebalancing
{
    const MAX_REBAL_ITERATIONS: usize = 2048;

    fn on_pop(_: usize, _: &mut usize, _: &mut Vec<T>, _: &mut Vec<B>) {}

    #[inline(always)]
    fn on_push(
        total_size: usize,
        _: usize,
        layer: usize,
        pivots: &mut Vec<T>,
        buckets: &mut Vec<B>,
        _: *mut BlockArena<T, K, CAP>,
    ) {
        let mut s = total_size;
        for i in (layer.max(1)..=pivots.len()).rev() {
            if one_in(s) {
                let rest: Vec<B> = buckets.drain(1..=i).collect();
                pivots.drain(0..i);
                for b in rest {
                    if b.len() > 0 {
                        buckets[0].concat(b);
                    }
                }
                break;
            }
            s -= buckets[i].len();
        }

        if buckets[pivots.len()].len() <= 16 {
            buckets[pivots.len()].sort_decreasing();
        }
    }
}

pub struct AlphaBalancedRebalancing {}
impl<
    T: Ord + EqualBucketConstraints + Sub<Output = T>,
    B: Bucket<T, K, CAP>,
    const K: usize,
    const CAP: usize,
> RebalancingStrategy<T, B, K, CAP> for AlphaBalancedRebalancing
{
    const MAX_REBAL_ITERATIONS: usize = 1;

    fn on_pop(
        total_size: usize,
        global_deletions: &mut usize,
        pivots: &mut Vec<T>,
        buckets: &mut Vec<B>,
    ) {
        // Condition 2
        if Self::check_degeneration(*global_deletions, total_size) {
            return;
        }

        *global_deletions = 0;

        // Flatten the whole structure

        let rest: Vec<B> = buckets.drain(1..=pivots.len()).collect();
        pivots.clear();
        for b in rest {
            buckets[0].concat(b);
        }

        // TODO: Hardcoded
        if buckets[0].len() > 0 && buckets[0].len() <= 16 {
            buckets[0].sort_decreasing();
        }
    }

    fn on_push(
        total_size: usize,
        global_deletions: usize,
        layer: usize,
        pivots: &mut Vec<T>,
        buckets: &mut Vec<B>,
        _: *mut BlockArena<T, K, CAP>,
    ) {
        // Condition 1

        let len = buckets[layer].len();
        if Self::lg(len) == Self::lg(len - 1) {
            return; // no height change possible
        }

        // Calculate heights and sizes of subtrees
        let mut heights = Vec::with_capacity(buckets.len());
        let mut sizes = Vec::with_capacity(buckets.len());

        let bucket_height = ((buckets[0].len() as f64).log(Self::BETA)).ceil();
        heights.push(1f64 + bucket_height);
        sizes.push(buckets[0].len());

        for i in 1..=pivots.len() {
            let bucket_height = 1f64 + ((buckets[i].len() as f64).log(Self::BETA)).ceil();
            heights.push(1f64 + bucket_height.max(heights[i - 1]));
            sizes.push(sizes[i - 1] + buckets[i].len());
        }

        let global_log = 1f64
            + ((Self::C as f64) * ((total_size + global_deletions) as f64).log(Self::BETA)).ceil();
        if heights[pivots.len()] <= global_log {
            return;
        }

        // Violation of Condition 1
        let mut lowest_violation = heights.len();
        for i in (0..heights.len()).rev() {
            let log = 1f64 + ((Self::C as f64) * (sizes[i] as f64).log(Self::BETA)).ceil();
            if heights[i] > log {
                // Violation found
                lowest_violation = i;
            }
        }

        // No violation found
        if lowest_violation == heights.len() {
            return;
        }

        debug_assert!(lowest_violation > 0);

        // Concat the violating bucket
        let bucket = buckets.remove(lowest_violation);
        pivots.remove(lowest_violation - 1);
        buckets[lowest_violation - 1].concat(bucket);

        if lowest_violation - 1 == pivots.len() && buckets[lowest_violation - 1].len() <= 16 {
            buckets[lowest_violation - 1].sort_decreasing();
        }
    }
}

impl AlphaBalancedRebalancing {
    const B: usize = 2;
    const C: usize = 2;

    const ALPHA: f64 = 0.30;
    const BETA: f64 = 1f64 / (1f64 - Self::ALPHA);

    #[inline(always)]
    fn lg(len: usize) -> f64 {
        (len.max(1) as f64).log(Self::BETA).ceil()
    }

    #[inline(always)]
    fn check_degeneration(d: usize, n: usize) -> bool {
        (d as f64)
            < ((Self::BETA).powf((Self::B - 1) as f64 / Self::C as f64) - 1 as f64) * n as f64
    }
}

#[cfg(test)]
mod test {
    use crate::{
        ConfigurableSimdQuickHeap, EqualBucketConstraints, SimdElem,
        buckets::{
            list_block_bucket::ListBlockBucket, vec_block_bucket::VecBlockBucket,
            vec_bucket::VecBucket,
        },
        pivot_strategies::MedianOfM,
        rebalancing_strategies::{
            ExponentialUpperBoundRebalancing, NoRebalancing, PivotForgetting, RandomizedRebalancing,
        },
        simd::Avx2,
    };

    #[test]
    fn test_eub() {
        let mut h = ConfigurableSimdQuickHeap::<
            u64,
            ListBlockBucket<u64, 128, 154>,
            Avx2,
            MedianOfM<3>,
            ExponentialUpperBoundRebalancing,
            16,
            128,
            154,
            true,
            false,
        >::default();

        for i in 0..10000 {
            h.push(i);
        }

        for i in 0..10000 {
            let r = h.pop().unwrap();
            println!("i: {}, r: {}", i, r);
            assert!(r == i);
        }
    }

    #[test]
    fn test_rqh() {
        let mut h = ConfigurableSimdQuickHeap::<
            u64,
            ListBlockBucket<u64, 128, 154>,
            Avx2,
            MedianOfM<3>,
            RandomizedRebalancing,
            16,
            128,
            154,
            true,
            false,
        >::default();

        for i in (0..100000).rev() {
            h.push(i);
        }

        for i in 0..100000 {
            let r = h.pop().unwrap();
            println!("i: {}, r: {}", i, r);
            assert!(r == i);
        }
    }
}
