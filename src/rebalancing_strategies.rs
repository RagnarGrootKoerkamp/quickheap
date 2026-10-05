// TODO: Rewrite all rebalancing strategies to use buckets
use crate::{
    EqualBucketConstraints,
    buckets::{Bucket, block_arena::BlockArena},
};

use std::ops::Sub;

pub trait RebalancingStrategy<T: Ord, B: Bucket<T, K, CAP>, const K: usize, const CAP: usize> {
    const MAX_REBAL_ITERATIONS: usize;
    const ALLOW_EMPTY_LAYERS: bool = false;
    fn on_pop(size: usize, pivots: &mut Vec<T>, buckets: &mut Vec<B>);
    fn on_push(
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
    fn on_pop(_: usize, _: &mut Vec<T>, _: &mut Vec<B>) {}
    fn on_push(_: usize, _: &mut Vec<T>, _: &mut Vec<B>, _: *mut BlockArena<T, K, CAP>) {}
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
    fn on_pop(size: usize, pivots: &mut Vec<T>, buckets: &mut Vec<B>) {
        let max = THRESH * size.ilog2() as usize;

        let mut flat = buckets.swap_remove(0);
        if pivots.len() > max {
            pivots.clear();
            // Merge all layers together
            for bucket in buckets.drain(0..) {
                flat.concat(bucket);
            }

            buckets.push(flat);

            debug_assert!(buckets.len() == 1);
            debug_assert!(pivots.is_empty());
        }
    }

    fn on_push(_: usize, _: &mut Vec<T>, _: &mut Vec<B>, _: *mut BlockArena<T, K, CAP>) {}
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
    fn on_pop(_: usize, pivots: &mut Vec<T>, buckets: &mut Vec<B>) {
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

    fn on_push(_: usize, _: &mut Vec<T>, _: &mut Vec<B>, _: *mut BlockArena<T, K, CAP>) {}
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

    fn on_pop(_: usize, _: &mut Vec<T>, _: &mut Vec<B>) {}

    fn on_push(
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
            pivots.insert(0, layer_min); // TODO: What is better?? layer_min oder layer_min - 1
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

// pub struct RandomizedRebalancing {}
// impl<T> RebalancingStrategy<T> for RandomizedRebalancing {
//     const MAX_REBAL_ITERATIONS: usize = 1024;

//     fn on_pop(size: usize, pivots: &mut Vec<T>, buckets: &mut Vec<Vec<T>>) {}

//     fn on_push(_: usize, _: &mut Vec<T>, _: &mut Vec<Vec<T>>) {}
// }

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
            ExponentialUpperBoundRebalancing, NoRebalancing, PivotForgetting,
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
}
