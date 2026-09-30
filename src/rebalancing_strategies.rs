// TODO: Rewrite all rebalancing strategies to use buckets

use crate::buckets::Bucket;

pub trait RebalancingStrategy<T: Ord, B: Bucket<T, K, CAP>, const K: usize, const CAP: usize> {
    const MAX_REBAL_ITERATIONS: usize;
    fn on_pop(size: usize, pivots: &mut Vec<T>, buckets: &mut Vec<B>);
    fn on_push(layer: usize, pivots: &mut Vec<T>, buckets: &mut Vec<B>);
}

pub struct NoRebalancing;
impl<T: Ord, B: Bucket<T, K, CAP>, const K: usize, const CAP: usize>
    RebalancingStrategy<T, B, K, CAP> for NoRebalancing
{
    const MAX_REBAL_ITERATIONS: usize = usize::MAX;
    fn on_pop(_: usize, _: &mut Vec<T>, _: &mut Vec<B>) {}
    fn on_push(_: usize, _: &mut Vec<T>, _: &mut Vec<B>) {}
}

/*

pub struct NaiveLogRebalancing<const THRESH: usize, const IT: usize>;
impl<T: Copy, const THRESH: usize, const IT: usize> RebalancingStrategy<T>
    for NaiveLogRebalancing<THRESH, IT>
{
    const MAX_REBAL_ITERATIONS: usize = IT;
    fn on_pop(size: usize, pivots: &mut Vec<T>, buckets: &mut Vec<Vec<T>>) {
        let max = THRESH * size.ilog2() as usize;

        if pivots.len() > max {
            pivots.clear();
            // Merge all layers together
            let mut flat_buckets: Vec<T> = vec![];
            for i in 0..buckets.len() {
                let bucket = &mut buckets[i].clone();
                flat_buckets.append(bucket);
            }
            buckets.clear();
            buckets.push(flat_buckets);

            debug_assert!(buckets.len() == 1);
            debug_assert!(pivots.is_empty());
        }
    }

    fn on_push(_: usize, _: &mut Vec<T>, _: &mut Vec<Vec<T>>) {}
} */

// , const K: usize, const CAP: usize>

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

    fn on_push(_: usize, _: &mut Vec<T>, _: &mut Vec<B>) {}
}

// pub struct RandomizedRebalancing {}
// impl<T> RebalancingStrategy<T> for RandomizedRebalancing {
//     const MAX_REBAL_ITERATIONS: usize = 128;

//     fn on_pop(size: usize, pivots: &mut Vec<T>, buckets: &mut Vec<Vec<T>>) {}

//     fn on_push(_: usize, _: &mut Vec<T>, _: &mut Vec<Vec<T>>) {}
// }

/*
pub struct LazyRandomizedRebalancing {}
impl<T> RebalancingStrategy<T> for LazyRandomizedRebalancing {
    const MAX_REBAL_ITERATIONS: usize = 128;

    fn on_pop(_: usize, pivots: &mut Vec<T>, buckets: &mut Vec<Vec<T>>) {}

    fn on_push(_: usize, _: &mut Vec<T>, _: &mut Vec<Vec<T>>) {}
}

pub struct ExponentialUpperBoundRebalancing {}
impl<T: Copy> RebalancingStrategy<T> for ExponentialUpperBoundRebalancing {
    const MAX_REBAL_ITERATIONS: usize = 1;
    fn on_pop(_: usize, _: &mut Vec<T>, _: &mut Vec<Vec<T>>) {}

    fn on_push(layer: usize, pivots: &mut Vec<T>, buckets: &mut Bucket<Vec<T>>) {
        // Exponential upper bound of layer
        let total_layers = buckets.len();
        let max_layer_size = 3 * 2 ^ (total_layers - layer);
        let layer_size = buckets[layer].len();

        if layer_size < max_layer_size {
            // Layer is small enough, no rebalancing necessary
            return;
        }

        // If it is already the last layer, insert a new one
        if layer == 0 {
            buckets.insert(0, vec![]);
        }

        // TODO: Handle pivots correctly
        // - Track minimum of each bucket, s.t. when pushing a whole bucket, we can do pivot - 1

        // TODO: Correct to layer - 1 (smallest layer on top)

        // TODO: Create new bucket if necessary

        let mut push_bucket = Vec::<T>::new();
        std::mem::swap(&mut push_bucket, &mut buckets[layer]);

        ExponentialUpperBoundRebalancing::push_layer(layer + 1, push_bucket, pivots, buckets);
    }
}

impl ExponentialUpperBoundRebalancing {
    fn push_layer<T: Copy>(
        layer: usize,
        bucket_to_push: Vec<T>,
        pivots: &mut Vec<T>,
        buckets: &mut Vec<Vec<T>>,
    ) {
        // If next layer is small enough to be pushed
        if buckets[layer].len() < 2 ^ layer {
            buckets[layer].extend(bucket_to_push);
            // Update minimum
            return;
        }

        let mut cur_bucket = bucket_to_push;
        std::mem::swap(&mut cur_bucket, &mut buckets[layer]);

        ExponentialUpperBoundRebalancing::push_layer(layer - 1, cur_bucket, pivots, buckets);

        // TODO: Correctly do the pivots
    }
} */
