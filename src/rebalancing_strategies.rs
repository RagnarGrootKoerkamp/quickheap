pub trait RebalancingStrategy<T> {
    fn rebalance(size: usize, pivots: &mut Vec<T>, buckets: &mut Vec<Vec<T>>);
}

pub struct NoRebalancing;

impl<T> RebalancingStrategy<T> for NoRebalancing {
    fn rebalance(_: usize, _: &mut Vec<T>, _: &mut Vec<Vec<T>>) {}
}

pub struct NaiveLogRebalancing<const THRESH: usize>; // TODO: Make it a function instead of a constant
impl<T: Copy, const THRESH: usize> RebalancingStrategy<T> for NaiveLogRebalancing<THRESH> {
    fn rebalance(size: usize, pivots: &mut Vec<T>, buckets: &mut Vec<Vec<T>>) {
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
}

pub struct PivotForgetting;
impl<T: Copy> RebalancingStrategy<T> for PivotForgetting {
    fn rebalance(_: usize, pivots: &mut Vec<T>, buckets: &mut Vec<Vec<T>>) {
        // Invariant: buckets[pivots.len()] contains the smallest elements
        let mut total: usize = 0;
        let mut layer: usize = pivots.len();
        loop {
            if buckets[layer].len() < total && layer > 0 {
                // TODO: This is wrong
                // Merge bucket with next one, forget the pivot of the layer
                // let mut old_bucket = buckets[layer].clone();
                // buckets[layer - 1].append(&mut old_bucket);
                let old_bucket = buckets.remove(layer);
                buckets[layer - 1].extend(old_bucket);
                pivots.remove(layer - 1);
            } else if layer == 0 {
                break;
            } else {
                total += buckets[layer].len();
            }
            layer -= 1;
        }
    }
}
