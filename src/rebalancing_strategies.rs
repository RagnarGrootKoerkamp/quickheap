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
            println!("Buckets length after clear: {}", buckets.len());
            buckets.push(flat_buckets);

            debug_assert!(buckets.len() == 1);
            debug_assert!(pivots.is_empty());
        }
    }
}

pub struct PivotForgetting;
impl<T: Clone> RebalancingStrategy<T> for PivotForgetting {
    fn rebalance(_: usize, pivots: &mut Vec<T>, buckets: &mut Vec<Vec<T>>) {
        // Invariant: buckets[pivots.len()] contains the smallest elements
        debug_assert!(pivots.len() > 0);

        let mut total: usize = 0;
        let mut layer: usize = pivots.len() - 1;

        loop {
            if buckets[layer].len() < total && layer > 0 {
                // Merge bucket with next one
                // Forget pivot of the layer

                let mut old_bucket = buckets[layer].clone();
                buckets[layer - 1].append(&mut old_bucket);
                buckets.remove(layer);
                pivots.remove(layer);
            } else {
                if layer == 0 {
                    break;
                }

                layer -= 1;
                total += buckets[layer].len();
            }
        }
    }
}
