pub trait RebalancingStrategy<T> {
    fn rebalance(pivots: &mut Vec<T>, buckets: &mut Vec<Vec<T>>);
}

pub struct NoRebalancing;

impl<T> RebalancingStrategy<T> for NoRebalancing {
    fn rebalance(_: &mut Vec<T>, _: &mut Vec<Vec<T>>) {}
}

pub struct NaiveRebalancing<const THRESH: usize>; // TODO: Make it a function instead of a constant
impl<T: Copy, const THRESH: usize> RebalancingStrategy<T> for NaiveRebalancing<THRESH> {
    fn rebalance(pivots: &mut Vec<T>, buckets: &mut Vec<Vec<T>>) {
        if pivots.len() > THRESH {
            pivots.clear();
            // Merge all layers together
            let mut flat_buckets: Vec<T> = vec![];
            for i in 0..buckets.len() {
                let bucket = &mut buckets[i].clone();
                flat_buckets.append(bucket);
            }
            buckets.clear();
            buckets.push(flat_buckets);
        }

        debug_assert!(buckets.len() == 1);
        debug_assert!(pivots.is_empty());
    }
}

pub struct PivotForgetting;
impl<T> RebalancingStrategy<T> for PivotForgetting {
    fn rebalance(pivots: &mut Vec<T>, buckets: &mut Vec<Vec<T>>) {
        // Invariant: buckets[pivots.len()] contains the smalles elements
        let mut total: usize = 0;
        let mut layer: usize = pivots.len();
        while layer < pivots.len() {
            if buckets[layer].len() < total {
                // Merge bucket with next one
                // Forget pivot of the layer
            }

            layer += 1;
            total += buckets[layer].len();
        }
    }
}
