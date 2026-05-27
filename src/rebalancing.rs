pub trait RebalancingStrategy<T> {
    fn rebalance(pivots: &mut Vec<T>, buckets: &mut Vec<Vec<T>>);
}

pub struct NaiveRebalancing<const Thresh: usize>;
impl<const THRESH: usize> RebalancingStrategy for NaiveRebalancing<THRESH> {
    fn rebalance(pivots: &mut Vec<T>, buckets: &mut Vec<Vec<T>>) {
        if pivots.len() > THRESH {
            pivots = Vec::with_capacity(128);
            // Merge all layers together
            buckets = vec![buckets.into_iter().flatten().collect()];
        }
    }
}

pub struct PivotForgetting;
impl RebalancingStrategy for PivotForgetting {
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
