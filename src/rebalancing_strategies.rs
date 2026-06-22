pub trait RebalancingStrategy<T> {
    const MAX_REBAL_ITERATIONS: usize;
    fn on_pop(size: usize, pivots: &mut Vec<T>, buckets: &mut Vec<Vec<T>>);
    fn on_push(size: usize, pivots: &mut Vec<T>, buckets: &mut Vec<Vec<T>>);
}

pub struct NoRebalancing<const IT: usize>;
impl<T, const IT: usize> RebalancingStrategy<T> for NoRebalancing<IT> {
    const MAX_REBAL_ITERATIONS: usize = IT;
    fn on_pop(_: usize, _: &mut Vec<T>, _: &mut Vec<Vec<T>>) {}
    fn on_push(_: usize, _: &mut Vec<T>, _: &mut Vec<Vec<T>>) {}
}

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
}

pub struct PivotForgetting<const F: usize, const IT: usize>;
impl<T: Copy, const F: usize, const IT: usize> RebalancingStrategy<T> for PivotForgetting<F, IT> {
    const MAX_REBAL_ITERATIONS: usize = IT;
    fn on_pop(_: usize, pivots: &mut Vec<T>, buckets: &mut Vec<Vec<T>>) {
        // Invariant: buckets[pivots.len()] contains the smallest elements
        let mut total: usize = 0;
        let mut layer: usize = pivots.len();
        loop {
            if layer > 0 && buckets[layer].len() + buckets[layer - 1].len() < F * total {
                // Merge bucket with next one, forget the pivot of the layer
                if buckets[layer].len() > buckets[layer - 1].len() {
                    let old_bucket = buckets.remove(layer - 1);
                    buckets[layer - 1].extend(old_bucket);
                } else {
                    let old_bucket = buckets.remove(layer);
                    buckets[layer - 1].extend(old_bucket);
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

    fn on_push(_: usize, _: &mut Vec<T>, _: &mut Vec<Vec<T>>) {}
}
