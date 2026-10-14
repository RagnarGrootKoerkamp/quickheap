use crate::Elem;

pub struct MedianOfM<const M: usize>;
impl<const M: usize> MedianOfM<M> {
    #[inline(always)]
    pub fn pick<T: Elem>(layer: &Vec<T>) -> (T, usize) {
        let n = layer.len();
        let k = M / 2;
        let mut pivots: [(T, usize); M] = std::array::from_fn(|_| {
            let pos = rand::random_range(0..n);
            (layer[pos], pos)
        });
        pivots.select_nth_unstable(k);
        (pivots[k].0, pivots[k].1)
    }
}
