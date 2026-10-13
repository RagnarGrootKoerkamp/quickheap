use crate::Elem;

pub struct MedianOfM<const M: usize>;
impl<const M: usize> MedianOfM<M> {
    #[inline(always)]
    pub fn pick<T: Elem>(layer: &Vec<T>) -> T {
        let n = layer.len();
        let k = M / 2;
        let mut pivots: [T; M] = std::array::from_fn(|_| {
            let pos = rand::random_range(0..n);
            layer[pos]
        });
        pivots.select_nth_unstable(k);
        pivots[k]
    }
}
