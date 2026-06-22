use crate::Elem;

pub trait PivotStrategy {
    fn pick<T: Elem>(layer: &Vec<T>) -> (T, usize);
}

fn get_m_median<T: Elem>(layer: &Vec<T>, mut m: usize) -> (T, usize) {
    #[cfg(feature = "pivots")]
    print!("{},", m);

    if m % 2 == 0 {
        m += 1;
    }
    let n = layer.len();
    let k: usize = m / 2;

    let mut pivots: Vec<(T, usize)> = (0..m)
        .map(|_| {
            let pos = rand::random_range(0..n);
            (layer[pos], pos)
        })
        .collect();

    pivots.select_nth_unstable(k);
    let pivot_pos = pivots[k].1;
    let pivot = pivots[k].0;

    (pivot, pivot_pos)
}

fn get_median<T: Elem, const M: usize>(layer: &Vec<T>) -> (T, usize) {
    assert!(M % 2 == 1, "M must be odd");
    #[cfg(feature = "pivots")]
    print!("{},", M);
    let n = layer.len();
    let k: usize = M / 2;

    let mut pivots: [(T, usize); M] = std::array::from_fn(|_| {
        let pos = rand::random_range(0..n);
        (layer[pos], pos)
    });

    pivots.select_nth_unstable(k);
    let pivot_pos = pivots[k].1;
    let pivot = pivots[k].0;

    (pivot, pivot_pos)
}

pub struct MedianOfM<const M: usize>;
impl<const M: usize> PivotStrategy for MedianOfM<M> {
    fn pick<T: Elem>(layer: &Vec<T>) -> (T, usize) {
        get_median::<T, M>(layer)
    }
}

pub struct RandomPivot;
impl PivotStrategy for RandomPivot {
    fn pick<T: Elem>(layer: &Vec<T>) -> (T, usize) {
        let n = layer.len();
        let pivot_pos = rand::random_range(0..n);
        let pivot = layer[pivot_pos];
        (pivot, pivot_pos)
    }
}

pub struct CbrtPivot<const A: usize, const B: usize>;

impl<const A: usize, const B: usize> CbrtPivot<A, B> {
    const CBRT_LOOKUP: [usize; 32] = [
        1, 1, 1, 2, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 101, 128, 161, 203,
        256, 322, 406, 512, 645, 812, 1024, 1290,
    ];
}

impl<const A: usize, const B: usize> PivotStrategy for CbrtPivot<A, B> {
    fn pick<T: Elem>(layer: &Vec<T>) -> (T, usize) {
        let n = layer.len();
        let idx = size_of::<T>() * 8 - n.leading_zeros() as usize;

        let cbrt = CbrtPivot::<A, B>::CBRT_LOOKUP[idx];
        let fac: f64 = 1 as f64 / A as f64;

        let m = (fac * cbrt as f64) as usize + B;

        get_m_median(layer, m)
    }
}

pub struct Log2Pivot<const A: usize, const B: usize>;
impl<const A: usize, const B: usize> PivotStrategy for Log2Pivot<A, B> {
    fn pick<T: Elem>(layer: &Vec<T>) -> (T, usize) {
        let n = layer.len();
        let idx = size_of::<T>() * 8 - n.leading_zeros() as usize;
        let m = A * idx + B;

        get_m_median(layer, m)
    }
}

pub struct TablePivot;
impl TablePivot {
    const LOOKUP: [usize; 32] = [
        1, 1, 1, 3, 3, 3, 5, 5, 5, 7, 7, 9, 9, 11, 13, 15, 15, 17, 17, 21, 21, 23, 23, 23, 27, 31,
        37, 37, 41, 41, 45, 45,
    ];
}

impl PivotStrategy for TablePivot {
    fn pick<T: Elem>(layer: &Vec<T>) -> (T, usize) {
        let i = size_of::<T>() * 8 - layer.len().leading_zeros() as usize;
        if i > 31 {
            return get_m_median(layer, 47);
        }
        get_m_median(layer, TablePivot::LOOKUP[i])
    }
}
