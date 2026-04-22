use crate::workloads::Elem;

pub trait PivotStrategy {
    fn pick<T: Elem>(layer: &Vec<T>) -> (T, usize);
    fn default() -> Self;
}

fn get_m_median<T: Elem>(layer: &Vec<T>, m: usize) -> (T, usize) {
    let n = layer.len();

    let mut pivots: Vec<(T, usize)> = (0..m)
        .map(|_| {
            let pos = rand::random_range(0..n);
            (layer[pos], pos)
        })
        .collect();

    pivots.sort();

    // Pivots are inclusive.
    let pivot = pivots[m / 2].0;
    let pivot_pos = pivots[m / 2].1;

    (pivot, pivot_pos)
}

pub struct MedianOfM<const M: usize>;
impl<const M: usize> PivotStrategy for MedianOfM<M> {
    fn pick<T: Elem>(layer: &Vec<T>) -> (T, usize) {
        get_m_median::<T>(layer, M)
    }

    fn default() -> Self {
        Self {}
    }
}

// pub struct PerfectPivot;
// impl<T: Elem> PivotStrategy for PerfectPivot {
//     fn pick(layer: &Vec<T>) -> (T, usize) {
//         let n = layer.len();
//         let mut sorted_layer = layer.clone();
//         sorted_layer.sort();
// 
//         let pivot_pos = (n - 1) / 2;
//         let pivot = sorted_layer[pivot_pos];
// 
//         (pivot, pivot_pos)
//     }
// }

// pub struct CbrtPivot<const A: usize, const B: usize>;
// impl<T: Elem, const A: usize, const B: usize> PivotStrategy for CbrtPivot<A, B> {
//     fn pick(layer: &Vec<T>) -> (T, usize) {
//         let n = layer.len() as f64;
//         let cbrt = n.cbrt().floor() as usize;
// 
//         let m = A * cbrt + B;
// 
//         get_m_median(layer, m)
//     }
// }

// pub struct SkewedPivot<const A: usize, const B: usize>;
// impl<T: Elem, const A: usize, const B: usize> PivotStrategy for CbrtPivot<A, B> {
//     fn pick(layer: Vec<T>) -> (T, usize) {
//         let n = layer.len() as f64;
//         let cbrt = n.cbrt().floor() as usize;

//         let m = A * cbrt + B;

//         get_m_median(layer, m)
//     }
// }
