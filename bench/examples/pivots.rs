#[cfg(feature = "avx2")]
use quickheap::Avx2;
#[cfg(feature = "avx512")]
use quickheap::Avx512;

use quickheap::ConfigurableSimdQuickHeap as SimdQuickHeap;
use quickheap::pivot_strategies::{CbrtPivot, Log2Pivot, MedianOfM, TablePivot};
use quickheap::rebalancing_strategies::NoRebalancing;

use bench::workloads::*;
use bench::*;
use serde::Serialize;
use std::any::type_name;

const REPEATS: usize = 1;

#[derive(Serialize, Default, Debug)]
struct Result {
    elem: &'static str,
    heap: &'static str,
    n: u64,
    workload: &'static str,
    repeat: usize,
    nanos: f64,
    push_comparisons: f64,
    pop_comparisons: f64,
    branch_misses: f64,
    l1_cache_misses: f64,
    hw_cache_misses: f64,
    l3_cache_misses: f64,
}

/// Runs the workload `REPEATS` times, writes each run as a CSV row, and
/// returns the median nanos (used to decide whether to skip larger `n`).
fn time_workload<T: Elem, H: Heap<T>, W: Workload>(n: u64) -> f64 {
    let f = W::setup::<T, H>(n);
    f();

    let mut all_nanos = vec![];

    for repeat in 0..REPEATS {
        let f = W::setup::<T, H>(n);

        let result;
        let start = std::time::Instant::now();
        f();
        let nanos = start.elapsed().as_nanos() as f64;

        result = Result {
            elem: type_name::<T>(),
            heap: type_name::<H>(),
            n,
            workload: type_name::<W>(),
            repeat,
            nanos,
            ..Default::default()
        };

        let avg_op_time: f64 = result.nanos as f64 / (result.n * W::NORMALIZATION) as f64;

        eprintln!("{:<130} {:>25.2}", result.heap, avg_op_time);
        all_nanos.push(result.nanos);
    }

    all_nanos.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_nanos[all_nanos.len() / 2]
}

fn main() {
    type T = i64;
    let n = 1 << 21;

    println!("name,median_of_m,pivoting_time,total,alpha");
    eprintln!("{:<130} {:>25}", "Method", "Average Operation Time");

    #[cfg(feature = "avx2")]
    {
        time_workload::<
            T,
            SimdQuickHeap<T, Avx2, MedianOfM<1>, NoRebalancing, 16>,
            RandomConstantSize,
        >(n);
        time_workload::<
            T,
            SimdQuickHeap<T, Avx2, MedianOfM<3>, NoRebalancing, 16>,
            RandomConstantSize,
        >(n);
        time_workload::<
            T,
            SimdQuickHeap<T, Avx2, MedianOfM<5>, NoRebalancing, 16>,
            RandomConstantSize,
        >(n);
        time_workload::<
            T,
            SimdQuickHeap<T, Avx2, CbrtPivot<1, 0>, NoRebalancing, 16>,
            RandomConstantSize,
        >(n);
        time_workload::<
            T,
            SimdQuickHeap<T, Avx2, CbrtPivot<2, 0>, NoRebalancing, 16>,
            RandomConstantSize,
        >(n);
        time_workload::<
            T,
            SimdQuickHeap<T, Avx2, CbrtPivot<4, 0>, NoRebalancing, 16>,
            RandomConstantSize,
        >(n);
        time_workload::<
            T,
            SimdQuickHeap<T, Avx2, Log2Pivot<1, 0>, NoRebalancing, 16>,
            RandomConstantSize,
        >(n);
        time_workload::<
            T,
            SimdQuickHeap<T, Avx2, Log2Pivot<2, 0>, NoRebalancing, 16>,
            RandomConstantSize,
        >(n);
        time_workload::<
            T,
            SimdQuickHeap<T, Avx2, Log2Pivot<4, 0>, NoRebalancing, 16>,
            RandomConstantSize,
        >(n);
        time_workload::<T, SimdQuickHeap<T, Avx2, TablePivot, NoRebalancing, 16>, RandomConstantSize>(
            n,
        );
    }
    #[cfg(feature = "avx512")]
    {
        time_workload::<T, SimdQuickHeap<T, Avx512<true>, MedianOfM<1>, 16>, RandomConstantSize>(n);
    }
}
