#[cfg(feature = "avx2")]
use quickheap::Avx2;
#[cfg(feature = "avx512")]
use quickheap::Avx512;

use quickheap::ConfigurableSimdQuickHeap as SimdQuickHeap;
use quickheap::pivot_strategies::MedianOfM;
use quickheap::rebalancing_strategies::{NaiveLogRebalancing, NoRebalancing, PivotForgetting};

use bench::workloads::*;
use bench::*;
use serde::Serialize;
use std::any::type_name;

const REPEATS: usize = 3;

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
    type T = i32;
    let n = 1 << 30; // TODO: Make it big!

    print!("rebalancing_strategy,size,num_buckets,partition_time\n");
    eprintln!("{:<130} {:>25}", "Method", "Average Operation Time");

    #[cfg(feature = "avx2")]
    {
        time_workload::<T, SimdQuickHeap<T, Avx2, MedianOfM<3>, NoRebalancing<128>, 16>, Wiggle>(n);
        time_workload::<
            T,
            SimdQuickHeap<T, Avx2, MedianOfM<3>, NaiveLogRebalancing<3, 512>, 16>,
            Wiggle,
        >(n);
        time_workload::<T, SimdQuickHeap<T, Avx2, MedianOfM<3>, PivotForgetting<2, 512>, 16>, Wiggle>(
            n,
        );

        // time_workload::<
        //     T,
        //     SimdQuickHeap<T, Avx2, MedianOfM<3>, NoRebalancing<128>, 16>,
        //     WorstCaseDescending,
        // >(n);
        // time_workload::<
        //     T,
        //     SimdQuickHeap<T, Avx2, MedianOfM<3>, NaiveLogRebalancing<3, 128>, 16>,
        //     WorstCaseDescending,
        // >(n);
        // time_workload::<
        //     T,
        //     SimdQuickHeap<T, Avx2, MedianOfM<3>, PivotForgetting<2, 128>, 16>,
        //     WorstCaseDescending,
        // >(n);
    }
    #[cfg(feature = "avx512")]
    {
        time_workload::<
            T,
            SimdQuickHeap<T, Avx512<true>, MedianOfM<1>, NoRebalancing, 16>,
            ConstantSize,
        >(n);
    }
}
