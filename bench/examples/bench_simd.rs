#![feature(where_clause_attrs)]

#[cfg(feature = "avx512")]
use bench::simd::Avx512;
use clap::Parser;
#[cfg(feature = "avx2")]
use quickheap::{Avx2, SimdElem, pivot_strategies::MedianOfM};

use quickheap::rebalancing_strategies::{NaiveLogRebalancing, NoRebalancing, PivotForgetting};

use bench::workloads::*;
use bench::*;
use serde::Serialize;
use std::any::type_name;
use std::cell::RefCell;
use std::sync::LazyLock;

const REPEATS: usize = 3;

#[derive(Serialize, Default)]
struct Result {
    elem: &'static str,
    heap: &'static str,
    n: u64,
    workload: &'static str,
    ops: u64,
    repeat: usize,
    nanos: f64,
}

static ARGS: LazyLock<Args> = LazyLock::new(|| Args::parse());

thread_local! {
    static CSV_WRITER: RefCell<csv::Writer<std::io::Stdout>> =
        RefCell::new(csv::Writer::from_writer(std::io::stdout()));
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
        // #[cfg(not(feature = "perf"))]
        {
            let start = std::time::Instant::now();
            f();
            let nanos = start.elapsed().as_nanos() as f64;

            result = Result {
                elem: type_name::<T>(),
                heap: type_name::<H>(),
                n,
                workload: type_name::<W>(),
                ops: n * W::NORMALIZATION,
                repeat,
                nanos,
                ..Default::default()
            };
        }

        CSV_WRITER.with(|w| w.borrow_mut().serialize(&result).unwrap());

        all_nanos.push(result.nanos);
    }

    all_nanos.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_nanos[all_nanos.len() / 2]
}

pub fn bench<T: Elem, H: Heap<T>>(minpow: u32, maxpow: u32) {
    let ns: Vec<_> = (minpow..=maxpow).map(|i| (2u64).pow(i)).collect();

    let mut ok = [true; 5];

    for n in ns {
        eprint!("{:<70} {} {n:>10}", type_name::<H>(), type_name::<T>());

        fn bench_one<T: Elem, H: Heap<T>, W: Workload>(n: u64, ok: &mut bool) {
            if !*ok {
                eprint!("{:>9}", "");
                return;
            }
            let ops = n as f64 * (n as f64).log2() * W::NORMALIZATION as f64;
            let median_nanos = time_workload::<T, H, W>(n);
            let t = median_nanos / ops;
            eprint!(" {t:>8.2}");

            // Don't test larger n for this workflow once things get too slow.
            if t > 32.0 {
                *ok = false;
            }
        }

        bench_one::<T, H, HeapSort>(n, &mut ok[0]);
        bench_one::<T, H, MonotoneConstantSize>(n, &mut ok[1]);
        bench_one::<T, H, MonotoneWiggle>(n, &mut ok[2]);
        bench_one::<T, H, RandomConstantSize>(n, &mut ok[3]);
        bench_one::<T, H, RandomWiggle>(n, &mut ok[4]);

        eprintln!();
    }
}

#[derive(clap::Parser)]
struct Args {
    #[clap(long, default_value = "20")]
    min: u32,
    #[clap(long, default_value = "25")]
    max: u32,
    #[clap(long)]
    r#i32: bool,
    #[clap(long)]
    r#i64: bool,
    #[clap(long)]
    comparisons: bool,
}

fn test<T: Elem + 'static>(args: &Args)
where
    #[cfg(feature = "avx2")]
    Avx2: SimdElem<T>,
    #[cfg(feature = "avx512")]
    Avx512<false>: SimdElem<T>,
    #[cfg(feature = "avx512")]
    Avx512<true>: SimdElem<T>,
{
    let minpow = args.min;
    let maxpow = args.max;

    // QUICKHEAP
    #[cfg(feature = "avx2")]
    {
        // Baseline
        bench::<
            T,
            quickheap::ConfigurableSimdQuickHeap<T, Avx2, MedianOfM<3>, NoRebalancing, 16, true>,
        >(minpow, maxpow);

        // Engineering 1
        bench::<
            T,
            quickheap::ConfigurableSimdQuickHeap<
                T,
                Avx2,
                MedianOfM<3>,
                NaiveLogRebalancing<3, 128>,
                16,
                true,
            >,
        >(minpow, maxpow);

        // Engineering 2
        bench::<
            T,
            quickheap::ConfigurableSimdQuickHeap<
                T,
                Avx2,
                MedianOfM<5>,
                PivotForgetting<2, 128>,
                16,
                true,
            >,
        >(minpow, maxpow);
    }

    #[cfg(feature = "avx512")]
    {
        bench::<
            T,
            quickheap::ConfigurableSimdQuickHeap<
                T,
                Avx512<true>,
                MedianOfM<3>,
                NoRebalancing,
                16,
                true,
            >,
        >(minpow, maxpow);
    }
}

fn main() {
    let args = &*ARGS;
    if !args.r#i64 {
        test::<i32>(&args);
    }
    if !args.r#i32 {
        test::<i64>(&args);
    }
    CSV_WRITER.with(|w| w.borrow_mut().flush().unwrap());
}
