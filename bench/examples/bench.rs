#![feature(where_clause_attrs)]

use bench::scalar_quickheap::Search;
use clap::Parser;
#[cfg(feature = "avx512")]
use quickheap::Avx512;
use quickheap::ConfigurableSimdQuickHeap;
use quickheap::pivot_strategies::{MedianOfM, RandomPivot};
#[cfg(feature = "avx2")]
use quickheap::{Avx2, SimdElem};

#[cfg(feature = "perf")]
use perfcnt::{
    AbstractPerfCounter,
    linux::{CacheId, CacheOpId, CacheOpResultId, PerfCounterBuilderLinux},
};

use bench::workloads::*;
use bench::*;
use serde::Serialize;
use std::any::type_name;
use std::cell::RefCell;
use std::{any::TypeId, sync::LazyLock};

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
    push_comparisons: f64,
    pop_comparisons: f64,
    branch_misses: f64,
    l1_cache_misses: f64,
    hw_cache_misses: f64,
    l3_cache_misses: f64,
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
        #[cfg(not(feature = "perf"))]
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

        #[cfg(feature = "perf")]
        {
            let mut branch_misses = PerfCounterBuilderLinux::from_hardware_event(perfcnt::linux::HardwareEventType::BranchMisses)
                .finish()
                .expect("Could not initialize perfcnt. Run:\necho '1' | sudo tee /proc/sys/kernel/perf_event_paranoid\n");
            let mut l1_cache_misses = PerfCounterBuilderLinux::from_cache_event(
                CacheId::L1D,
                CacheOpId::Read,
                CacheOpResultId::Miss,
            )
                .finish()
                .expect("Could not initialize perfcnt. Run:\necho '1' | sudo tee /proc/sys/kernel/perf_event_paranoid\n");
            let mut hw_cache_misses = PerfCounterBuilderLinux::from_hardware_event(
                perfcnt::linux::HardwareEventType::CacheMisses,
            )
                .finish()
                .expect("Could not initialize perfcnt. Run:\necho '1' | sudo tee /proc/sys/kernel/perf_event_paranoid\n");
            let l3_cache_misses = PerfCounterBuilderLinux::from_cache_event(
                CacheId::LL,
                CacheOpId::Read,
                CacheOpResultId::Miss,
            )
            .finish();
            if l3_cache_misses.is_err() {
                // eprintln!(
                //     "Could not initialize l3 cache miss counter; it somehow doesn't work on EPYC."
                // );
            }

            branch_misses.start().unwrap();
            l1_cache_misses.start().unwrap();
            hw_cache_misses.start().unwrap();
            let _ = l3_cache_misses.as_ref().map(|c| c.start().unwrap());

            let start = std::time::Instant::now();
            f();
            let nanos = start.elapsed().as_nanos() as f64;

            branch_misses.stop().unwrap();
            l1_cache_misses.stop().unwrap();
            hw_cache_misses.stop().unwrap();
            let _ = l3_cache_misses.as_ref().map(|c| c.stop().unwrap());

            result = Result {
                elem: type_name::<T>(),
                heap: type_name::<H>(),
                n,
                workload: type_name::<W>(),
                ops: n * W::NORMALIZATION,
                repeat,
                nanos,
                push_comparisons: 0.0,
                pop_comparisons: 0.0,
                branch_misses: branch_misses.read().unwrap() as f64,
                l1_cache_misses: l1_cache_misses.read().unwrap() as f64,
                hw_cache_misses: hw_cache_misses.read().unwrap() as f64,
                l3_cache_misses: l3_cache_misses
                    .map(|mut c| c.read().unwrap() as f64)
                    .unwrap_or_default(),
            };
        }

        CSV_WRITER.with(|w| {
            w.borrow_mut().serialize(&result).unwrap();
            w.borrow_mut().flush().unwrap();
        });

        all_nanos.push(result.nanos);
    }

    all_nanos.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_nanos[all_nanos.len() / 2]
}

fn comparisons_workload<T: Elem, H: Heap<T>, W: Workload>(n: u64) -> f64 {
    type H2<T, H> = <H as Heap<T>>::CountedHeap;
    let f = W::setup::<T, H2<T, H>>(n);
    H2::<T, H>::reset_comparisons();
    f();
    let (push_comparisons, pop_comparisons) = H2::<T, H>::get_comparisons();

    let result = Result {
        elem: type_name::<T>(),
        heap: type_name::<H>(),
        n,
        workload: type_name::<W>(),
        push_comparisons: push_comparisons as f64,
        pop_comparisons: pop_comparisons as f64,
        ..Default::default()
    };
    CSV_WRITER.with(|w| {
        w.borrow_mut().serialize(&result).unwrap();
        w.borrow_mut().flush().unwrap();
    });
    (push_comparisons + pop_comparisons) as f64
}

pub fn bench<T: Elem, H: Heap<T>>() {
    let minpow = if ARGS.comparisons { ARGS.max } else { ARGS.min };
    let maxpow = ARGS.max;
    let stride = if ARGS.table { 5 } else { 1 };
    let ns: Vec<_> = (minpow..=maxpow)
        .step_by(stride)
        .map(|i| (2u64).pow(i))
        .collect();

    let mut ok = [true; 5];

    for n in ns {
        eprint!("{:<70} {} {n:>10}", type_name::<H>(), type_name::<T>());

        fn bench_one<T: Elem, H: Heap<T>, W: Workload>(n: u64, ok: &mut bool) {
            if !*ok {
                eprint!("{:>9}", "");
                return;
            }
            let ops = n as f64 * (n as f64).log2() * W::NORMALIZATION as f64;
            let median_nanos = if ARGS.comparisons {
                comparisons_workload::<T, H, W>(n)
            } else {
                time_workload::<T, H, W>(n)
            };
            let t = median_nanos / ops;
            eprint!(" {t:>8.2}");

            // Don't test larger n for this workflow once things get too slow.
            // FIXME: Increase for table benchmark
            let cutoff = if ARGS.table { 128.0 } else { 32.0 };
            if t > cutoff {
                *ok = false;
            }
        }

        if ARGS.table {
            bench_one::<T, H, MonotoneConstantSize>(n, &mut ok[1]);
        } else {
            bench_one::<T, H, HeapSort>(n, &mut ok[0]);
            bench_one::<T, H, MonotoneConstantSize>(n, &mut ok[1]);
            bench_one::<T, H, MonotoneWiggle>(n, &mut ok[2]);
            if !H::MONOTONE {
                bench_one::<T, H, RandomWiggle>(n, &mut ok[3]);
                // bench_one::<T, H, RandomConstantSize>(n, &mut ok[4]);
            }
        }

        eprintln!();
    }
}

#[derive(clap::Parser)]
struct Args {
    #[clap(long, default_value = "10")]
    min: u32,
    #[clap(long, default_value = "25")]
    max: u32,

    #[clap(long)]
    r#i32: bool,
    #[clap(long)]
    r#i64: bool,

    /// Comparisons plot.
    #[clap(long)]
    comparisons: bool,

    /// Experiments for the table. Includes all competitors for MonotoneConstantSize benchmark at subset of sizes.
    /// Measures both time and cache misses.
    #[clap(long)]
    table: bool,

    /// SimdQuickHeap Median of 1/3/5 comparison.
    #[clap(long)]
    ablation: bool,

    /// Measure the memory (vector capacity) used by BinaryHeap and SimdQuickHeap.
    #[clap(long)]
    memory: bool,
}

fn bench_plots<T: Elem + 'static>()
where
    #[cfg(feature = "avx2")]
    Avx2: SimdElem<T>,
    #[cfg(feature = "avx512")]
    Avx512<false>: SimdElem<T>,
    #[cfg(feature = "avx512")]
    Avx512<true>: SimdElem<T>,
{
    // QUICKHEAP
    bench::<T, scalar_quickheap::ScalarQuickHeap<T, 3, false, { Search::LinearScan }>>();

    #[cfg(feature = "avx2")]
    bench::<T, ConfigurableSimdQuickHeap<T, Avx2, MedianOfM<3>>>();

    #[cfg(feature = "avx512")]
    bench::<T, ConfigurableSimdQuickHeap<T, Avx512<true>, MedianOfM<3>>>();

    // ENGINEERED
    #[cfg(feature = "ffi")]
    match TypeId::of::<T>() {
        x if x == TypeId::of::<i32>() => {
            bench::<i32, sequence_heap::SequenceHeapI32>();
            // FIXME???
            // bench::<i32, s3q::S3qHeapI32>(minpow, maxpow.min(20), false);
            bench::<i32, s3q::S3qHeapI32>();
            bench::<i32, randomized_quickheap::RandQH2HeapI32>();
        }
        x if x == TypeId::of::<i64>() => {
            bench::<i64, sequence_heap::SequenceHeapI64>();
            bench::<i64, s3q::S3qHeapI64>();
            bench::<i64, randomized_quickheap::RandQH2HeapI64>();
        }
        _ => unimplemented!(),
    }

    bench::<T, original_quickheap::OriginalQuickHeap<T>>();

    // BASELINE
    bench::<T, impls::BinaryHeap<T>>();

    // DARY
    bench::<T, impls::OrxDaryHeap<T, 8>>();

    // AMORTIZED
    bench::<T, impls::WeakHeap<T>>();

    // MONOTONE
    bench::<T, impls::RadixHeap<T>>();
}

fn bench_table() {
    type T = i64;

    // SIMD QUICKHEAP
    #[cfg(feature = "avx2")]
    bench::<T, ConfigurableSimdQuickHeap<T, Avx2, MedianOfM<3>>>();

    #[cfg(feature = "avx512")]
    bench::<T, ConfigurableSimdQuickHeap<T, Avx512<true>, MedianOfM<3>>>();

    // QUICKHEAP
    bench::<T, scalar_quickheap::ScalarQuickHeap<T, 3, false, { Search::LinearScan }>>();

    // ENGINEERED
    #[cfg(feature = "ffi")]
    {
        bench::<i64, sequence_heap::SequenceHeapI64>();
        bench::<i64, s3q::S3qHeapI64>();
        bench::<i64, randomized_quickheap::RandQH2HeapI64>();
    }

    // BOOST
    #[cfg(feature = "boost")]
    {
        bench::<i64, boost_heap::BoostDary4HeapI64>();
        bench::<i64, boost_heap::BoostFibHeapI64>();
        bench::<i64, boost_heap::BoostPairingHeapI64>();
        bench::<i64, boost_heap::BoostBinomialHeapI64>();
        bench::<i64, boost_heap::BoostSkewHeapI64>();
    }

    bench::<T, original_quickheap::OriginalQuickHeap<T>>();

    // BASELINE
    bench::<T, impls::BinaryHeap<T>>();

    // DARY
    bench::<T, impls::DaryHeap<T, 4>>();
    bench::<T, impls::DaryHeap<T, 8>>();
    // bench::<T, impls::DaryHeap<T, 16>>();
    bench::<T, impls::OrxDaryHeap<T, 4>>();
    bench::<T, impls::OrxDaryHeap<T, 8>>();
    // bench::<T, impls::OrxDaryHeap<T, 16>>();

    // AMORTIZED
    bench::<T, impls::PairingHeap<T>>();
    bench::<T, impls::FibonacciHeap<T>>();
    bench::<T, impls::WeakHeap<T>>();

    // MONOTONE
    bench::<T, impls::RadixHeap<T>>();
}

fn bench_comparisons() {
    type T = i64;

    // QUICKHEAP
    bench::<T, scalar_quickheap::ScalarQuickHeap<T, 1, false, { Search::LinearScan }>>();
    bench::<T, scalar_quickheap::ScalarQuickHeap<T, 3, false, { Search::LinearScan }>>();
    bench::<T, scalar_quickheap::ScalarQuickHeap<T, 1, true, { Search::LinearScan }>>();
    bench::<T, scalar_quickheap::ScalarQuickHeap<T, 1, false>>();
    bench::<T, scalar_quickheap::ScalarQuickHeap<T, 3, false>>();
    bench::<T, scalar_quickheap::ScalarQuickHeap<T, 1, true>>();

    // ENGINEERED
    #[cfg(feature = "ffi")]
    match TypeId::of::<T>() {
        x if x == TypeId::of::<i32>() => {
            bench::<i32, sequence_heap::SequenceHeapI32>();
            // FIXME???
            // bench::<i32, s3q::S3qHeapI32>(minpow, maxpow.min(20), false);
            bench::<i32, s3q::S3qHeapI32>();
            bench::<i32, randomized_quickheap::RandQH2HeapI32>();
        }
        x if x == TypeId::of::<i64>() => {
            bench::<i64, sequence_heap::SequenceHeapI64>();
            bench::<i64, s3q::S3qHeapI64>();
            bench::<i64, randomized_quickheap::RandQH2HeapI64>();
        }
        _ => unimplemented!(),
    }

    bench::<T, original_quickheap::OriginalQuickHeap<T>>();

    // BASELINE
    bench::<T, impls::BinaryHeap<T>>();

    // DARY
    bench::<T, impls::OrxDaryHeap<T, 8>>();

    // AMORTIZED
    bench::<T, impls::WeakHeap<T>>();
}

fn bench_ablation<T: Elem + 'static>()
where
    #[cfg(feature = "avx2")]
    Avx2: SimdElem<T>,
    #[cfg(feature = "avx512")]
    Avx512<false>: SimdElem<T>,
    #[cfg(feature = "avx512")]
    Avx512<true>: SimdElem<T>,
{
    #[cfg(feature = "avx2")]
    {
        // FIXME: DEDUP results for median of 1/3/5 ablation
        bench::<T, ConfigurableSimdQuickHeap<T, Avx2, RandomPivot>>();
        bench::<T, ConfigurableSimdQuickHeap<T, Avx2, MedianOfM<3>>>();
        bench::<T, ConfigurableSimdQuickHeap<T, Avx2, MedianOfM<5>>>();
    }

    #[cfg(feature = "avx512")]
    {
        bench::<T, ConfigurableSimdQuickHeap<T, Avx512<true>, RandomPivot>>();
        bench::<T, ConfigurableSimdQuickHeap<T, Avx512<true>, MedianOfM<3>>>();
        bench::<T, ConfigurableSimdQuickHeap<T, Avx512<true>, MedianOfM<5>>>();
    }
}

fn bench_memory() {
    type T = i64;

    fn bench<H: Heap<T>, W: Workload>() {
        eprintln!("Workload: {}", type_name::<W>());
        for exp in [10, 15, 20, 25] {
            let n = 1 << exp;

            let f = W::setup::<T, H>(n);
            let h = f();
            let cap = h.capacity();
            eprintln!(
                "{:>20}  n: {n:>10}  Cap: {cap:>10}  Ratio: {:>6.3}",
                type_name::<H>(),
                cap as f64 / n as f64
            );
        }
        eprintln!();
    }
    bench::<SimdQuickHeap<T>, HeapSort>();
    bench::<SimdQuickHeap<T>, MonotoneConstantSize>();
    bench::<SimdQuickHeap<T>, MonotoneWiggle>();
    bench::<SimdQuickHeap<T>, RandomWiggle>();
}

fn main() {
    let args = &*ARGS;

    'done: {
        if args.table {
            bench_table();
            break 'done;
        }
        if args.ablation {
            bench_ablation::<i32>();
            bench_ablation::<i64>();
            break 'done;
        }
        if args.comparisons {
            bench_comparisons();
            break 'done;
        }

        if args.memory {
            bench_memory();
            break 'done;
        }

        // The default mode, optionally filtered for only i32 or i64.
        if !args.r#i64 {
            bench_plots::<i32>();
        }
        if !args.r#i32 {
            bench_plots::<i64>();
        }
    }
    CSV_WRITER.with(|w| w.borrow_mut().flush().unwrap());
}
