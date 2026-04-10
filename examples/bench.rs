#![feature(where_clause_attrs)]

use clap::Parser;
use quickheap::scalar_quickheap::Search;
#[cfg(feature = "avx512")]
use quickheap::simd::Avx512;
#[cfg(feature = "avx2")]
use quickheap::simd::{Avx2, SimdElem};

#[cfg(feature = "perf")]
use perfcnt::{
    AbstractPerfCounter,
    linux::{CacheId, CacheOpId, CacheOpResultId, PerfCounterBuilderLinux},
};

use quickheap::workloads::*;
use quickheap::*;
use serde::Serialize;
use std::any::type_name;
use std::cell::RefCell;
use std::{any::TypeId, sync::LazyLock};

const REPEATS: usize = 3;

#[derive(Serialize)]
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

static ARGS: LazyLock<Args> = LazyLock::new(|| Args::parse());

thread_local! {
    static CSV_WRITER: RefCell<csv::Writer<std::io::Stdout>> =
        RefCell::new(csv::Writer::from_writer(std::io::stdout()));
}

/// Runs the workload `REPEATS` times, writes each run as a CSV row, and
/// returns the median nanos (used to decide whether to skip larger `n`).
fn time_workload<T: Elem, H: Heap<T>, W: Workload>(n: u64) -> f64
where
    <H as quickheap::Heap<T>>::Casted<quickheap::workloads::CountComparisons<T>>: 'static,
{
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
                repeat,
                nanos,
                push_comparisons: 0.0,
                pop_comparisons: 0.0,
                branch_misses: 0.0,
                l1_cache_misses: 0.0,
                hw_cache_misses: 0.0,
                l3_cache_misses: 0.0,
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

        CSV_WRITER.with(|w| w.borrow_mut().serialize(&result).unwrap());

        all_nanos.push(result.nanos);
    }

    all_nanos.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_nanos[all_nanos.len() / 2]
}

fn comparisons_workload<T: Elem, H: Heap<T>, W: Workload>(n: u64) -> f64 {
    type H2<T, H> = CountingHeap<T, <H as Heap<T>>::Casted<CountComparisons<T>>>;
    let f = W::setup::<T, H2<T, H>>(n);
    H2::<T, H>::reset_comparisons();
    f();
    let (push_comparisons, pop_comparisons) = H2::<T, H>::get_comparisons();

    let result = Result {
        elem: type_name::<T>(),
        heap: type_name::<H>(),
        n,
        workload: type_name::<W>(),
        repeat: 0,
        nanos: 0.0,
        push_comparisons: push_comparisons as f64,
        pop_comparisons: pop_comparisons as f64,
        branch_misses: 0.0,
        l1_cache_misses: 0.0,
        hw_cache_misses: 0.0,
        l3_cache_misses: 0.0,
    };
    CSV_WRITER.with(|w| w.borrow_mut().serialize(&result).unwrap());
    (push_comparisons + pop_comparisons) as f64
}

pub fn bench<T: Elem, H: Heap<T>>(minpow: u32, maxpow: u32)
where
    <H as quickheap::Heap<T>>::Casted<quickheap::workloads::CountComparisons<T>>: 'static,
{
    let ns: Vec<_> = (minpow..=maxpow).map(|i| (2u64).pow(i)).collect();

    let mut ok = [true; 3];

    for n in ns {
        eprint!("{:<70} {} {n:>10}", type_name::<H>(), type_name::<T>());

        fn bench_one<T: Elem, H: Heap<T>, W: Workload>(n: u64, ok: &mut bool)
        where
            <H as quickheap::Heap<T>>::Casted<quickheap::workloads::CountComparisons<T>>: 'static,
        {
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
            if t > 100.0 {
                *ok = false;
            }
        }

        bench_one::<T, H, HeapSort>(n, &mut ok[0]);
        bench_one::<T, H, ConstantSize>(n, &mut ok[1]);
        bench_one::<T, H, MonotoneWiggle>(n, &mut ok[1]);
        if !H::MONOTONE {
            bench_one::<T, H, Wiggle>(n, &mut ok[2]);
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
    quickheap: bool,
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
    let mut minpow = args.min;
    if args.comparisons {
        minpow = args.max;
    }
    let maxpow = args.max;

    // QUICKHEAP
    if args.comparisons {
        bench::<T, scalar_quickheap::ScalarQuickHeap<T, 1, false, { Search::LinearScan }>>(
            minpow, maxpow,
        );
        bench::<T, scalar_quickheap::ScalarQuickHeap<T, 3, false, { Search::LinearScan }>>(
            minpow, maxpow,
        );
        bench::<T, scalar_quickheap::ScalarQuickHeap<T, 1, true, { Search::LinearScan }>>(
            minpow, maxpow,
        );
        bench::<T, scalar_quickheap::ScalarQuickHeap<T, 1, false>>(minpow, maxpow);
        bench::<T, scalar_quickheap::ScalarQuickHeap<T, 3, false>>(minpow, maxpow);
        bench::<T, scalar_quickheap::ScalarQuickHeap<T, 1, true>>(minpow, maxpow);
    }

    if !args.comparisons {
        #[cfg(feature = "avx2")]
        bench::<T, simd_quickheap::SimdQuickHeap<T, Avx2, 16, 1>>(minpow, maxpow);
        #[cfg(feature = "avx512")]
        bench::<T, simd_quickheap::SimdQuickHeap<T, Avx512<true>, 16, 1>>(minpow, maxpow);
    }

    // bench::<T, simd_quickheap::SimdQuickHeap<T, 8, 1>>(minpow, maxpow);
    // bench::<T, simd_quickheap::SimdQuickHeap<T, 8, 3>>(minpow, maxpow);

    if args.quickheap {
        return;
    }

    // ENGINEERED
    if !args.comparisons {
        #[cfg(feature = "ffi")]
        match TypeId::of::<T>() {
            // TODO: Figure out if we can count comparisons here.
            x if x == TypeId::of::<i32>() => {
                bench::<i32, sequence_heap::SequenceHeapI32>(minpow, maxpow);
                bench::<i32, s3q::S3qHeapI32>(minpow, maxpow.min(20));
            }
            x if x == TypeId::of::<i64>() => {
                bench::<i64, sequence_heap::SequenceHeapI64>(minpow, maxpow);
                bench::<i64, s3q::S3qHeapI64>(minpow, maxpow);
            }
            _ => unimplemented!(),
        }
    }

    // REIMPLS
    // bench::<T, binary_heap::CustomBinaryHeap<T>>(minpow, maxpow);
    // bench::<T, dary_heap::CustomDaryHeap<T, 2>>(minpow, maxpow);
    // bench::<T, dary_heap::CustomDaryHeap<T, 3>>(minpow, maxpow);
    // bench::<T, dary_heap::CustomDaryHeap<T, 4>>(minpow, maxpow);
    // bench::<T, dary_heap::CustomDaryHeap<T, 8>>(minpow, maxpow);
    // bench::<T, dary_heap::CustomDaryHeap<T, 16>>(minpow, maxpow);

    // FIXME: Investigate why this is so slow.
    // bench::<T, original_quickheap::OriginalQuickHeap<T>>(minpow, maxpow);

    // BASELINE
    bench::<T, impls::BinaryHeap<T>>(minpow, maxpow);

    // DARY
    // bench::<T, impls::DaryHeap<T, 2>>(minpow, maxpow);
    // bench::<T, impls::DaryHeap<T, 4>>(minpow, maxpow);
    // bench::<T, impls::DaryHeap<T, 8>>(minpow, maxpow);
    // bench::<T, impls::DaryHeap<T, 16>>(minpow, maxpow);
    // bench::<T, impls::OrxDaryHeap<T, 2>>(minpow, maxpow);
    bench::<T, impls::OrxDaryHeap<T, 4>>(minpow, maxpow);
    bench::<T, impls::OrxDaryHeap<T, 8>>(minpow, maxpow);
    // bench::<T, impls::OrxDaryHeap<T, 16>>(minpow, maxpow);

    // AMORTIZED
    if args.comparisons {
        bench::<T, impls::PairingHeap<T>>(minpow, maxpow);
        bench::<T, impls::FibonacciHeap<T>>(minpow, maxpow);
        bench::<T, impls::WeakHeap<T>>(minpow, maxpow);
    }

    // MONOTONE
    if !args.comparisons {
        bench::<T, impls::RadixHeap<T>>(minpow, maxpow);
    }

    // eprintln!("Set");
    // bench::<T, impls::BTreeSet<T>>(minpow, maxpow);
    // bench::<T, impls::RevBTreeSet<T>>(minpow, maxpow);
    // bench::<T, impls::IndexSetBTreeSet<T>>(minpow, maxpow);
    // bench::<T, impls::IndexSetRevBTreeSet<T>>(minpow, maxpow);
}

fn main() {
    let args = &*ARGS;
    if !args.r#i64 && !args.comparisons {
        test::<i32>(&args);
    }
    if !args.r#i32 {
        test::<i64>(&args);
    }
    CSV_WRITER.with(|w| w.borrow_mut().flush().unwrap());
}
