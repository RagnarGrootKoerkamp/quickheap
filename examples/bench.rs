use perfcnt::linux::CacheId;
use perfcnt::linux::CacheOpId;
use perfcnt::linux::CacheOpResultId;
use perfcnt::linux::PerfCounterBuilderLinux;
use perfcnt::AbstractPerfCounter;
use quickheap::impls::NoHeap;
use quickheap::simd::SimdElem;
use quickheap::workloads::*;
use quickheap::*;
use serde::Serialize;
use std::any::type_name;
use std::any::TypeId;
use std::cell::RefCell;

const REPEATS: usize = 3;

#[derive(Serialize)]
struct Result {
    elem: &'static str,
    heap: &'static str,
    n: u64,
    workload: &'static str,
    repeat: usize,
    nanos: f64,
    comparisons: f64,
    branch_misses: f64,
    l1_cache_misses: f64,
    hw_cache_misses: f64,
    l3_cache_misses: f64,
}

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
        let mut l3_cache_misses = PerfCounterBuilderLinux::from_cache_event(
            CacheId::LL,
            CacheOpId::Read,
            CacheOpResultId::Miss,
        )
            .finish()
            .expect("Could not initialize perfcnt. Run:\necho '1' | sudo tee /proc/sys/kernel/perf_event_paranoid\n");

        branch_misses.start().unwrap();
        l1_cache_misses.start().unwrap();
        hw_cache_misses.start().unwrap();
        l3_cache_misses.start().unwrap();

        let start = std::time::Instant::now();
        f();
        let nanos = start.elapsed().as_nanos() as f64;
        branch_misses.stop().unwrap();
        l1_cache_misses.stop().unwrap();
        hw_cache_misses.stop().unwrap();
        l3_cache_misses.stop().unwrap();

        let comparisons = {
            type T2<T> = CountComparisons<T>;
            #[allow(type_alias_bounds)]
            type H2<T, H: Heap<T>> = H::Casted<T2<T>>;
            T2::<T>::reset_count();
            if TypeId::of::<H2<T, H>>() != TypeId::of::<NoHeap>() {
                let f = W::setup::<T2<T>, H2<T, H>>(n);
                T2::<T>::reset_count();
                f();
            }
            T2::<T>::get_count() as f64
        };

        let result = Result {
            elem: type_name::<T>(),
            heap: type_name::<H>(),
            n,
            workload: type_name::<W>(),
            repeat,
            nanos,
            comparisons,
            branch_misses: branch_misses.read().unwrap() as f64,
            l1_cache_misses: l1_cache_misses.read().unwrap() as f64,
            hw_cache_misses: hw_cache_misses.read().unwrap() as f64,
            l3_cache_misses: l3_cache_misses.read().unwrap() as f64,
        };
        CSV_WRITER.with(|w| w.borrow_mut().serialize(&result).unwrap());

        all_nanos.push(nanos);
    }

    all_nanos.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_nanos[all_nanos.len() / 2]
}

pub fn bench<T: Elem, H: Heap<T>>()
where
    <H as quickheap::Heap<T>>::Casted<quickheap::workloads::CountComparisons<T>>: 'static,
{
    let minpow = 10;
    let maxpow = 25;
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
            let ops = n as f64 * (n as f64).log2();
            let median_nanos = time_workload::<T, H, W>(n);
            let t = median_nanos / ops;
            eprint!(" {t:>8.2}");

            // Don't test larger n for this workflow once things get too slow.
            if t > 100.0 {
                *ok = false;
            }
        }

        bench_one::<T, H, HeapSort>(n, &mut ok[0]);
        bench_one::<T, H, ConstantSize>(n, &mut ok[1]);
        if !H::MONOTONE {
            bench_one::<T, H, Decreasing>(n, &mut ok[2]);
        }

        eprintln!();
    }
}

fn test<T: Elem + SimdElem + 'static>() {
    eprintln!("QUICKHEAP");
    bench::<T, scalar_quickheap::ScalarQuickHeap<T, 1>>();
    bench::<T, scalar_quickheap::ScalarQuickHeap<T, 3>>();

    bench::<T, simd_quickheap::SimdQuickHeap<T, 16, 1>>();
    bench::<T, simd_quickheap::SimdQuickHeap<T, 8, 3>>();

    eprintln!("Engineered");
    match TypeId::of::<T>() {
        x if x == TypeId::of::<i64>() => bench::<i64, s3q::S3qHeapI64>(),
        x if x == TypeId::of::<i32>() => bench::<i32, s3q::S3qHeapI32>(),
        _ => unimplemented!(),
    }

    eprintln!("BASELINE");
    bench::<T, impls::BinaryHeap<T>>();

    eprintln!("DARY");
    bench::<T, impls::DaryHeap<T, 2>>();
    bench::<T, impls::DaryHeap<T, 4>>();
    bench::<T, impls::DaryHeap<T, 8>>();
    bench::<T, impls::DaryHeap<T, 16>>();
    bench::<T, impls::OrxDaryHeap<T, 2>>();
    bench::<T, impls::OrxDaryHeap<T, 4>>();
    bench::<T, impls::OrxDaryHeap<T, 8>>();
    bench::<T, impls::OrxDaryHeap<T, 16>>();

    eprintln!("Amortized");
    bench::<T, impls::PairingHeap<T>>();

    if TypeId::of::<T>() == TypeId::of::<i32>() {
        bench::<i32, impls::FibonacciHeap>();
    }
    bench::<T, impls::WeakHeap<T>>();

    eprintln!("Monotone");
    bench::<T, impls::RadixHeap<T>>();

    // eprintln!("Set");
    // bench::<T, impls::BTreeSet<T>>();
    // bench::<T, impls::RevBTreeSet<T>>();
    // bench::<T, impls::IndexSetBTreeSet<T>>();
    // bench::<T, impls::IndexSetRevBTreeSet<T>>();
}

fn main() {
    test::<i32>();
    test::<i64>();
    CSV_WRITER.with(|w| w.borrow_mut().flush().unwrap());
}
