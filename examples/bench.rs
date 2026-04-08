use perfcnt::linux::CacheId;
use perfcnt::linux::CacheOpId;
use perfcnt::linux::CacheOpResultId;
use perfcnt::linux::PerfCounterBuilderLinux;
use perfcnt::AbstractPerfCounter;
use quickheap::impls::NoHeap;
use quickheap::workloads::*;
use quickheap::*;
use std::any::type_name;
use std::any::TypeId;

const REPEATS: usize = 3;

struct Result {
    nanos: f64,
    branch_misses: f64,
    l1_cache_misses: f64,
    hw_cache_misses: f64,
    l3_cache_misses: f64,
}

fn time_workload<T: Elem, H: Heap<T>, W: Workload>(n: u64) -> Result {
    let mut nanos = vec![];
    let mut bm = vec![];
    let mut l1 = vec![];
    let mut hw = vec![];
    let mut l3 = vec![];
    for _ in 0..REPEATS {
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
        nanos.push(start.elapsed().as_nanos() as f64);
        branch_misses.stop().unwrap();
        l1_cache_misses.stop().unwrap();
        hw_cache_misses.stop().unwrap();
        l3_cache_misses.stop().unwrap();

        bm.push(branch_misses.read().unwrap() as f64);
        l1.push(l1_cache_misses.read().unwrap() as f64);
        hw.push(hw_cache_misses.read().unwrap() as f64);
        l3.push(l3_cache_misses.read().unwrap() as f64);
    }
    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2] as f64
    };
    Result {
        nanos: median(nanos),
        branch_misses: median(bm),
        l1_cache_misses: median(l1),
        hw_cache_misses: median(hw),
        l3_cache_misses: median(l3),
    }
}

pub fn bench<T: Elem, H: Heap<T>>(increasing: bool)
where
    <H as quickheap::Heap<T>>::Casted<quickheap::workloads::CountComparisons<T>>: 'static,
{
    let minpow = 20;
    let maxpow = 20;
    let ns: Vec<_> = (minpow..=maxpow).map(|i| (2u64).pow(i)).collect();

    type T2<T> = CountComparisons<T>;
    #[allow(type_alias_bounds)]
    type H2<T, H: Heap<T>> = H::Casted<T2<T>>;

    let mut ok = [true; 3];

    for n in ns {
        eprint!("{:<70} {} {n:>10}", type_name::<H>(), type_name::<T>());
        print!("{}\t{}\t{n}", type_name::<H>(), type_name::<T>());

        fn bench_one<T: Elem, H: Heap<T>, W: Workload>(n: u64, ok: &mut bool)
        where
            <H as quickheap::Heap<T>>::Casted<quickheap::workloads::CountComparisons<T>>: 'static,
        {
            if !*ok {
                eprint!("{:>9}", "");
                print!("\t");
                return;
            }
            let ops = n as f64 * (n as f64).log2();
            let t = time_workload::<T, H, W>(n).nanos / ops;
            eprint!("{t:>9.2}");
            print!("\t{t:>.2}");

            T2::<T>::reset_count();
            if TypeId::of::<H2<T, H>>() != TypeId::of::<NoHeap>() {
                let f = W::setup::<T, H>(n);
                T2::<T>::reset_count();
                f();
            }
            let count = T2::<T>::get_count() as f64 / ops;
            eprint!(" {count:.2}");
            print!("\t{count:.2}");

            // Don't test larger n for this workflow once things get too slow.
            if t > 100.0 {
                *ok = false;
            }
        }

        bench_one::<T, H, HeapSort>(n, &mut ok[0]);
        bench_one::<T, H, ConstantSize>(n, &mut ok[1]);
        if !increasing {
            bench_one::<T, H, Decreasing>(n, &mut ok[2]);
        }

        eprintln!();
        println!();
    }
}

fn test<T: Elem + 'static>() {
    eprintln!("QUICKHEAP");
    bench::<T, scalar_quickheap::ScalarQuickHeap<T, 1>>(false);
    bench::<T, scalar_quickheap::ScalarQuickHeap<T, 3>>(false);
    if TypeId::of::<T>() == TypeId::of::<i32>() {
        bench::<i32, simd_quickheap::SimdQuickHeap<16, 1>>(false);
        bench::<i32, simd_quickheap::SimdQuickHeap<8, 3>>(false);
    }

    eprintln!("BASELINE");
    bench::<T, impls::BinaryHeap<T>>(false);

    eprintln!("DARY");
    bench::<T, impls::DaryHeap<T, 2>>(false);
    bench::<T, impls::DaryHeap<T, 4>>(false);
    bench::<T, impls::DaryHeap<T, 8>>(false);
    bench::<T, impls::DaryHeap<T, 16>>(false);
    bench::<T, impls::OrxDaryHeap<T, 2>>(false);
    bench::<T, impls::OrxDaryHeap<T, 4>>(false);
    bench::<T, impls::OrxDaryHeap<T, 8>>(false);
    bench::<T, impls::OrxDaryHeap<T, 16>>(false);

    eprintln!("Amortized");
    bench::<T, impls::PairingHeap<T>>(false);

    if TypeId::of::<T>() == TypeId::of::<i32>() {
        bench::<i32, impls::FibonacciHeap>(false);
    }
    bench::<T, impls::WeakHeap<T>>(false);

    eprintln!("Monotone");
    bench::<T, impls::RadixHeap<T>>(true);

    // eprintln!("Set");
    // bench::<T, impls::BTreeSet<T>>(true);
    // bench::<T, impls::RevBTreeSet<T>>(true);
    // bench::<T, impls::IndexSetBTreeSet<T>>(true);
    // bench::<T, impls::IndexSetRevBTreeSet<T>>(true);
}

fn main() {
    test::<i32>();
    test::<i64>();
}
