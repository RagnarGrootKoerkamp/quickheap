#[cfg(feature = "avx2")]
use quickheap::simd::Avx2;
#[cfg(feature = "avx512")]
use quickheap::simd::Avx512;

use quickheap::pivot_strategies::MedianOfM;

#[cfg(feature = "perf")]
use perfcnt::{
    AbstractPerfCounter,
    linux::{CacheId, CacheOpId, CacheOpResultId, PerfCounterBuilderLinux},
};

use quickheap::workloads::*;
use quickheap::*;
use serde::Serialize;
use std::any::type_name;

const REPEATS: usize = 3;

#[derive(Serialize, Default)]
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

        all_nanos.push(result.nanos);
    }

    all_nanos.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_nanos[all_nanos.len() / 2]
}

fn main() {
    type T = i64;
    let n = 1 << 25;

    #[cfg(feature = "avx2")]
    time_workload::<T, simd_quickheap::SimdQuickHeap<T, Avx2, MedianOfM<1>, 16>, ConstantSize>(n);
    #[cfg(feature = "avx512")]
    time_workload::<T, simd_quickheap::SimdQuickHeap<T, Avx512<true>, MedianOfM<1>, 16>, ConstantSize>(n);
}
