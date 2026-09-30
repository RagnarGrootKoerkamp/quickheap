use bench::{
    Heap,
    scalar_quickheap::{ScalarQuickHeap, Search},
};

use bench::workloads::{
    Elem, HeapSort, MonotoneConstantSize, MonotoneWiggle, RandomConstantSize, RandomWiggle,
    Workload,
};

use quickheap::Avx2;
use quickheap::Avx512;
use quickheap::buckets::list_block_bucket::ListBlockBucket;
use quickheap::buckets::vec_block_bucket::VecBlockBucket;
use quickheap::buckets::vec_bucket::VecBucket;
use quickheap::pivot_strategies::MedianOfM;
use quickheap::pivot_strategies::RandomPivot;
use quickheap::pivot_strategies::TablePivot;
use quickheap::rebalancing_strategies::NoRebalancing;

use std::any::type_name;
use std::time;

struct WorkloadResult {
    workload_name: String,
    result: Vec<(u64, f64)>,
}

fn run_variant<T: Elem, H: Heap<T>>() {
    let heap_name = type_name::<H>()
        .replace(" ", "")
        .replace("ConfigurableSimdQuickHeap", "SimdQH")
        .replace("ScalarQuickHeap", "ScalarQH")
        .replace("bench::scalar_quickheap::", "")
        .replace("quickheap::", "")
        .replace("buckets::", "")
        .replace("pivot_strategies::", "")
        .replace("rebalancing_strategies::", "")
        .replace("vec_bucket::", "")
        .replace("vec_block_bucket::", "")
        .replace("list_block_bucket::", "")
        .replace("simd::", "");

    println!("Run variant {}", heap_name);

    // let result_hs = run_variant_with_workload::<T, H, HeapSort>();
    // let result_rw = run_variant_with_workload::<T, H, RandomWiggle>();
    // let result_rc = run_variant_with_workload::<T, H, RandomConstantSize>();
    // let result_mw = run_variant_with_workload::<T, H, MonotoneWiggle>();
    let result_mc = run_variant_with_workload::<T, H, MonotoneConstantSize>();

    // let r = [times_hs, times_rw, times_rc, times_mw, times_mc];
    let r = [result_mc];
}

fn run_variant_with_workload<T: Elem, H: Heap<T>, W: Workload>() -> WorkloadResult {
    let workload_name = type_name::<W>().replace("bench::workloads::", "");
    println!("Workload {}", workload_name);

    let mut result: Vec<(u64, f64)> = vec![];

    // TODO: Bigger n
    let ns: Vec<u64> = vec![2 << 21];

    const REPEATS: usize = 3;
    let mut times: [f64; REPEATS] = [0f64; REPEATS];

    for n in ns {
        for i in 0..REPEATS {
            let f = W::setup::<T, H>(n);

            let t = time::Instant::now();
            f();
            let elapsed = t.elapsed().as_nanos() as f64;

            times[i] = elapsed;
        }

        times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // TODO: Supposed to be: let ops = n as f64 * (n as f64).log2() * W::NORMALIZATION as f64;
        let ops = n as f64 * W::NORMALIZATION as f64;
        let norm_time = times[REPEATS / 2] as f64 / ops;

        result.push((n, norm_time));
    }

    WorkloadResult {
        workload_name,
        result,
    }
}

fn main() {
    println!("Running the different variants of the SIMD quickheap:");

    /*

    // Run the Scalar Variant
    run_variant::<i64, ScalarQuickHeap<i64, 3, false, { Search::LinearScan }>>(&mut pc, &mut cnt);

    cnt += 1;

    // Run the SIMD Variants
    run_variant::<
        i64, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i64,            // Elem Type
            VecBucket<i64>, // Bucket
            Avx2,           // Simd
            MedianOfM<3>,   // Pivot Strategy
            NoRebalancing,  // Rebalancing Strategy
            16,             // Size smallest bucket
            true,           // Last layer sorted
            false,          // Use equal buckets
        >,
    >();

    cnt += 1;

    run_variant::<
        i64, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i64,            // Elem Type
            VecBucket<i64>, // Bucket
            Avx512,         // Simd
            MedianOfM<3>,   // Pivot Strategy
            NoRebalancing,  // Rebalancing Strategy
            16,             // Size smallest bucket
            true,           // Last layer sorted
            false,          // Use equal buckets
        >,
    >(&mut pc, &mut cnt);

    cnt += 1;

    */

    // Different Buckets
    run_variant::<
        i64, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i64,                 // Elem Type
            VecBucket<i64, 128>, // Bucket
            Avx2,                // Simd
            MedianOfM<3>,        // Pivot Strategy
            NoRebalancing,       // Rebalancing Strategy
            16,                  // Size smallest bucket
            128,                 // Bucket Size
            true,                // Last layer sorted
            false,               // Use equal buckets
        >,
    >();
}
