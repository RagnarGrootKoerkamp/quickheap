use bench::{
    Heap,
    scalar_quickheap::{ScalarQuickHeap, Search},
};

use bench::workloads::{
    Elem, HeapSort, MonotoneConstantSize, MonotoneWiggle, MostlyPushDecreasing, RandomConstantSize,
    RandomWiggle, Workload,
};

use quickheap::partitioning;
use quickheap::simd::Avx2;
use quickheap::simd::Avx512;
use quickheap::simd::Neon;

use quickheap_dev::buckets::{
    block_partitioning, list_block_bucket::ListBlockBucket, vec_block_bucket::VecBlockBucket,
    vec_bucket::VecBucket,
};
use quickheap_dev::pivot_strategies::{
    EqualMultiMedianOfM, GeoMultiMedianOfM, MedianOfM, RandomPivot, TablePivot,
};
use quickheap_dev::rebalancing_strategies::{
    AlphaBalancedRebalancing, ExponentialUpperBoundRebalancing, NaiveLogRebalancing, NoRebalancing,
    PivotForgetting, RandomizedRebalancing,
};

#[cfg(feature = "partition_stats")]
use quickheap_dev::stats::partition_stats;

use std::any::type_name;
use std::io::Write;
use std::time;

fn run_variant<T: Elem, H: Heap<T>>() {
    let heap_name = type_name::<H>()
        .replace(" ", "")
        .replace("ConfigurableSimdQuickHeap", "SimdQH")
        .replace("ScalarQuickHeap", "ScalarQH")
        .replace("bench::scalar_quickheap::", "")
        .replace("quickheap_dev::", "")
        .replace("quickheap::", "")
        .replace("buckets::", "")
        .replace("pivot_strategies::", "")
        .replace("rebalancing_strategies::", "")
        .replace("vec_bucket::", "")
        .replace("vec_block_bucket::", "")
        .replace("list_block_bucket::", "")
        .replace("block_partitioning::", "")
        .replace("partitioning::", "")
        .replace("simd::", "");

    println!("Run variant {}", heap_name);

    // let result_hs = run_variant_with_workload::<T, H, HeapSort>(pcs);
    // let result_bad = run_variant_with_workload::<T, H, MostlyPushDecreasing>(pcs);
    // let result_rw = run_variant_with_workload::<T, H, RandomWiggle>(pcs);
    // let result_rc = run_variant_with_workload::<T, H, RandomConstantSize>(pcs);
    // let result_mw = run_variant_with_workload::<T, H, MonotoneWiggle>(pcs);
    run_variant_with_workload::<T, H, MonotoneConstantSize>();
}

fn run_variant_with_workload<T: Elem, H: Heap<T>, W: Workload>() {
    let workload_name = type_name::<W>().replace("bench::workloads::", "");
    println!("Workload {}", workload_name);

    let n: u64 = 2u64.pow(25);
    const REPEATS: usize = 5;
    let mut times: [f64; REPEATS] = [0f64; REPEATS];

    print!("Running n = {}", n);
    std::io::stdout().flush().expect("Flush I/O");
    for i in 0..REPEATS {
        let f = W::setup::<T, H>(n);
        #[cfg(feature = "partition_stats")]
        quickheap_dev::stats::partition_stats::reset();

        let t = time::Instant::now();
        f();
        let elapsed = t.elapsed().as_nanos() as f64;
        times[i] = elapsed;

        #[cfg(feature = "partition_stats")]
        {
            let result = quickheap_dev::stats::partition_stats::result();
            partitions[i] = result.partitions;
            redos_cur[i] = result.redos_cur_layer_empty;
            redos_next[i] = result.redos_next_layer_empty;
            equal_buckets[i] = result.equal_buckets_found;
            avg_partitioined_elems[i] = result.avg_num_elem_partitioned;
        }

        print!(" .");
        std::io::stdout().flush().expect("Flush I/O");
    }

    println!();
}

fn main() {
    println!("Run heap for profiling.");

    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    type Simd = Avx2;

    #[cfg(target_feature = "avx512f")]
    type Simd = Avx512;

    #[cfg(target_arch = "aarch64")]
    type Simd = Neon;

    type T = i32;

    // Run the SIMD Variants
    run_variant::<
        T, // Elem Type
        quickheap::SimdQuickHeap<T, Simd>,
    >();

    run_variant::<
        T, // Elem Type
        quickheap_dev::ConfigurableSimdQuickHeap<
            T,                                          // Elem Type
            VecBucket<T>,                               // Bucket
            block_partitioning::FlatPartitioning<Simd>, // Partitioning Schema
            Simd,                                       // SIMD Type
            MedianOfM<3>,                               // Pivot Strategy
            GeoMultiMedianOfM<1>,                       // Multi Pivot Strategy
            NoRebalancing,                              // Rebalancing Strategy
            16,                                         // Size smallest bucket
            0,                                          // Bucket Size
            0,                                          // Bucket Cap
            true,                                       // Last layer sorted
            false,                                      // Use equal buckets
            2,
            0,
        >,
    >();
}
