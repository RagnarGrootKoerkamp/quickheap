use bench::{
    Heap,
    scalar_quickheap::{ScalarQuickHeap, Search},
    workloads::RandomConstantSizeManyEqual,
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

use std::any::type_name;
use std::io::Write;
use std::time;

fn run_variant_with_workload<T: Elem, H: Heap<T>, W: Workload>() {
    print!("\x1B[2J\x1B[1;1H");
    println!("Run.");
    std::io::stdout().flush().unwrap();

    let workload_name = type_name::<W>().replace("bench::workloads::", "");

    let n: u64 = 2u64.pow(25);
    const REPEATS: usize = 2;

    for i in 0..REPEATS {
        std::io::stdout().flush().unwrap();
        let f = W::setup::<T, H>(n);

        let t = time::Instant::now();
        f();
        let elapsed = t.elapsed().as_nanos() as f64;
        let normed = elapsed / (W::NORMALIZATION * n) as f64;
        println!("Finished repeat in {:.2}ns", normed);
    }
}

fn main() {
    println!("Run heap for profiling.");

    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    type Simd = Avx2;

    #[cfg(target_feature = "avx512f")]
    type Simd = Avx512;

    #[cfg(target_arch = "aarch64")]
    type Simd = Neon;

    type T = i64;

    run_variant_with_workload::<
        T,
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
        RandomConstantSizeManyEqual,
    >();
}
