use bench::{
    Heap,
    scalar_quickheap::{ScalarQuickHeap, Search},
};

use bench::workloads::{
    Elem, HeapSort, MonotoneConstantSize, MonotoneWiggle, MostlyPushDecreasing, RandomConstantSize,
    RandomWiggle, Workload,
};

use quickheap::Neon;
use quickheap::buckets::list_block_bucket::ListBlockBucket;
use quickheap::buckets::vec_block_bucket::VecBlockBucket;
use quickheap::buckets::vec_bucket::VecBucket;
use quickheap::pivot_strategies::MedianOfM;
use quickheap::pivot_strategies::RandomPivot;
use quickheap::pivot_strategies::TablePivot;
use quickheap::rebalancing_strategies::AlphaBalancedRebalancing;
use quickheap::rebalancing_strategies::ExponentialUpperBoundRebalancing;
use quickheap::rebalancing_strategies::NaiveLogRebalancing;
use quickheap::rebalancing_strategies::NoRebalancing;
use quickheap::rebalancing_strategies::PivotForgetting;
use quickheap::rebalancing_strategies::RandomizedRebalancing;

use std::any::type_name;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time;

struct WorkloadResult {
    workload_name: String,
    result: Vec<(u64, f64)>,
}

fn run_variant<T: Elem, H: Heap<T>>(idx: &mut usize) {
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

    let file = File::create(
        String::from("./evals/data/variants/") + &idx.to_string() + "_" + &heap_name + ".csv",
    )
    .unwrap();

    let mut writer = BufWriter::with_capacity(1 << 15, file);

    let result_hs = run_variant_with_workload::<T, H, HeapSort>();
    // let result_bad = run_variant_with_workload::<T, H, MostlyPushDecreasing>();
    // let result_rw = run_variant_with_workload::<T, H, RandomWiggle>(pcs);
    // let result_rc = run_variant_with_workload::<T, H, RandomConstantSize>(pcs);
    // let result_mw = run_variant_with_workload::<T, H, MonotoneWiggle>(pcs);
    let result_mc = run_variant_with_workload::<T, H, MonotoneConstantSize>();

    // let r = [times_hs, times_rw, times_rc, times_mw, times_mc];
    let r = [result_mc, result_hs];

    writer
        .write_all("workload,n,nanoseconds,cache_misses,inst_per_cycle\n".as_bytes())
        .unwrap();

    for wr in r {
        for (n, t) in wr.result {
            let line = format!("{},{},{:.1}\n", wr.workload_name, n, t);
            writer.write_all(line.as_bytes()).unwrap();
        }
    }

    writer.flush().unwrap();
}

fn run_variant_with_workload<T: Elem, H: Heap<T>, W: Workload>() -> WorkloadResult {
    let workload_name = type_name::<W>().replace("bench::workloads::", "");
    println!("Workload {}", workload_name);

    let mut result: Vec<(u64, f64)> = vec![];

    // TODO: Bigger n

    // TODO:
    let ns: Vec<u64> = (15..=22).step_by(1).map(|i| (2u64).pow(i)).collect();
    // let ns: Vec<u64> = (10..=25).step_by(5).map(|i| (2u64).pow(i)).collect();

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
    println!("Running the different SIMD quickheap experiments:");

    let mut cnt = 0;

    // Run the Scalar Variant
    // run_variant::<i64, ScalarQuickHeap<i64, 3, false, { Search::LinearScan }>>(&mut cnt);

    cnt += 1;

    // Run the SIMD Variants
    /*
    run_variant::<
        i64, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i64,                      // Elem Type
            VecBucket<i64, 128, 128>, // Bucket
            Avx2,                     // Simd
            MedianOfM<3>,             // Pivot Strategy
            NoRebalancing,            // Rebalancing Strategy
            16,                       // Size smallest bucket
            128,                      // Bucket Size
            128,                      // Bucket Cap
            true,                     // Last layer sorted
            false,                    // Use equal buckets
        >,
    >(pcs, &mut cnt); */

    /*
    cnt += 1;

    run_variant::<
        i64, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i64,                      // Elem Type
            VecBucket<i64, 128, 128>, // Bucket
            Avx2,                     // Simd
            MedianOfM<3>,             // Pivot Strategy
            NoRebalancing,            // Rebalancing Strategy
            16,                       // Size smallest bucket
            128,                      // Bucket Size
            128,                      // Bucket Cap
            true,                     // Last layer sorted
            false,                    // Use equal buckets
        >,
    >(pcs, &mut cnt);

    cnt += 1; */

    /*
    // Different Buckets
    run_variant::<
        i64, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i64,                           // Elem Type
            VecBlockBucket<i64, 128, 154>, // Bucket
            Avx2,                          // Simd
            MedianOfM<3>,                  // Pivot Strategy
            NoRebalancing,                 // Rebalancing Strategy
            16,                            // Size smallest bucket
            128,                           // Bucket Size
            154,                           // Bucket Cap
            true,                          // Last layer sorted
            false,                         // Use equal buckets
        >,
    >(pcs, &mut cnt);

    cnt += 1;

    run_variant::<
        i64, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i64,                            // Elem Type
            ListBlockBucket<i64, 128, 154>, // Bucket
            Avx2,                           // Simd
            MedianOfM<3>,                   // Pivot Strategy
            NoRebalancing,                  // Rebalancing Strategy
            16,                             // Size smallest bucket
            128,                            // Bucket Size
            154,                            // Bucket Cap
            true,                           // Last layer sorted
            false,                          // Use equal buckets
        >,
    >(pcs, &mut cnt);

    cnt += 1;
     */

    /* run_variant::<
        i64, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i64,                      // Elem Type
            VecBucket<i64, 128, 128>, // Bucket
            Avx2,                     // Simd
            MedianOfM<3>,             // Pivot Strategy
            NoRebalancing,            // Rebalancing Strategy
            16,                       // Size smallest bucket
            128,                      // Bucket Size
            128,                      // Bucket Cap
            true,                     // Last layer sorted
            false,                    // Use equal buckets
        >,
    >(pcs, &mut cnt);

    cnt += 1; */

    // FROM HERE - - - - - - - -

    /*
    run_variant::<
        i64, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i64,                      // Elem Type
            VecBucket<i64, 128, 128>, // Bucket
            Avx512,                   // Simd
            MedianOfM<3>,             // Pivot Strategy
            NoRebalancing,            // Rebalancing Strategy
            16,                       // Size smallest bucket
            128,                      // Bucket Size
            128,                      // Bucket Cap
            true,                     // Last layer sorted
            false,                    // Use equal buckets
        >,
    >(&mut pcs, &mut cnt); */

    cnt += 1;

    /*
    run_variant::<
        i64, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i64,                           // Elem Type
            VecBlockBucket<i64, 128, 154>, // Bucket
            Avx512,                        // Simd
            MedianOfM<3>,                  // Pivot Strategy
            NoRebalancing,                 // Rebalancing Strategy
            16,                            // Size smallest bucket
            128,                           // Bucket Size
            154,                           // Bucket Cap
            true,                          // Last layer sorted
            false,                         // Use equal buckets
        >,
    >(&mut pcs, &mut cnt); */

    cnt += 1;

    /*
    run_variant::<
        i64, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i64,                            // Elem Type
            ListBlockBucket<i64, 128, 154>, // Bucket
            Avx512,                         // Simd
            MedianOfM<3>,                   // Pivot Strategy
            NoRebalancing,                  // Rebalancing Strategy
            16,                             // Size smallest bucket
            128,                            // Bucket Size
            154,                            // Bucket Cap
            true,                           // Last layer sorted
            false,                          // Use equal buckets
        >,
    >(&mut pcs, &mut cnt); */

    cnt += 1;

    /*
    run_variant::<
        i64, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i64,                      // Elem Type
            VecBucket<i64, 128, 128>, // Bucket
            Avx512,                   // Simd
            MedianOfM<3>,             // Pivot Strategy
            PivotForgetting<2, 2048>, // Rebalancing Strategy
            16,                       // Size smallest bucket
            128,                      // Bucket Size
            128,                      // Bucket Cap
            true,                     // Last layer sorted
            false,                    // Use equal buckets
        >,
    >(&mut pcs, &mut cnt); */

    cnt += 1;

    /*
    run_variant::<
        i64, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i64,                           // Elem Type
            VecBlockBucket<i64, 128, 154>, // Bucket
            Avx512,                        // Simd
            MedianOfM<3>,                  // Pivot Strategy
            PivotForgetting<2, 2048>,      // Rebalancing Strategy
            16,                            // Size smallest bucket
            128,                           // Bucket Size
            154,                           // Bucket Cap
            true,                          // Last layer sorted
            false,                         // Use equal buckets
        >,
    >(&mut pcs, &mut cnt); */

    cnt += 1;

    /*
    run_variant::<
        i64, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i64,                            // Elem Type
            ListBlockBucket<i64, 128, 154>, // Bucket
            Avx512,                         // Simd
            MedianOfM<3>,                   // Pivot Strategy
            PivotForgetting<2, 2048>,       // Rebalancing Strategy
            16,                             // Size smallest bucket
            128,                            // Bucket Size
            154,                            // Bucket Cap
            true,                           // Last layer sorted
            false,                          // Use equal buckets
        >,
    >(&mut pcs, &mut cnt); */

    cnt += 1;

    /*
    run_variant::<
        i64, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i64,                              // Elem Type
            ListBlockBucket<i64, 128, 154>,   // Bucket
            Avx512,                           // Simd
            MedianOfM<3>,                     // Pivot Strategy
            ExponentialUpperBoundRebalancing, // Rebalancing Strategy
            16,                               // Size smallest bucket
            128,                              // Bucket Size
            154,                              // Bucket Cap
            true,                             // Last layer sorted
            false,                            // Use equal buckets
        >,
    >(&mut pcs, &mut cnt); */

    cnt += 1;

    run_variant::<
        i64, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i64,                            // Elem Type
            ListBlockBucket<i64, 128, 154>, // Bucket
            Neon,                           // Simd
            MedianOfM<3>,                   // Pivot Strategy
            RandomizedRebalancing,          // Rebalancing Strategy
            16,                             // Size smallest bucket
            128,                            // Bucket Size
            154,                            // Bucket Cap
            true,                           // Last layer sorted
            false,                          // Use equal buckets
        >,
    >(&mut cnt);

    cnt += 1;

    /*
    run_variant::<
        i64, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i64,                      // Elem Type
            VecBucket<i64, 128, 128>, // Bucket
            Avx512,                   // Simd
            MedianOfM<3>,             // Pivot Strategy
            AlphaBalancedRebalancing, // Rebalancing Strategy
            16,                       // Size smallest bucket
            128,                      // Bucket Size
            128,                      // Bucket Cap
            true,                     // Last layer sorted
            false,                    // Use equal buckets
        >,
    >(&mut pcs, &mut cnt);

    cnt += 1;

    run_variant::<
        i64, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i64,                           // Elem Type
            VecBucket<i64, 128, 128>,      // Bucket
            Avx512,                        // Simd
            MedianOfM<3>,                  // Pivot Strategy
            NaiveLogRebalancing<10, 2048>, // Rebalancing Strategy
            16,                            // Size smallest bucket
            128,                           // Bucket Size
            128,                           // Bucket Cap
            true,                          // Last layer sorted
            false,                         // Use equal buckets
        >,
    >(&mut pcs, &mut cnt);

    cnt += 1; */
    /*


    // Different Buckets
    run_variant::<
        i64, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i64,                           // Elem Type
            VecBlockBucket<i64, 128, 154>, // Bucket
            Avx512,                        // Simd
            MedianOfM<3>,                  // Pivot Strategy
            NoRebalancing,                 // Rebalancing Strategy
            16,                            // Size smallest bucket
            128,                           // Bucket Size
            154,                           // Bucket Cap
            true,                          // Last layer sorted
            false,                         // Use equal buckets
        >,
    >(&mut pc, &mut cnt);

    cnt += 1;

    run_variant::<
        i64, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i64,                            // Elem Type
            ListBlockBucket<i64, 128, 154>, // Bucket
            Avx512,                         // Simd
            MedianOfM<3>,                   // Pivot Strategy
            NoRebalancing,                  // Rebalancing Strategy
            16,                             // Size smallest bucket
            128,                            // Bucket Size
            154,                            // Bucket Cap
            true,                           // Last layer sorted
            false,                          // Use equal buckets
        >,
    >(&mut pc, &mut cnt);
     */

    /*

    cnt += 1;

    // Different Pivot Strategies
    run_variant::<
        i64, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i64,            // Elem Type
            VecBucket<i64>, // Bucket
            Avx512,         // Simd
            RandomPivot,    // Pivot Strategy
            NoRebalancing,  // Rebalancing Strategy
            16,             // Size smallest bucket
            true,           // Last layer sorted
            false,          // Use equal buckets
        >,
    >(&mut pc, &mut cnt);

    cnt += 1;

    run_variant::<
        i64, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i64,            // Elem Type
            VecBucket<i64>, // Bucket
            Avx512,         // Simd
            MedianOfM<5>,   // Pivot Strategy
            NoRebalancing,  // Rebalancing Strategy
            16,             // Size smallest bucket
            true,           // Last layer sorted
            false,          // Use equal buckets
        >,
    >(&mut pc, &mut cnt);

    cnt += 1;

    run_variant::<
        i64, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i64,            // Elem Type
            VecBucket<i64>, // Bucket
            Avx512,         // Simd
            TablePivot,     // Pivot Strategy
            NoRebalancing,  // Rebalancing Strategy
            16,             // Size smallest bucket
            true,           // Last layer sorted
            false,          // Use equal buckets
        >,
    >(&mut pc, &mut cnt);

    cnt += 1;

    */

    /*

    cnt = 8;

    // Equal Buckets
    run_variant::<
        i32, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i32,            // Elem Type
            VecBucket<i32>, // Bucket
            Avx512,         // Simd
            MedianOfM<3>,   // Pivot Strategy
            NoRebalancing,  // Rebalancing Strategy
            16,             // Size smallest bucket
            true,           // Last layer sorted
            false,          // Use equal buckets
        >,
    >(&mut pc, &mut cnt);

    cnt += 1;

    run_variant::<
        i32, // Elem Type
        quickheap::ConfigurableSimdQuickHeap<
            i32,            // Elem Type
            VecBucket<i32>, // Bucket
            Avx512,         // Simd
            MedianOfM<3>,   // Pivot Strategy
            NoRebalancing,  // Rebalancing Strategy
            16,             // Size smallest bucket
            true,           // Last layer sorted
            true,           // Use equal buckets
        >,
    >(&mut pc, &mut cnt);

     */
}
