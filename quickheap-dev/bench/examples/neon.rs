use bench::{
    Heap,
    scalar_quickheap::{ScalarQuickHeap, Search},
};

use quickheap_dev::stats;

use bench::workloads::{
    Elem, HeapSort, MonotoneConstantSize, MonotoneWiggle, MostlyPushDecreasing, RandomConstantSize,
    RandomWiggle, Workload,
};

use quickheap::simd::Neon;
use quickheap_dev::buckets::vec_bucket::VecBucket;
use quickheap_dev::buckets::{
    block::Block, block_partitioning::BlockedPartitioning, list_block_bucket::ListBlockBucket,
};
use quickheap_dev::buckets::{
    block_partitioning::FlatPartitioning, vec_block_bucket::VecBlockBucket,
};
use quickheap_dev::pivot_strategies::{MedianOfM, RandomPivot, TablePivot};
use quickheap_dev::rebalancing_strategies::{
    AlphaBalancedRebalancing, ExponentialUpperBoundRebalancing, NaiveLogRebalancing, NoRebalancing,
    PivotForgetting, RandomizedRebalancing,
};

use std::any::type_name;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time;

struct WorkloadResult {
    workload_name: String,
    result: Vec<(u64, f64, usize, usize, usize, usize, usize)>,
}

fn run_variant<T: Elem, H: Heap<T>>(idx: &mut usize) {
    let heap_name = type_name::<H>()
        .replace(" ", "")
        .replace("ConfigurableSimdQuickHeap", "SimdQH")
        .replace("ScalarQuickHeap", "ScalarQH")
        .replace("bench::scalar_quickheap::", "")
        .replace("quickheap::", "")
        .replace("quickheap_dev::", "")
        .replace("buckets::", "")
        .replace("pivot_strategies::", "")
        .replace("rebalancing_strategies::", "")
        .replace("vec_bucket::", "")
        .replace("vec_block_bucket::", "")
        .replace("list_block_bucket::", "")
        .replace("block_partitioning::", "")
        .replace("simd::", "");

    println!("Run variant {}", heap_name);

    let file = File::create(
        String::from("./evals/data/variants/") + &idx.to_string() + "_" + &heap_name + ".csv",
    )
    .unwrap();

    let mut writer = BufWriter::with_capacity(1 << 15, file);

    // let result_hs = run_variant_with_workload::<T, H, HeapSort>();
    // let result_bad = run_variant_with_workload::<T, H, MostlyPushDecreasing>();
    // let result_rw = run_variant_with_workload::<T, H, RandomWiggle>();
    // let result_rc = run_variant_with_workload::<T, H, RandomConstantSize>();
    // let result_mw = run_variant_with_workload::<T, H, MonotoneWiggle>();
    let result_mc = run_variant_with_workload::<T, H, MonotoneConstantSize>();

    let r = [result_mc];

    writer
        .write_all(
            "workload,n,nanoseconds,partitions,redos_cur,redos_next,equal_buckets,avg_elems_partitioned\n".as_bytes(),
        )
        .unwrap();

    for wr in r {
        for (n, t, p, pc, pn, eq, avgelem) in wr.result {
            let line = format!(
                "{},{},{:.1},{},{},{},{},{}\n",
                wr.workload_name, n, t, p, pc, pn, eq, avgelem
            );
            writer.write_all(line.as_bytes()).unwrap();
        }
    }

    writer.flush().unwrap();
}

fn run_variant_with_workload<T: Elem, H: Heap<T>, W: Workload>() -> WorkloadResult {
    let workload_name = type_name::<W>().replace("bench::workloads::", "");
    println!("Workload {}", workload_name);

    let mut result: Vec<(u64, f64, usize, usize, usize, usize, usize)> = vec![];

    // TODO: Bigger n

    // TODO:
    // let ns: Vec<u64> = (13..=20).step_by(1).map(|i| (2u64).pow(i)).collect();
    let ns: Vec<u64> = (15..=22).step_by(1).map(|i| (2u64).pow(i)).collect();
    // let ns: Vec<u64> = vec![2u64.pow(22)];
    // let ns: Vec<u64> = (10..=23).step_by(5).map(|i| (2u64).pow(i)).collect();

    const REPEATS: usize = 3;
    let mut times: [f64; REPEATS] = [0f64; REPEATS];
    let mut partitions: [usize; REPEATS] = [0; REPEATS];
    let mut redos_cur: [usize; REPEATS] = [0; REPEATS];
    let mut redos_next: [usize; REPEATS] = [0; REPEATS];
    let mut equal_buckets: [usize; REPEATS] = [0; REPEATS];
    let mut avg_partitioined_elems: [usize; REPEATS] = [0; REPEATS];

    for n in ns {
        for i in 0..REPEATS {
            let f = W::setup::<T, H>(n);
            stats::partition_stats::reset();
            let t = time::Instant::now();
            f();
            let elapsed = t.elapsed().as_nanos() as f64;

            let result = stats::partition_stats::result();

            times[i] = elapsed;
            partitions[i] = result.partitions;
            redos_cur[i] = result.redos_cur_layer_empty;
            redos_next[i] = result.redos_next_layer_empty;
            equal_buckets[i] = result.equal_buckets_found;
            avg_partitioined_elems[i] = result.avg_num_elem_partitioned;
        }

        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        partitions.sort_by(|a, b| a.partial_cmp(b).unwrap());
        redos_cur.sort_by(|a, b| a.partial_cmp(b).unwrap());
        redos_next.sort_by(|a, b| a.partial_cmp(b).unwrap());
        equal_buckets.sort_by(|a, b| a.partial_cmp(b).unwrap());
        avg_partitioined_elems.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let ops = n as f64 * W::NORMALIZATION as f64;
        let norm_time = times[REPEATS / 2] as f64 / ops;

        result.push((
            n,
            norm_time,
            partitions[REPEATS / 2],
            redos_cur[REPEATS / 2],
            redos_next[REPEATS / 2],
            equal_buckets[REPEATS / 2],
            avg_partitioined_elems[REPEATS / 2],
        ));
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
    // cnt += 1;

    // Run the SIMD Variants
    /*run_variant::<
        i64, // Elem Type
        quickheap::SimdQuickHeap<
            i64,  // Elem Type
            Neon, // Simd
        >,
    >(&mut cnt);
    cnt += 1;*/

    /*
    run_variant::<
        i32, // Elem Type
        quickheap_dev::ConfigurableSimdQuickHeap<
            i32,                      // Elem Type
            VecBucket<i32, 128, 128>, // Bucket
            FlatPartitioning<Neon, 128, 128>,
            Neon,          // Simd
            MedianOfM<3>,  // Pivot Strategy
            NoRebalancing, // Rebalancing Strategy
            16,            // Size smallest bucket
            128,           // Bucket Size
            128,           // Bucket Cap
            true,          // Last layer sorted
            true,          // Use equal buckets
        >,
    >(&mut cnt);

    cnt += 1; */

    /* - - - - - -  - - - - */

    run_variant::<
        i64, // Elem Type
        quickheap_dev::ConfigurableSimdQuickHeap<
            i64,                      // Elem Type
            VecBucket<i64, 128, 128>, // Bucket
            FlatPartitioning<Neon, 128, 128>,
            Neon,          // Simd
            MedianOfM<3>,  // Pivot Strategy
            NoRebalancing, // Rebalancing Strategy
            16,            // Size smallest bucket
            128,           // Bucket Size
            128,           // Bucket Cap
            true,          // Last layer sorted
            false,         // Use equal buckets
        >,
    >(&mut cnt);

    run_variant::<
        i64, // Elem Type
        quickheap_dev::ConfigurableSimdQuickHeap<
            i64,                            // Elem Type
            ListBlockBucket<i64, 112, 128>, // Bucket
            BlockedPartitioning<Neon, 112, 128>,
            Neon,          // Simd
            MedianOfM<3>,  // Pivot Strategy
            NoRebalancing, // Rebalancing Strategy
            16,            // Size smallest bucket
            112,           // Bucket Size
            128,           // Bucket Cap
            true,          // Last layer sorted
            false,         // Use equal buckets
        >,
    >(&mut cnt);

    /* - - - - - -  - - - - */

    /*

    run_variant::<
        i32, // Elem Type
        quickheap_dev::ConfigurableSimdQuickHeap<
            i32,                      // Elem Type
            VecBucket<i32, 128, 128>, // Bucket
            FlatPartitioning<Neon, 128, 128>,
            Neon,          // Simd
            MedianOfM<3>,  // Pivot Strategy
            NoRebalancing, // Rebalancing Strategy
            16,            // Size smallest bucket
            128,           // Bucket Size
            128,           // Bucket Cap
            true,          // Last layer sorted
            true,          // Use equal buckets
        >,
    >(&mut cnt);

    cnt += 1;

    run_variant::<
        i32, // Elem Type
        quickheap_dev::ConfigurableSimdQuickHeap<
            i32,                            // Elem Type
            ListBlockBucket<i32, 112, 128>, // Bucket
            BlockedPartitioning<Neon, 112, 128>,
            Neon,          // Simd
            MedianOfM<3>,  // Pivot Strategy
            NoRebalancing, // Rebalancing Strategy
            16,            // Size smallest bucket
            112,           // Bucket Size
            128,           // Bucket Cap
            true,          // Last layer sorted
            true,          // Use equal buckets
        >,
    >(&mut cnt);

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
    >(pcs, &mut cnt);*/

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

    /*
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
    >(&mut cnt); */

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
