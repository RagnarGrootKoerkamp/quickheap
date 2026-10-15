use perfcnt::linux::{HardwareEventType, PerfCounterBuilderLinux};
use perfcnt::{AbstractPerfCounter, PerfCounter};

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

use quickheap_dev::buckets::{
    block_partitioning, list_block_bucket::ListBlockBucket, vec_block_bucket::VecBlockBucket,
    vec_bucket::VecBucket,
};
use quickheap_dev::pivot_strategies::{MedianOfM, RandomPivot, TablePivot};
use quickheap_dev::rebalancing_strategies::{
    AlphaBalancedRebalancing, ExponentialUpperBoundRebalancing, NaiveLogRebalancing, NoRebalancing,
    PivotForgetting, RandomizedRebalancing,
};

#[cfg(feature = "layer_stats")]
use quickheap_dev::stats::layer_stats;

#[cfg(feature = "partition_stats")]
use quickheap_dev::stats::partition_stats;

use std::any::type_name;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time;

struct WorkloadResult {
    workload_name: String,
    result: Vec<(
        u64,
        f64,
        f64,
        f64,
        f64,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    )>,
}

struct PerfCounters<'a> {
    cache: &'a mut PerfCounter,
    instructions: &'a mut PerfCounter,
    cycles: &'a mut PerfCounter,
}

fn run_variant<T: Elem, H: Heap<T>>(pcs: &mut PerfCounters, idx: &mut usize) {
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

    let file = File::create(
        String::from("./evals/data/variants/") + &idx.to_string() + "_" + &heap_name + ".csv",
    )
    .unwrap();

    let mut writer = BufWriter::with_capacity(1 << 15, file);

    let result_hs = run_variant_with_workload::<T, H, HeapSort>(pcs);
    // let result_bad = run_variant_with_workload::<T, H, MostlyPushDecreasing>(pcs);
    // let result_rw = run_variant_with_workload::<T, H, RandomWiggle>(pcs);
    // let result_rc = run_variant_with_workload::<T, H, RandomConstantSize>(pcs);
    // let result_mw = run_variant_with_workload::<T, H, MonotoneWiggle>(pcs);
    let result_mc = run_variant_with_workload::<T, H, MonotoneConstantSize>(pcs);

    // let r = [times_hs, times_rw, times_rc, times_mw, times_mc];
    let r = [result_mc, result_hs];

    writer
        .write_all(
            "workload,n,nanoseconds,cache_misses,inst_per_cycle,bytes_per_elem,median_layers,partitions,redos_cur,redos_next,equal_buckets,avg_elems_partitioned\n"
                .as_bytes(),
        )
        .unwrap();

    for wr in r {
        for (n, t, c, ipc, mem, layers, p, pc, pn, eq, avgelem) in wr.result {
            let line = format!(
                "{},{},{:.1},{:.2},{:.2},{:.2},{},{},{},{},{},{}\n",
                wr.workload_name, n, t, c, ipc, mem, layers, p, pc, pn, eq, avgelem
            );
            writer.write_all(line.as_bytes()).unwrap();
        }
    }

    writer.flush().unwrap();
}

fn run_variant_with_workload<T: Elem, H: Heap<T>, W: Workload>(
    pcs: &mut PerfCounters,
) -> WorkloadResult {
    let workload_name = type_name::<W>().replace("bench::workloads::", "");
    println!("Workload {}", workload_name);

    let mut result: Vec<(
        u64,
        f64,
        f64,
        f64,
        f64,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    )> = vec![];

    // TODO: Bigger n

    // let ns: Vec<u64> = (15..=22).step_by(1).map(|i| (2u64).pow(i)).collect();
    let ns: Vec<u64> = (22..=22).step_by(1).map(|i| (2u64).pow(i)).collect();

    const REPEATS: usize = 3;
    let mut times: [f64; REPEATS] = [0f64; REPEATS];
    let mut all_cache_misses: [f64; REPEATS] = [0f64; REPEATS];
    let mut all_ipcs: [f64; REPEATS] = [0f64; REPEATS];
    let mut all_memory: [usize; REPEATS] = [0; REPEATS];
    let mut all_layers: [usize; REPEATS] = [0; REPEATS];
    let mut partitions: [usize; REPEATS] = [0; REPEATS];
    let mut redos_cur: [usize; REPEATS] = [0; REPEATS];
    let mut redos_next: [usize; REPEATS] = [0; REPEATS];
    let mut equal_buckets: [usize; REPEATS] = [0; REPEATS];
    let mut avg_partitioined_elems: [usize; REPEATS] = [0; REPEATS];

    for n in ns {
        for i in 0..REPEATS {
            let f = W::setup::<T, H>(n);

            #[cfg(feature = "layer_stats")]
            quickheap_dev::stats::layer_stats::reset();
            #[cfg(feature = "partition_stats")]
            quickheap_dev::stats::partition_stats::reset();
            pcs.cache
                .reset()
                .expect("Could not reset cache miss counter");
            pcs.instructions
                .reset()
                .expect("Could not reset instructions counter");
            pcs.cycles.reset().expect("Could not reset cycle counter");

            pcs.cache
                .start()
                .expect("Could not start cache miss counter");
            pcs.instructions
                .start()
                .expect("Could not start instructions counter");
            pcs.cycles.start().expect("Could not start cycle counter");
            let t = time::Instant::now();
            let h = f();
            let elapsed = t.elapsed().as_nanos() as f64;

            pcs.cache.stop().expect("Could not stop cache miss counter");
            pcs.instructions
                .stop()
                .expect("Could not stop instructions counter");
            pcs.cycles.stop().expect("Could not stop cycle counter");

            let cache_misses = pcs.cache.read().expect("Could not read cache miss counter") as f64;
            let instructions =
                pcs.instructions
                    .read()
                    .expect("Could not read instructions counter") as f64;
            let cycles = pcs.cycles.read().expect("Could not read cycle counter") as f64;

            all_cache_misses[i] = cache_misses;
            all_ipcs[i] = instructions / cycles;
            all_memory[i] = h.capacity() * size_of::<T>();
            times[i] = elapsed;
            #[cfg(feature = "layer_stats")]
            {
                all_layers[i] = quickheap_dev::stats::layer_stats::median();
            }

            #[cfg(feature = "partition_stats")]
            {
                let result = quickheap_dev::stats::partition_stats::result();
                partitions[i] = result.partitions;
                redos_cur[i] = result.redos_cur_layer_empty;
                redos_next[i] = result.redos_next_layer_empty;
                equal_buckets[i] = result.equal_buckets_found;
                avg_partitioined_elems[i] = result.avg_num_elem_partitioned;
            }
        }

        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        all_cache_misses.sort_by(|a, b| a.partial_cmp(b).unwrap());
        all_ipcs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        all_memory.sort_by(|a, b| a.partial_cmp(b).unwrap());
        all_layers.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // TODO: Supposed to be: let ops = n as f64 * (n as f64).log2() * W::NORMALIZATION as f64;
        let ops = n as f64 * W::NORMALIZATION as f64;
        let norm_time = times[REPEATS / 2] as f64 / ops;
        let norm_cache = all_cache_misses[REPEATS / 2] as f64 / ops;
        let norm_mem = all_memory[2] as f64 / ops;

        result.push((
            n,
            norm_time,
            norm_cache,
            all_ipcs[REPEATS / 2],
            norm_mem,
            all_layers[REPEATS / 2],
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

    let mut cache: PerfCounter =
        PerfCounterBuilderLinux::from_hardware_event(HardwareEventType::CacheMisses)
            .exclude_kernel()
            .finish()
            .expect("Could not create the counter");

    let mut cycles: PerfCounter =
        PerfCounterBuilderLinux::from_hardware_event(HardwareEventType::CPUCycles)
            .exclude_kernel()
            .finish()
            .expect("cycles");
    let mut instructions: PerfCounter =
        PerfCounterBuilderLinux::from_hardware_event(HardwareEventType::Instructions)
            .exclude_kernel()
            .finish()
            .expect("instructions");

    let mut cnt = 0;

    let mut pcs = PerfCounters {
        cache: &mut cache,
        instructions: &mut instructions,
        cycles: &mut cycles,
    };

    // Run the Scalar Variant
    // run_variant::<i64, ScalarQuickHeap<i64, 3, false, { Search::LinearScan }>>(pcs, &mut cnt);

    // Run the SIMD Variants
    /* run_variant::<
        i64, // Elem Type
        quickheap::SimdQuickHeap<i64, Avx512>,
    >(&mut pcs, &mut cnt); */

    cnt += 1;

    run_variant::<
        i64, // Elem Type
        quickheap_dev::ConfigurableSimdQuickHeap<
            i64,                                                    // Elem Type
            VecBucket<i64, 128, 128>,                               // Bucket
            block_partitioning::FlatPartitioning<Avx512, 128, 128>, // Partitioning Schema
            Avx512,                                                 // SIMD Type
            MedianOfM<3>,                                           // Pivot Strategy
            NoRebalancing,                                          // Rebalancing Strategy
            16,                                                     // Size smallest bucket
            128,                                                    // Bucket Size
            128,                                                    // Bucket Cap
            true,                                                   // Last layer sorted
            false,                                                  // Use equal buckets
        >,
    >(&mut pcs, &mut cnt);

    cnt += 1;

    run_variant::<
        i64, // Elem Type
        quickheap_dev::ConfigurableSimdQuickHeap<
            i64,                                                       // Elem Type
            ListBlockBucket<i64, 112, 128>,                            // Bucket
            block_partitioning::BlockedPartitioning<Avx512, 112, 128>, // Partitioning Schema
            Avx512,                                                    // SIMD Type
            MedianOfM<3>,                                              // Pivot Strategy
            NoRebalancing,                                             // Rebalancing Strategy
            16,                                                        // Size smallest bucket
            112,                                                       // Bucket Size
            128,                                                       // Bucket Cap
            true,                                                      // Last layer sorted
            false,                                                     // Use equal buckets
        >,
    >(&mut pcs, &mut cnt);
}
