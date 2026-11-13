#[cfg(feature = "perf")]
use perfcnt::{
    AbstractPerfCounter, PerfCounter,
    linux::{HardwareEventType, PerfCounterBuilderLinux},
};

use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time;
use std::{
    any::{TypeId, type_name},
    fs::File,
};

use bench::{
    impls, original_quickheap, radix_heap, s3q,
    scalar_quickheap::{self, Search},
    sequence_heap,
};

use quickheap::SimdElem;

#[cfg(feature = "layer_stats")]
use quickheap_dev::stats::layer_stats;

use quickheap::simd;
use quickheap::{
    KvSimdQuickHeap, SimdQuickHeap,
    decrease_key::{DelTokenCountingBloom, DelTokenHash, DelTokenSort, SimpleDK},
};
use quickheap_dev::ConfigurableSimdQuickHeap;
use quickheap_dev::RadixSimdQuickHeap;

use quickheap_dev::{
    buckets::{
        block_partitioning::{BlockedPartitioning, FlatPartitioning},
        list_block_bucket::ListBlockBucket,
        vec_block_bucket::VecBlockBucket,
        vec_bucket::VecBucket,
    },
    pivot_strategies::{
        EqualMultiMedianOfM, GeoMultiMedianOfM, MedianOfM, RandomPivot, TablePivot,
    },
    rebalancing_strategies::{
        AlphaBalancedRebalancing, ExponentialUpperBoundRebalancing, NaiveLogRebalancing,
        NoRebalancing, PivotForgetting, RandomizedRebalancing,
    },
};

use bench::{
    Heap, KvHeap,
    dijkstra::DijkstraQuery,
    graph::Graph,
    graph_workloads::{DijkstraWorkload, GraphWorkload, KvGraphWorkload, PrimWorkload},
    prim::PrimMST,
    workloads::{self, Elem, Workload},
};

use crate::Pivots::EqualMedianOf2;
use crate::Pivots::MedianOf3;

const REPEATS: usize = 5;

pub struct HeapStats {
    pub capacity: usize,
}

pub trait MetricMeasurement<T> {
    fn new() -> Self
    where
        Self: Sized;
    fn start(&mut self);
    fn reset(&mut self);
    fn stop(&mut self, stats: &HeapStats);
    fn get(&self) -> T;
    fn get_name(&self) -> String;
    fn norm_by_ops(&mut self, ops: usize);
}

struct NanoTimer {
    last_start: time::Instant,
    last_stop: time::Instant,
    measurements: Vec<u128>,
    result: f64,
}
impl MetricMeasurement<f64> for NanoTimer {
    fn new() -> Self
    where
        Self: Sized,
    {
        Self {
            last_start: time::Instant::now(),
            last_stop: time::Instant::now(),
            result: 0f64,
            measurements: vec![],
        }
    }

    fn reset(&mut self) {
        self.measurements.clear();
        self.result = 0f64;
    }

    fn start(&mut self) {
        self.last_start = time::Instant::now();
        self.last_stop = time::Instant::now();
    }

    fn stop(&mut self, _: &HeapStats) {
        self.last_stop = time::Instant::now();
        self.measurements
            .push((self.last_stop - self.last_start).as_nanos());
    }

    fn norm_by_ops(&mut self, ops: usize) {
        self.measurements.sort();
        let num_measures = self.measurements.len();
        let median = self.measurements[num_measures / 2];
        self.result = median as f64 / ops as f64;
    }

    fn get(&self) -> f64 {
        self.result
    }

    fn get_name(&self) -> String {
        "nano_seconds".to_string()
    }
}

struct MemoryConsumption {
    measurements: Vec<usize>,
    result: f64,
}
impl MetricMeasurement<f64> for MemoryConsumption {
    fn new() -> Self
    where
        Self: Sized,
    {
        Self {
            result: 0f64,
            measurements: vec![],
        }
    }

    fn reset(&mut self) {
        self.measurements.clear();
        self.result = 0f64;
    }

    fn start(&mut self) {}

    fn stop(&mut self, stats: &HeapStats) {
        self.measurements.push(stats.capacity);
    }

    fn norm_by_ops(&mut self, ops: usize) {
        self.measurements.sort();
        let num_measures = self.measurements.len();
        let median = self.measurements[num_measures / 2];
        self.result = median as f64 / ops as f64;
    }

    fn get(&self) -> f64 {
        self.result
    }

    fn get_name(&self) -> String {
        "bytes_per_elem".to_string()
    }
}

struct MeasureLayerStats {
    measurements: Vec<usize>,
    result: usize,
}
impl MetricMeasurement<usize> for MeasureLayerStats {
    fn new() -> Self
    where
        Self: Sized,
    {
        Self {
            result: 0,
            measurements: vec![],
        }
    }

    fn reset(&mut self) {
        #[cfg(feature = "layer_stats")]
        layer_stats::reset();

        self.measurements.clear();
        self.result = 0;
    }

    fn start(&mut self) {
        #[cfg(feature = "layer_stats")]
        quickheap_dev::stats::layer_stats::reset();
    }

    fn stop(&mut self, _: &HeapStats) {
        #[cfg(feature = "layer_stats")]
        {
            self.measurements
                .push(quickheap_dev::stats::layer_stats::median());
        }
    }

    fn norm_by_ops(&mut self, _: usize) {
        self.measurements.sort();
        let num_measures = self.measurements.len();
        let median = self.measurements[num_measures / 2];
        self.result = median;
    }

    fn get(&self) -> usize {
        self.result
    }

    fn get_name(&self) -> String {
        "median_layers".to_string()
    }
}

#[cfg(feature = "perf")]
struct MeasurePerfStats {
    cache: PerfCounter,
    instructions: PerfCounter,
    cycles: PerfCounter,

    cache_measures: Vec<f64>,
    ipc_measures: Vec<f64>,

    cache_misses: f64,
    ipc: f64,
}
#[cfg(feature = "perf")]
impl MetricMeasurement<String> for MeasurePerfStats {
    fn new() -> Self
    where
        Self: Sized,
    {
        Self {
            cache: PerfCounterBuilderLinux::from_hardware_event(HardwareEventType::CacheMisses)
                .exclude_kernel()
                .finish()
                .expect("Could not create the cache counter."),
            instructions: PerfCounterBuilderLinux::from_hardware_event(
                HardwareEventType::Instructions,
            )
            .exclude_kernel()
            .finish()
            .expect("Could not create the instruction counter."),
            cycles: PerfCounterBuilderLinux::from_hardware_event(HardwareEventType::CPUCycles)
                .exclude_kernel()
                .finish()
                .expect("Could not create the cycle counter."),
            cache_measures: vec![],
            ipc_measures: vec![],
            cache_misses: 0f64,
            ipc: 0f64,
        }
    }

    fn start(&mut self) {
        self.cache
            .start()
            .expect("Could not start cache miss counter");
        self.instructions
            .start()
            .expect("Could not start instructions counter");
        self.cycles.start().expect("Could not start cycle counter");
    }

    fn reset(&mut self) {
        self.cache_measures.clear();
        self.ipc_measures.clear();

        self.cache
            .reset()
            .expect("Could not reset cache miss counter");
        self.instructions
            .reset()
            .expect("Could not reset instructions counter");
        self.cycles.reset().expect("Could not reset cycle counter");
    }

    fn stop(&mut self, _: &HeapStats) {
        self.cache
            .stop()
            .expect("Could not stop cache miss counter");
        self.instructions
            .stop()
            .expect("Could not stop instructions counter");
        self.cycles.stop().expect("Could not stop cycle counter");

        let cache_misses = self
            .cache
            .read()
            .expect("Could not read cache miss counter") as f64;
        let instructions = self
            .instructions
            .read()
            .expect("Could not read instructions counter") as f64;
        let cycles = self.cycles.read().expect("Could not read cycle counter") as f64;

        self.cache_measures.push(cache_misses);
        self.ipc_measures.push(instructions / cycles);
    }

    fn get(&self) -> String {
        format!("{:.2},{:.1}", self.cache_misses, self.ipc)
    }

    fn get_name(&self) -> String {
        "cache_misses,ipc".to_string()
    }

    fn norm_by_ops(&mut self, ops: usize) {
        let num_measures = self.cache_measures.len();
        self.cache_measures.sort_by(|a, b| a.total_cmp(b));
        self.ipc_measures.sort_by(|a, b| a.total_cmp(b));

        self.ipc = self.ipc_measures[num_measures / 2];
        self.cache_misses = self.cache_measures[num_measures / 2] / ops as f64;
    }
}

/*
struct PartitionStats {}
struct MeasurePartitionStats {
    measurements: Vec<PartitionStats>,
    result: f64,
}
impl MetricMeasurement<PartitionStats> for MeasurePartitionStats {
    fn new() -> Self
    where
        Self: Sized,
    {
        Self {
            result: 0f64,
            measurements: vec![],
        }
    }

    fn reset(&mut self) {
        self.measurements.clear();
        self.result = 0f64;
    }

    fn start(&mut self) {}

    fn stop(&mut self, stats: &HeapStats) {
        self.measurements.push(stats.capacity);
    }

    fn norm_by_ops(&mut self, ops: usize) {
        self.measurements.sort();
        let num_measures = self.measurements.len();
        let median = self.measurements[num_measures / 2];
        self.result = median as f64 / ops as f64;
    }

    fn get(&self) -> f64 {
        self.result
    }

    fn get_name(&self) -> String {
        "bytes_per_elem".to_string()
    }
} */

/*
 * Possible Configuration Parameters
 */
#[derive(Clone, Copy)]
pub enum Type {
    I32,
    I64,
    // U32,
    // U64,
}

#[derive(Clone, Copy)]
pub enum HeapFamily {
    // Baseline SimdQH
    SimdQH,
    // Dev SimdQH
    ConfigSimdQH,

    // Pure SimdQH variants
    RadixSimdQH,

    // Competitors
    ScalarQH,
    OriginalQH,
    SequenceHeap,
    S3qHeap,
    RadixHeap,
    BinaryHeap,
    Orx8aryHeap,
    WeakHeap,
}

#[derive(Clone, Copy)]
pub enum Work {
    Heapsort,
    MonotoneCS,
    RandomManyEqual,
    MostlyPushDecreasing,
    // RandomCS,
    // MonotoneW,
    // RandomW,
}

// Config Parameters for the ConfigSimdQH
#[derive(Clone, Copy)]
pub struct ConfigSimdQH {
    bucket: Buckets,
    pivot: Pivots,
    multi_pivot: Pivots,
    rebal: Rebalancing,
    n: usize,
    sort: bool,
    equal: bool,
    p: usize,
    multi_thresh: usize,
}

impl ConfigSimdQH {
    fn empty() -> Self {
        Self {
            bucket: Buckets::InvalBucket,
            pivot: Pivots::InvalPivot,
            multi_pivot: Pivots::InvalPivot,
            rebal: Rebalancing::InvalRebal,
            n: 0,
            sort: false,
            equal: false,
            p: 0,
            multi_thresh: 0,
        }
    }
}

#[derive(Clone, Copy)]
pub struct ConfigRadixQH {
    b: usize,
    thresh: usize,
}
impl ConfigRadixQH {
    fn empty() -> Self {
        Self { b: 0, thresh: 0 }
    }
}

#[derive(Clone, Copy)]
pub struct HeapConfig {
    family: HeapFamily,
    config_simd: ConfigSimdQH,
    config_radix: ConfigRadixQH,
}

#[derive(Clone, Copy)]
pub enum Buckets {
    InvalBucket,
    Vec,
    VecBlock,
    ListBlock,
}

#[derive(Clone, Copy)]
pub enum Pivots {
    InvalPivot,
    Random,
    MedianOf3,
    MedianOf5,
    EqualMedianOf2,
}

#[derive(Clone, Copy)]
pub enum Rebalancing {
    InvalRebal,
    NoRebal,
    PivotForgetting,
    RandomizedRebalancing,
    ExponentialUpperBound,
}

pub enum AnyMeasurement {
    Usize(Box<dyn MetricMeasurement<usize>>),
    U128(Box<dyn MetricMeasurement<u128>>),
    F64(Box<dyn MetricMeasurement<f64>>),
    String(Box<dyn MetricMeasurement<String>>),
}

impl AnyMeasurement {
    fn reset(&mut self) {
        match self {
            Self::Usize(b) => b.reset(),
            Self::U128(b) => b.reset(),
            Self::F64(b) => b.reset(),
            Self::String(b) => b.reset(),
        }
    }

    fn start(&mut self) {
        match self {
            Self::Usize(b) => b.start(),
            Self::U128(b) => b.start(),
            Self::F64(b) => b.start(),
            Self::String(b) => b.start(),
        }
    }

    fn stop(&mut self, stats: &HeapStats) {
        match self {
            Self::Usize(b) => b.stop(stats),
            Self::U128(b) => b.stop(stats),
            Self::F64(b) => b.stop(stats),
            Self::String(b) => b.stop(stats),
        }
    }

    fn norm_by_ops(&mut self, n: usize) {
        match self {
            Self::Usize(b) => b.norm_by_ops(n),
            Self::U128(b) => b.norm_by_ops(n),
            Self::F64(b) => b.norm_by_ops(n),
            Self::String(b) => b.norm_by_ops(n),
        }
    }

    fn get(&mut self) -> String {
        match self {
            Self::Usize(b) => {
                let r = b.get();
                r.to_string()
            }
            Self::U128(b) => {
                let r = b.get();
                r.to_string()
            }
            Self::F64(b) => {
                let r = b.get();
                format!("{:.2}", r)
            }
            Self::String(b) => {
                let r = b.get();
                format!("{}", r)
            }
        }
    }

    fn get_name(&self) -> String {
        match self {
            Self::Usize(b) => b.get_name(),
            Self::U128(b) => b.get_name(),
            Self::F64(b) => b.get_name(),
            Self::String(b) => b.get_name(),
        }
    }
}

pub struct Experiment {
    name: String,
    output_name: String,
    ns: Vec<usize>,
    heaps: Vec<HeapConfig>,
    types: Vec<Type>,
    workloads: Vec<Work>,
    metrics: Vec<AnyMeasurement>,
}

#[cfg(feature = "simd_qhs")]
fn simd_qhs() {
    let heap_config = vec![
        HeapConfig {
            family: HeapFamily::SimdQH,
            config_simd: ConfigSimdQH::empty(),
            config_radix: ConfigRadixQH::empty(),
        },
        HeapConfig {
            family: HeapFamily::ConfigSimdQH,
            config_simd: ConfigSimdQH {
                bucket: Buckets::Vec,
                pivot: MedianOf3,
                multi_pivot: EqualMedianOf2,
                rebal: Rebalancing::NoRebal,
                n: 16,
                sort: true,
                equal: false,
                p: 2,
                multi_thresh: 0,
            },
            config_radix: ConfigRadixQH::empty(),
        },
    ];

    let nano_timer = AnyMeasurement::F64(Box::new(NanoTimer::new()));

    let mut e = Experiment {
        name: "SimdQuickHeaps".to_string(),
        output_name: "simd_qhs".to_string(),
        ns: (18..=24).step_by(2).map(|i| (2usize).pow(i)).collect(),
        heaps: heap_config,
        workloads: vec![Work::MonotoneCS],
        types: vec![Type::I32, Type::I64],
        metrics: vec![nano_timer],
    };

    run(&mut e);
}

#[cfg(feature = "competitors")]
fn competitors() {
    let heap_config = vec![
        HeapConfig {
            family: HeapFamily::SimdQH,
            config_simd: ConfigSimdQH::empty(),
            config_radix: ConfigRadixQH::empty(),
        },
        HeapConfig {
            family: HeapFamily::ScalarQH,
            config_simd: ConfigSimdQH::empty(),
            config_radix: ConfigRadixQH::empty(),
        },
        HeapConfig {
            family: HeapFamily::OriginalQH,
            config_simd: ConfigSimdQH::empty(),
            config_radix: ConfigRadixQH::empty(),
        },
        HeapConfig {
            family: HeapFamily::SequenceHeap,
            config_simd: ConfigSimdQH::empty(),
            config_radix: ConfigRadixQH::empty(),
        },
        HeapConfig {
            family: HeapFamily::S3qHeap,
            config_simd: ConfigSimdQH::empty(),
            config_radix: ConfigRadixQH::empty(),
        },
        HeapConfig {
            family: HeapFamily::RadixHeap,
            config_simd: ConfigSimdQH::empty(),
            config_radix: ConfigRadixQH::empty(),
        },
        HeapConfig {
            family: HeapFamily::BinaryHeap,
            config_simd: ConfigSimdQH::empty(),
            config_radix: ConfigRadixQH::empty(),
        },
        HeapConfig {
            family: HeapFamily::Orx8aryHeap,
            config_simd: ConfigSimdQH::empty(),
            config_radix: ConfigRadixQH::empty(),
        },
        HeapConfig {
            family: HeapFamily::WeakHeap,
            config_simd: ConfigSimdQH::empty(),
            config_radix: ConfigRadixQH::empty(),
        },
    ];

    let nano_timer = AnyMeasurement::F64(Box::new(NanoTimer::new()));

    let mut e = Experiment {
        name: "Competitors".to_string(),
        output_name: "competitors".to_string(),
        ns: (10..=17).step_by(1).map(|i| (2usize).pow(i)).collect(),
        heaps: heap_config,
        workloads: vec![Work::MonotoneCS],
        types: vec![Type::I32, Type::I64],
        metrics: vec![nano_timer],
    };

    run(&mut e);
}

#[cfg(feature = "implicit_equal_buckets")]
fn implicit_equal_buckets() {
    let heap_config = vec![
        HeapConfig {
            family: HeapFamily::ConfigSimdQH,
            config_simd: ConfigSimdQH {
                bucket: Buckets::Vec,
                pivot: MedianOf3,
                multi_pivot: EqualMedianOf2,
                rebal: Rebalancing::NoRebal,
                n: 16,
                sort: true,
                equal: false,
                p: 2,
                multi_thresh: 0,
            },
            config_radix: ConfigRadixQH::empty(),
        },
        HeapConfig {
            family: HeapFamily::ConfigSimdQH,
            config_simd: ConfigSimdQH {
                bucket: Buckets::Vec,
                pivot: MedianOf3,
                multi_pivot: EqualMedianOf2,
                rebal: Rebalancing::NoRebal,
                n: 16,
                sort: true,
                equal: true,
                p: 2,
                multi_thresh: 0,
            },
            config_radix: ConfigRadixQH::empty(),
        },
    ];

    let nano_timer = AnyMeasurement::F64(Box::new(NanoTimer::new()));
    let layer_measure = AnyMeasurement::Usize(Box::new(MeasureLayerStats::new()));

    let mut e = Experiment {
        name: "Implicit Equal Buckets".to_string(),
        output_name: "implicit_equal_buckets".to_string(),
        ns: (18..=24).step_by(2).map(|i| (2usize).pow(i)).collect(),
        heaps: heap_config,
        workloads: vec![Work::MonotoneCS, Work::RandomManyEqual],
        types: vec![Type::I32, Type::I64],
        metrics: vec![nano_timer, layer_measure],
    };

    run(&mut e);
}

#[cfg(feature = "rebalancing")]
fn rebalancing() {
    let heap_config = vec![
        HeapConfig {
            family: HeapFamily::SimdQH,
            config_simd: ConfigSimdQH::empty(),
            config_radix: ConfigRadixQH::empty(),
        },
        HeapConfig {
            family: HeapFamily::ConfigSimdQH,
            config_simd: ConfigSimdQH {
                bucket: Buckets::Vec,
                pivot: MedianOf3,
                multi_pivot: EqualMedianOf2,
                rebal: Rebalancing::NoRebal,
                n: 16,
                sort: true,
                equal: false,
                p: 2,
                multi_thresh: 0,
            },
            config_radix: ConfigRadixQH::empty(),
        },
        HeapConfig {
            family: HeapFamily::ConfigSimdQH,
            config_simd: ConfigSimdQH {
                bucket: Buckets::Vec,
                pivot: MedianOf3,
                multi_pivot: EqualMedianOf2,
                rebal: Rebalancing::PivotForgetting,
                n: 16,
                sort: true,
                equal: false,
                p: 2,
                multi_thresh: 0,
            },
            config_radix: ConfigRadixQH::empty(),
        },
        HeapConfig {
            family: HeapFamily::ConfigSimdQH,
            config_simd: ConfigSimdQH {
                bucket: Buckets::ListBlock,
                pivot: MedianOf3,
                multi_pivot: EqualMedianOf2,
                rebal: Rebalancing::PivotForgetting,
                n: 16,
                sort: true,
                equal: false,
                p: 2,
                multi_thresh: 0,
            },
            config_radix: ConfigRadixQH::empty(),
        },
        /* HeapConfig {
            family: HeapFamily::ConfigSimdQH,
            config_simd: ConfigSimdQH {
                bucket: Buckets::Vec,
                pivot: MedianOf3,
                multi_pivot: EqualMedianOf2,
                rebal: Rebalancing::ExponentialUpperBound,
                n: 16,
                sort: true,
                equal: false,
                p: 2,
                multi_thresh: 0,
            },
            config_radix: ConfigRadixQH::empty(),
        }, */
        HeapConfig {
            family: HeapFamily::ConfigSimdQH,
            config_simd: ConfigSimdQH {
                bucket: Buckets::Vec,
                pivot: MedianOf3,
                multi_pivot: EqualMedianOf2,
                rebal: Rebalancing::RandomizedRebalancing,
                n: 16,
                sort: true,
                equal: false,
                p: 2,
                multi_thresh: 0,
            },
            config_radix: ConfigRadixQH::empty(),
        },
        HeapConfig {
            family: HeapFamily::ConfigSimdQH,
            config_simd: ConfigSimdQH {
                bucket: Buckets::ListBlock,
                pivot: MedianOf3,
                multi_pivot: EqualMedianOf2,
                rebal: Rebalancing::RandomizedRebalancing,
                n: 16,
                sort: true,
                equal: false,
                p: 2,
                multi_thresh: 0,
            },
            config_radix: ConfigRadixQH::empty(),
        },
    ];

    // NoRebalancing,
    // PivotForgetting<3, 2048>,
    // ExponentialUpperBoundRebalancing,
    // RandomizedRebalancing<128>,

    let nano_timer = AnyMeasurement::F64(Box::new(NanoTimer::new()));
    let layer_measure = AnyMeasurement::Usize(Box::new(MeasureLayerStats::new()));
    let memory_measure = AnyMeasurement::F64(Box::new(MemoryConsumption::new()));

    let mut e = Experiment {
        name: "Rebalancing".to_string(),
        output_name: "rebalancing".to_string(),
        ns: (16..=24).step_by(2).map(|i| (2usize).pow(i)).collect(),
        heaps: heap_config,
        workloads: vec![Work::MonotoneCS, Work::MostlyPushDecreasing],
        types: vec![Type::I32, Type::I64],
        metrics: vec![nano_timer, layer_measure, memory_measure],
    };

    run(&mut e);
}

#[cfg(feature = "radix")]
fn radix() {
    let heap_config = vec![
        HeapConfig {
            family: HeapFamily::SimdQH,
            config_simd: ConfigSimdQH::empty(),
            config_radix: ConfigRadixQH::empty(),
        },
        HeapConfig {
            family: HeapFamily::ConfigSimdQH,
            config_simd: ConfigSimdQH {
                bucket: Buckets::Vec,
                pivot: MedianOf3,
                multi_pivot: EqualMedianOf2,
                rebal: Rebalancing::NoRebal,
                n: 16,
                sort: true,
                equal: false,
                p: 2,
                multi_thresh: 0,
            },
            config_radix: ConfigRadixQH::empty(),
        },
        HeapConfig {
            family: HeapFamily::RadixSimdQH,
            config_simd: ConfigSimdQH::empty(),
            config_radix: ConfigRadixQH { b: 2, thresh: 1024 },
        },
        HeapConfig {
            family: HeapFamily::RadixSimdQH,
            config_simd: ConfigSimdQH::empty(),
            config_radix: ConfigRadixQH { b: 2, thresh: 2048 },
        },
        HeapConfig {
            family: HeapFamily::ConfigSimdQH,
            config_simd: ConfigSimdQH {
                bucket: Buckets::Vec,
                pivot: MedianOf3,
                multi_pivot: EqualMedianOf2,
                rebal: Rebalancing::NoRebal,
                n: 16,
                sort: true,
                equal: false,
                p: 4,
                multi_thresh: 1024,
            },
            config_radix: ConfigRadixQH::empty(),
        },
        HeapConfig {
            family: HeapFamily::ConfigSimdQH,
            config_simd: ConfigSimdQH {
                bucket: Buckets::Vec,
                pivot: MedianOf3,
                multi_pivot: EqualMedianOf2,
                rebal: Rebalancing::NoRebal,
                n: 16,
                sort: true,
                equal: false,
                p: 4,
                multi_thresh: 2048,
            },
            config_radix: ConfigRadixQH::empty(),
        },
    ];

    let nano_timer = AnyMeasurement::F64(Box::new(NanoTimer::new()));
    let layer_measurement = AnyMeasurement::Usize(Box::new(MeasureLayerStats::new()));

    let mut e = Experiment {
        name: "Radix Partitioning".to_string(),
        output_name: "raidx".to_string(),
        ns: (19..=25).step_by(2).map(|i| (2usize).pow(i)).collect(),
        heaps: heap_config,
        workloads: vec![Work::MonotoneCS],
        types: vec![Type::I32, Type::I64],
        metrics: vec![nano_timer, layer_measurement],
    };

    run(&mut e);
}

fn read_paths() -> Vec<PathBuf> {
    // Scan `path` for all input files.
    let path = PathBuf::from("input/");
    let mut paths = vec![];
    let entries = std::fs::read_dir(&path).expect("Could not read input directory");
    for entry in entries {
        let entry = entry.expect("Could not read entry in input directory");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("gr") {
            paths.push(path);
        }
    }
    paths
}

fn read_graphs_u64() -> Vec<(String, Graph<u64>)> {
    let paths = read_paths();
    let mut graphs: Vec<(String, Graph<u64>)> = vec![];
    // Load all the graphs into memory
    for path in paths {
        let graph = Graph::<u64>::from_dimacs_instance(&path);
        eprintln!(
            "Loaded graph {} with {} vertices and {} edges",
            path.to_string_lossy(),
            graph.num_vertices(),
            graph.num_edges()
        );
        graphs.push((path.to_string_lossy().to_string(), graph));
    }

    graphs
}

fn read_graphs_u32() -> Vec<(String, Graph<u32>)> {
    let paths = read_paths();
    let mut graphs: Vec<(String, Graph<u32>)> = vec![];
    // Load all the graphs into memory
    for path in paths {
        let graph = Graph::<u32>::from_dimacs_instance(&path);
        eprintln!(
            "Loaded graph {} with {} vertices and {} edges",
            path.to_string_lossy(),
            graph.num_vertices(),
            graph.num_edges()
        );
        graphs.push((path.to_string_lossy().to_string(), graph));
    }

    graphs
}

fn run_graph<S: SimdElem<u64>, H: Heap<u64>>(
    graphs: &Vec<(String, Graph<u32>)>,
    writer: &mut BufWriter<File>,
) {
    println!("Run heap {}", sanitize(type_name::<H>().to_owned()));

    for (instance, graph) in graphs {
        println!("Run instance {}", instance);

        // Run once to warm up caches
        let f = DijkstraWorkload::setup::<H>(graph);
        f();

        println!("Run Dijkstra");
        let mut all_nanos = vec![];
        for _ in 0..REPEATS {
            let f = DijkstraWorkload::setup::<H>(graph);

            let start = std::time::Instant::now();
            let (push_count, pop_count) = f();
            let nanos = start.elapsed().as_nanos() as f64;

            all_nanos.push((2f64 * nanos) / (push_count + pop_count) as f64);
        }
        all_nanos.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = all_nanos[all_nanos.len() / 2];

        let line = format!(
            "{},DijkstraWorkload,\"{}\",{},{:.1}\n",
            type_name::<S>(),
            type_name::<H>(),
            instance,
            median
        );
        writer
            .write_all(sanitize_graph(sanitize(line)).as_bytes())
            .unwrap();
        writer.flush().unwrap();

        if !H::MONOTONE {
            println!("Run Prim");
            let f = PrimWorkload::setup::<H>(graph);
            f();

            let mut all_nanos = vec![];
            for _ in 0..REPEATS {
                let f = PrimWorkload::setup::<H>(graph);

                let start = std::time::Instant::now();
                let (push_count, pop_count) = f();
                let nanos = start.elapsed().as_nanos() as f64;

                all_nanos.push((2f64 * nanos) / (push_count + pop_count) as f64);
            }
            all_nanos.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = all_nanos[all_nanos.len() / 2];

            let line = format!(
                "{},PrimWorkload,\"{}\",{},{:.1}\n",
                type_name::<S>(),
                type_name::<H>(),
                instance,
                median
            );
            writer.write_all(sanitize(line).as_bytes()).unwrap();
            writer.flush().unwrap();
        }
    }
}

fn run_kv_graph<S: SimdElem<u64>, H: KvHeap<u64>>(
    graphs: &Vec<(String, Graph<u64>)>,
    writer: &mut BufWriter<File>,
) {
    println!("Run heap {}", sanitize(type_name::<H>().to_owned()));

    for (instance, graph) in graphs {
        println!("Run instance {}", instance);

        // Run once to warm up caches
        let f = DijkstraWorkload::setup_kv::<H>(graph);
        f();

        let mut all_nanos = vec![];
        let mut all_mem = vec![];
        for _ in 0..REPEATS {
            let f = DijkstraWorkload::setup_kv::<H>(graph);

            let start = std::time::Instant::now();
            let (h, push_count, pop_count) = f();
            let nanos = start.elapsed().as_nanos() as f64;
            let memory = h.capacity();

            all_nanos.push((2f64 * nanos) / (push_count + pop_count) as f64);
            all_mem.push((2f64 * memory as f64) / (push_count + pop_count) as f64);
        }
        all_nanos.sort_by(|a, b| a.partial_cmp(b).unwrap());
        all_mem.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median_time = all_nanos[all_nanos.len() / 2];
        let median_mem = all_mem[all_mem.len() / 2];

        let line = format!(
            "{},{},{},{:.1},{:.2}\n",
            type_name::<S>(),
            type_name::<H>(),
            instance,
            median_time,
            median_mem
        );
        writer
            .write_all(sanitize_graph(sanitize(line)).as_bytes())
            .unwrap();
        writer.flush().unwrap();
    }
}

#[cfg(feature = "graphs")]
fn graphs() {
    println!("Run experiemnt graphs.");

    println!("Read graphs.");
    let graphs = read_graphs_u32();

    let mut writer;
    let top_line = "simd_backend,workload,heap,instance,nano_seconds\n";

    #[cfg(target_feature = "avx512f")]
    type PlaceholderSimd = simd::Avx2;

    #[cfg(target_arch = "aarch64")]
    type PlaceholderSimd = simd::Neon;

    #[cfg(target_feature = "avx512f")]
    {
        writer = build_writer("graphs_avx");
        writer.write_all(top_line.as_bytes()).unwrap();

        writer.flush().expect("Flush the writer");

        run_graph::<simd::Avx2, SimdQuickHeap<u64, simd::Avx2>>(&graphs, &mut writer);
        run_graph::<simd::Avx512, SimdQuickHeap<u64, simd::Avx512>>(&graphs, &mut writer);
    }

    #[cfg(target_arch = "aarch64")]
    {
        writer = build_writer("graphs_neon");
        writer.write_all(top_line.as_bytes()).unwrap();

        run_graph::<simd::Neon, SimdQuickHeap<u64, Neon>>(&graphs, &mut writer);
    }

    // Run the other competitors
    run_graph::<
        PlaceholderSimd,
        scalar_quickheap::ScalarQuickHeap<u64, 3, false, { Search::LinearScan }>,
    >(&graphs, &mut writer);

    run_graph::<PlaceholderSimd, original_quickheap::OriginalQuickHeapU64>(&graphs, &mut writer);
    run_graph::<PlaceholderSimd, sequence_heap::SequenceHeapU64>(&graphs, &mut writer);
    run_graph::<PlaceholderSimd, s3q::S3qHeapU64>(&graphs, &mut writer);
    run_graph::<PlaceholderSimd, radix_heap::RadixHeapU64>(&graphs, &mut writer);

    run_graph::<PlaceholderSimd, impls::BinaryHeap<u64>>(&graphs, &mut writer);
    run_graph::<PlaceholderSimd, impls::OrxDaryHeap<u64, 8>>(&graphs, &mut writer);
    run_graph::<PlaceholderSimd, impls::WeakHeap<u64>>(&graphs, &mut writer);
}

#[cfg(feature = "decrease_key")]
fn decrease_key() {
    println!("Run experiemnt decrease_key.");

    println!("Read graphs.");
    let graphs = read_graphs_u64();

    let mut writer;
    let top_line = "simd_backend,heap,instance,nano_seconds,bytes_per_elem\n";
    #[cfg(target_feature = "avx512f")]
    {
        writer = build_writer("decrease_key_avx");
        writer.write_all(top_line.as_bytes()).unwrap();

        writer.flush().expect("Flush the writer");

        // Run all the competitors
        run_kv_graph::<simd::Avx2, KvSimdQuickHeap<u64, simd::Avx2, SimpleDK<u64>>>(
            &graphs,
            &mut writer,
        );
        run_kv_graph::<simd::Avx2, KvSimdQuickHeap<u64, simd::Avx2, DelTokenSort<u64, simd::Avx2>>>(
            &graphs,
            &mut writer,
        );
        run_kv_graph::<simd::Avx2, KvSimdQuickHeap<u64, simd::Avx2, DelTokenHash<u64, simd::Avx2>>>(
            &graphs,
            &mut writer,
        );
        /* run_kv_graph::<
            simd::Avx2,
            KvSimdQuickHeap<u64, simd::Avx2, DelTokenCountingBloom<u64, simd::Avx2, 16>>,
        >(&graphs, &mut writer); */

        run_kv_graph::<simd::Avx512, KvSimdQuickHeap<u64, simd::Avx512, SimpleDK<u64>>>(
            &graphs,
            &mut writer,
        );
        run_kv_graph::<
            simd::Avx512,
            KvSimdQuickHeap<u64, simd::Avx512, DelTokenSort<u64, simd::Avx512>>,
        >(&graphs, &mut writer);
        run_kv_graph::<
            simd::Avx512,
            KvSimdQuickHeap<u64, simd::Avx512, DelTokenHash<u64, simd::Avx512>>,
        >(&graphs, &mut writer);
        /* run_kv_graph::<
            simd::Avx512,
            KvSimdQuickHeap<u64, simd::Avx512, DelTokenCountingBloom<u64, simd::Avx512, 16>>,
        >(&graphs, &mut writer); */
    }

    #[cfg(target_arch = "aarch64")]
    {
        writer = build_writer("decrease_key_neon");
        writer.write_all(top_line.as_bytes());

        run_kv_graph::<simd::Neon, KvSimdQuickHeap<u64, simd::Neon, SimpleDK<u64>>>(
            &graphs,
            &mut writer,
        );
        run_kv_graph::<simd::Neon, KvSimdQuickHeap<u64, simd::Neon, DelTokenSort<u64, simd::Neon>>>(
            &graphs,
            &mut writer,
        );
        run_kv_graph::<simd::Neon, KvSimdQuickHeap<u64, simd::Neon, DelTokenHash<u64, simd::Neon>>>(
            &graphs,
            &mut writer,
        );
        /* run_kv_graph::<
            simd::Neon,
            KvSimdQuickHeap<u64, simd::Neon, DelTokenCountingBloom<u64, simd::Neon, 16>>,
        >(&graphs, &mut writer); */
    }
}

fn sanitize(s: String) -> String {
    s.replace("simd::", "")
        .replace(" ", "")
        .replace("input/", "")
        .replace("_graph.gr", "")
        .replace("sequence_heap::", "")
        .replace("s3q::", "")
        .replace("radix_heap::", "")
        .replace("scalar_quickheap::", "")
        .replace("original_quickheap::", "")
        .replace("Search::", "")
        .replace("alloc::collections::binary_heap::", "")
        .replace("orx_priority_queue::dary::daryheap::", "")
        .replace("weakheap::", "")
        .replace("radix_quickheap::", "")
        .replace("quickheap_dev::", "")
        .replace("quickheap::", "")
        .replace("buckets::", "")
        .replace("<i32,", "<")
        .replace("<i64,", "<")
        .replace("<i32>", "")
        .replace("<i64>", "")
        .replace("<Neon>", "")
        .replace("<Avx2>", "")
        .replace("<Avx512>", "")
        .replace("vec_bucket::", "")
        .replace("decrease_key::", "")
        .replace("rebalancing_strategies::", "")
        .replace("pivot_strategies::", "")
        .replace("block_partitioning::", "")
        .replace("bench::", "")
        .replace("workloads::", "")
        .replace("list_block_bucket::", "")
        .replace("I32", "")
        .replace("I64", "")
        .replace("U32", "")
        .replace("U64", "")
        .replace("", "")
        .replace("", "")
        .replace("", "")
        .replace("", "")
        .replace("", "")
        .replace("", "")
}

fn sanitize_graph(s: String) -> String {
    s.replace(",Avx2", "")
        .replace(",Avx512", "")
        .replace(",Neon", "")
        .replace("u64", "")
        .replace("<>", "")
        .replace("<,", ">")
}

fn run_config<T: Elem, S: SimdElem<T>, H: Heap<T>, W: Workload>(
    e: &mut Experiment,
    writer: &mut BufWriter<File>,
) {
    println!(
        "Run heap: {}",
        sanitize(std::any::type_name::<H>().to_string())
    );
    for &n in e.ns.iter() {
        for metric in e.metrics.iter_mut() {
            metric.reset();
        }

        for _ in 0..REPEATS {
            // Perform the setup
            let f = W::setup::<T, H>(n as u64);

            // Start all metrics
            for metric in e.metrics.iter_mut() {
                metric.start();
            }

            let h = f();

            let stats = HeapStats {
                capacity: h.capacity(),
            };

            // Stop all metrics
            for metric in e.metrics.iter_mut() {
                metric.stop(&stats);
            }
        }

        // Generate the result line for this config
        let mut result: String = String::new();

        result.push_str(std::any::type_name::<T>());
        result.push_str(",");
        result.push_str(std::any::type_name::<S>());
        result.push_str(",");
        result.push_str(&n.to_string());
        result.push_str(",");
        result.push_str(std::any::type_name::<W>());
        result.push_str(",");
        result.push_str("\"");
        result.push_str(
            &std::any::type_name::<H>()
                .replace("<Avx2>", "")
                .replace("<Avx512>", "")
                .replace("<Neon>", "")
                .replace("Avx2,", "")
                .replace("Neon", "")
                .replace("Avx512,", ""),
        );
        result.push_str("\"");

        for metric in e.metrics.iter_mut() {
            metric.norm_by_ops(n * W::NORMALIZATION as usize);
            let s = metric.get();
            result.push_str(",");
            result.push_str(&s);
        }

        result.push_str("\n");

        writer
            .write_all(sanitize(result).as_bytes())
            .expect("Something went wrong writing to the file.");
        writer
            .flush()
            .expect("Something went wrong writing to the file.");
    }
}

macro_rules! dispatch_radix_ll {
    ($e:expr, $t:ty, $w:ty, $simd:ty, $b:literal, $ll:expr, $writer: expr) => {
        match $ll {
            512 => run_config::<$t, $simd, RadixSimdQuickHeap<$t, $simd, $b, 512>, $w>($e, $writer),
            1024 => {
                run_config::<$t, $simd, RadixSimdQuickHeap<$t, $simd, $b, 1024>, $w>($e, $writer)
            }
            2048 => {
                run_config::<$t, $simd, RadixSimdQuickHeap<$t, $simd, $b, 2048>, $w>($e, $writer)
            }
            x => panic!("unsupported radix thresh: {}", x),
        }
    };
}

macro_rules! dispatch_radix {
    ($e:expr, $t:ty, $w:ty, $simd:ty, $b:expr, $ll:expr, $writer: expr) => {
        match $b {
            2 => dispatch_radix_ll!($e, $t, $w, $simd, 2, $ll, $writer),
            3 => dispatch_radix_ll!($e, $t, $w, $simd, 3, $ll, $writer),
            x => panic!("unsupported radix b: {}", x),
        }
    };
}

macro_rules! dispatch_config_simdqh {
    ($e: expr, $t: ty, $w: ty, $simd: ty, $c: expr, $writer: expr) => {
        match (
            $c.bucket,
            $c.pivot,
            $c.multi_pivot,
            $c.rebal,
            $c.n,
            $c.sort,
            $c.equal,
            $c.p,
            $c.multi_thresh,
        ) {
            /* Default Config for ConfigurableSimdQH */
            (
                Buckets::Vec,
                Pivots::MedianOf3,
                Pivots::EqualMedianOf2,
                Rebalancing::NoRebal,
                16,
                true,
                false,
                2,
                0,
            ) => {
                run_config::<
                    $t,
                    $simd,
                    ConfigurableSimdQuickHeap<
                        $t,
                        VecBucket<$t>,
                        FlatPartitioning<$simd>,
                        $simd,
                        MedianOfM<3>,
                        EqualMultiMedianOfM<2>,
                        NoRebalancing,
                        16,
                        0,
                        0,
                        true,
                        false,
                        2,
                        0,
                    >,
                    $w,
                >($e, $writer);
            }
            /* Rebalancing */
            (
                Buckets::Vec,
                Pivots::MedianOf3,
                Pivots::EqualMedianOf2,
                Rebalancing::PivotForgetting,
                16,
                true,
                false,
                2,
                0,
            ) => {
                run_config::<
                    $t,
                    $simd,
                    ConfigurableSimdQuickHeap<
                        $t,
                        VecBucket<$t>,
                        FlatPartitioning<$simd>,
                        $simd,
                        MedianOfM<3>,
                        EqualMultiMedianOfM<2>,
                        PivotForgetting<2, 256>,
                        16,
                        0,
                        0,
                        true,
                        false,
                        2,
                        0,
                    >,
                    $w,
                >($e, $writer);
            },
            (
                Buckets::ListBlock,
                Pivots::MedianOf3,
                Pivots::EqualMedianOf2,
                Rebalancing::PivotForgetting,
                16,
                true,
                false,
                2,
                0,
            ) => {
                run_config::<
                    $t,
                    $simd,
                    ConfigurableSimdQuickHeap<
                        $t,
                        ListBlockBucket<$t, 224, 256>,
                        BlockedPartitioning<$simd, 224 ,256>,
                        $simd,
                        MedianOfM<3>,
                        EqualMultiMedianOfM<2>,
                        PivotForgetting<2, 256>,
                        16,
                        224,
                        256,
                        true,
                        false,
                        2,
                        0,
                    >,
                    $w,
                >($e, $writer);
            }
            (
                Buckets::Vec,
                Pivots::MedianOf3,
                Pivots::EqualMedianOf2,
                Rebalancing::ExponentialUpperBound,
                16,
                true,
                false,
                2,
                0,
            ) => {
                run_config::<
                    $t,
                    $simd,
                    ConfigurableSimdQuickHeap<
                        $t,
                        VecBucket<$t>,
                        FlatPartitioning<$simd>,
                        $simd,
                        MedianOfM<3>,
                        EqualMultiMedianOfM<2>,
                        ExponentialUpperBoundRebalancing,
                        16,
                        0,
                        0,
                        true,
                        false,
                        2,
                        0,
                    >,
                    $w,
                >($e, $writer);
            }
            (
                Buckets::Vec,
                Pivots::MedianOf3,
                Pivots::EqualMedianOf2,
                Rebalancing::RandomizedRebalancing,
                16,
                true,
                false,
                2,
                0,
            ) => {
                run_config::<
                    $t,
                    $simd,
                    ConfigurableSimdQuickHeap<
                        $t,
                        VecBucket<$t>,
                        FlatPartitioning<$simd>,
                        $simd,
                        MedianOfM<3>,
                        EqualMultiMedianOfM<2>,
                        RandomizedRebalancing<128>,
                        16,
                        0,
                        0,
                        true,
                        false,
                        2,
                        0,
                    >,
                    $w,
                >($e, $writer);
            },
            (
                Buckets::ListBlock,
                Pivots::MedianOf3,
                Pivots::EqualMedianOf2,
                Rebalancing::RandomizedRebalancing,
                16,
                true,
                false,
                2,
                0,
            ) => {
                run_config::<
                    $t,
                    $simd,
                    ConfigurableSimdQuickHeap<
                        $t,
                        ListBlockBucket<$t, 224, 256>,
                        BlockedPartitioning<$simd, 224, 256>,
                        $simd,
                        MedianOfM<3>,
                        EqualMultiMedianOfM<2>,
                        RandomizedRebalancing<128>,
                        16,
                        224,
                        256,
                        true,
                        false,
                        2,
                        0,
                    >,
                    $w,
                >($e, $writer);
            }
            /* Equal Buckets ConfigurableSimdQH */
            (
                Buckets::Vec,
                Pivots::MedianOf3,
                Pivots::EqualMedianOf2,
                Rebalancing::NoRebal,
                16,
                true,
                true,
                2,
                0,
            ) => {
                run_config::<
                    $t,
                    $simd,
                    ConfigurableSimdQuickHeap<
                        $t,
                        VecBucket<$t>,
                        FlatPartitioning<$simd>,
                        $simd,
                        MedianOfM<3>,
                        EqualMultiMedianOfM<2>,
                        NoRebalancing,
                        16,
                        0,
                        0,
                        true,
                        true,
                        2,
                        0,
                    >,
                    $w,
                >($e, $writer);
            }
            /* Multiway Splits */
            (
                Buckets::Vec,
                Pivots::MedianOf3,
                Pivots::EqualMedianOf2,
                Rebalancing::NoRebal,
                16,
                true,
                false,
                4,
                1024,
            ) => {
                run_config::<
                    $t,
                    $simd,
                    ConfigurableSimdQuickHeap<
                        $t,
                        VecBucket<$t>,
                        FlatPartitioning<$simd>,
                        $simd,
                        MedianOfM<3>,
                        EqualMultiMedianOfM<2>,
                        NoRebalancing,
                        16,
                        0,
                        0,
                        true,
                        false,
                        4,
                        1024,
                    >,
                    $w,
                >($e, $writer);
            }
            (
                Buckets::Vec,
                Pivots::MedianOf3,
                Pivots::EqualMedianOf2,
                Rebalancing::NoRebal,
                16,
                true,
                false,
                4,
                2048,
            ) => {
                run_config::<
                    $t,
                    $simd,
                    ConfigurableSimdQuickHeap<
                        $t,
                        VecBucket<$t>,
                        FlatPartitioning<$simd>,
                        $simd,
                        MedianOfM<3>,
                        EqualMultiMedianOfM<2>,
                        NoRebalancing,
                        16,
                        128,
                        128,
                        true,
                        false,
                        4,
                        2048,
                    >,
                    $w,
                >($e, $writer);
            }
            _ => panic!("unsupported simd qh configuration."),
        }
    };
}

macro_rules! dispatch_heap {
    ($e: expr, $t: ty, $w: ty, $simd: ty, $writer: expr) => {
        println!("Run workload: {}", sanitize(std::any::type_name::<$w>().to_string()));
        for h in $e.heaps.clone() {
            match h.family {
                HeapFamily::SimdQH => {
                    run_config::<$t, $simd, SimdQuickHeap<$t, $simd>, $w>($e, $writer);
                }
                HeapFamily::ConfigSimdQH => {
                    dispatch_config_simdqh!($e, $t, $w, $simd, h.config_simd, $writer)
                }
                HeapFamily::RadixSimdQH => {
                    dispatch_radix!($e, $t, $w, $simd, h.config_radix.b, h.config_radix.thresh, $writer);
                }

                HeapFamily::ScalarQH => {
                    //
                    run_config::<$t, $simd, scalar_quickheap::ScalarQuickHeap<$t, 3, false, { Search::LinearScan }>, $w>($e, $writer);
                }
                HeapFamily::OriginalQH => {
                    if TypeId::of::<$t>() == TypeId::of::<i32>() {
                        run_config::<i32, $simd, original_quickheap::OriginalQuickHeapI32, $w>($e, $writer);
                    }
                    if TypeId::of::<$t>() == TypeId::of::<i64>() {
                        run_config::<i64, $simd, original_quickheap::OriginalQuickHeapI64, $w>($e, $writer);
                    }
                }
                HeapFamily::SequenceHeap => {
                    if TypeId::of::<$t>() == TypeId::of::<i32>() {
                        run_config::<i32, $simd, sequence_heap::SequenceHeapI32, $w>($e, $writer);
                    }
                    if TypeId::of::<$t>() == TypeId::of::<i64>() {
                        run_config::<i64, $simd, sequence_heap::SequenceHeapI64, $w>($e, $writer);
                    }
                }
                HeapFamily::S3qHeap => {
                    if TypeId::of::<$t>() == TypeId::of::<i32>() {
                        run_config::<i32, $simd, s3q::S3qHeapI32, $w>($e, $writer);
                    }
                    if TypeId::of::<$t>() == TypeId::of::<i64>() {
                        run_config::<i64, $simd, s3q::S3qHeapI64, $w>($e, $writer);
                    }
                }
                HeapFamily::RadixHeap => {
                    if TypeId::of::<$t>() == TypeId::of::<i32>() {
                        run_config::<i32, $simd, radix_heap::RadixHeapI32, $w>($e, $writer);
                    }
                    if TypeId::of::<$t>() == TypeId::of::<i64>() {
                        run_config::<i64, $simd, radix_heap::RadixHeapI64, $w>($e, $writer);
                    }
                }
                HeapFamily::BinaryHeap => {
                    run_config::<$t, $simd, impls::BinaryHeap<$t>, $w>($e, $writer);
                }
                HeapFamily::Orx8aryHeap => {
                    run_config::<$t, $simd, impls::OrxDaryHeap<$t, 8>, $w>($e, $writer);
                }
                HeapFamily::WeakHeap => {
                    run_config::<$t, $simd, impls::WeakHeap<$t>, $w>($e, $writer);
                }
            }
        }
    };
}

macro_rules! dispatch_workload {
    ($e: expr, $t: ty, $simd: ty, $writer: expr) => {
        println!("Run for type {}", std::any::type_name::<$t>());
        for w in $e.workloads.clone() {
            match w {
                Work::Heapsort => {dispatch_heap!($e, $t, workloads::HeapSort, $simd, $writer);},
                // Work::MonotoneW => {dispatch_heap!($e, $t, workloads::MonotoneWiggle, $simd, $writer);},
                // Work::RandomW => {dispatch_heap!($e, $t, workloads::RandomWiggle, $simd, $writer);},
                Work::MonotoneCS => {dispatch_heap!($e, $t, workloads::MonotoneConstantSize, $simd, $writer);},
                // Work::RandomCS => {dispatch_heap!($e, $t, workloads::RandomConstantSize, $simd, $writer);},
                Work::MostlyPushDecreasing => {dispatch_heap!($e, $t, workloads::MostlyPushDecreasing, $simd, $writer);},
                Work::RandomManyEqual => {dispatch_heap!($e, $t, workloads::RandomConstantSizeManyEqual, $simd, $writer);},
            }
        }
    };
}

macro_rules! dispatch_type {
    ($e: expr, $simd: ty, $writer: expr) => {
        println!("Run backend {}", sanitize(std::any::type_name::<$simd>().to_string()));
        for t in $e.types.clone() {
            match t {
                Type::I32 => {
                    dispatch_workload!($e, i32, $simd, $writer);
                }
                Type::I64 => {
                    dispatch_workload!($e, i64, $simd, $writer);
                } /*
                  Type::U32 => {
                      dispatch_workload!($e, u32, $simd, $writer);
                  }
                  Type::U64 => {
                      dispatch_workload!($e, u64, $simd, $writer);
                  } */
            }
        }
    };
}

macro_rules! dispatch_backend {
    ($e: expr, $writer: expr) => {
        #[cfg(target_arch = "aarch64")]
        dispatch_type!($e, simd::Neon, $writer);

        #[cfg(target_feature = "avx2")]
        dispatch_type!($e, simd::Avx2, $writer);

        #[cfg(target_feature = "avx512f")]
        dispatch_type!($e, simd::Avx512<true>, $writer);
    };
}

fn main() {
    println!("\x1B[2J\x1B[1;1H");
    println!("| - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - |");
    println!("| - - - - - - ENGINEERING QUICK HEAPS EXPERIMENTS - - - - - - |");
    println!("| - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - |");

    // Benchmarks using different worklaods
    #[cfg(feature = "competitors")]
    competitors();

    #[cfg(feature = "simd_qhs")]
    simd_qhs();

    #[cfg(feature = "implicit_equal_buckets")]
    implicit_equal_buckets();

    #[cfg(feature = "multiway")]
    multiway();

    #[cfg(feature = "radix")]
    radix();

    #[cfg(feature = "pivots")]
    pivots();

    #[cfg(feature = "rebalancing")]
    rebalancing();

    // Graph benchmarks
    #[cfg(feature = "graphs")]
    graphs();
    #[cfg(feature = "decrease_key")]
    decrease_key();
}

fn build_writer(name: &str) -> BufWriter<File> {
    let path =
        String::from(env!("CARGO_MANIFEST_DIR")) + "/evals/data/experiments/" + name + ".csv";
    let file = File::create(path).unwrap();
    BufWriter::with_capacity(1 << 15, file)
}

fn run(e: &mut Experiment) {
    println!("Run experiment: {}", e.name);

    let backend_name;
    #[cfg(target_arch = "aarch64")]
    {
        backend_name = "_neon";
    }

    #[cfg(target_feature = "avx2")]
    {
        backend_name = "_avx";
    }

    // Create and open the file for the experiment
    let name = e.output_name.to_owned() + backend_name;
    let mut writer = build_writer(&name);
    let mut top_line = "elem_type,simd_backend,n,workload,heap".to_string();

    for metric in e.metrics.iter() {
        top_line += ",";
        top_line += &metric.get_name();
    }

    top_line += "\n";

    writer
        .write_all(top_line.as_bytes())
        .expect("Something went wrong writing to the file.");

    writer
        .flush()
        .expect("Something went wrong writing to the file.");

    dispatch_backend!(e, &mut writer);
}
