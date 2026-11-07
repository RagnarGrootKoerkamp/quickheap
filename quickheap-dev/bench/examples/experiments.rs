use std::fs::File;
use std::io::{BufWriter, Write};
use std::time;

use quickheap::SimdElem;
#[cfg(feature = "layer_stats")]
use quickheap_dev::stats::layer_stats;

use quickheap::SimdQuickHeap;
use quickheap::simd;
use quickheap_dev::ConfigurableSimdQuickHeap;
use quickheap_dev::RadixSimdQuickHeap;

use quickheap_dev::{
    buckets::{
        block_partitioning::FlatPartitioning, list_block_bucket::ListBlockBucket,
        vec_block_bucket::VecBlockBucket, vec_bucket::VecBucket,
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
    Heap,
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
    SimdQH,
    ConfigSimdQH,
    RadixSimdQH,
}

#[derive(Clone, Copy)]
pub enum Work {
    Heapsort,
    MonotoneCS,
    RandomManyEqual,
    MostlyPush,
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
    k: usize,
    cap: usize,
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
            k: 0,
            cap: 0,
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
}

pub enum AnyMeasurement {
    Usize(Box<dyn MetricMeasurement<usize>>),
    U128(Box<dyn MetricMeasurement<u128>>),
    F64(Box<dyn MetricMeasurement<f64>>),
}

impl AnyMeasurement {
    fn reset(&mut self) {
        match self {
            Self::Usize(b) => b.reset(),
            Self::U128(b) => b.reset(),
            Self::F64(b) => b.reset(),
        }
    }

    fn start(&mut self) {
        match self {
            Self::Usize(b) => b.start(),
            Self::U128(b) => b.start(),
            Self::F64(b) => b.start(),
        }
    }

    fn stop(&mut self, stats: &HeapStats) {
        match self {
            Self::Usize(b) => b.stop(stats),
            Self::U128(b) => b.stop(stats),
            Self::F64(b) => b.stop(stats),
        }
    }

    fn norm_by_ops(&mut self, n: usize) {
        match self {
            Self::Usize(b) => b.norm_by_ops(n),
            Self::U128(b) => b.norm_by_ops(n),
            Self::F64(b) => b.norm_by_ops(n),
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
        }
    }

    fn get_name(&self) -> String {
        match self {
            Self::Usize(b) => b.get_name(),
            Self::U128(b) => b.get_name(),
            Self::F64(b) => b.get_name(),
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
                k: 0,
                cap: 0,
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
        workloads: vec![Work::Heapsort, Work::MonotoneCS],
        types: vec![Type::I32, Type::I64],
        metrics: vec![nano_timer],
    };

    run(&mut e);
}

fn implicit_equal_buckets() {
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
                k: 0,
                cap: 0,
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
                k: 0,
                cap: 0,
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

fn multiway() {
    let heap_config = vec![
        HeapConfig {
            family: HeapFamily::SimdQH,
            config_simd: ConfigSimdQH::empty(),
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
                k: 0,
                cap: 0,
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
                k: 0,
                cap: 0,
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
        name: "Multiway Partitioning".to_string(),
        output_name: "multiway".to_string(),
        ns: (19..=25).step_by(2).map(|i| (2usize).pow(i)).collect(),
        heaps: heap_config,
        workloads: vec![Work::MonotoneCS],
        types: vec![Type::I32, Type::I64],
        metrics: vec![nano_timer, layer_measurement],
    };

    run(&mut e);
}

fn sanitize(s: String) -> String {
    s.replace("simd::", "")
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
        .replace("rebalancing_strategies::", "")
        .replace("pivot_strategies::", "")
        .replace("block_partitioning::", "")
        .replace("bench::", "")
        .replace("workloads::", "")
        .replace(" ", "")
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
        result.push_str(std::any::type_name::<H>());
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

/*
*
bucket: Buckets,
pivot: Pivots,
multi_pivot: Pivots,
rebal: Rebalancing,
n: usize,
k: usize,
cap: usize,
sort: bool,
equal: bool,
p: usize,
multi_thresh: usize,
*/

macro_rules! dispatch_config_simdqh {
    ($e: expr, $t: ty, $w: ty, $simd: ty, $c: expr, $writer: expr) => {
        match (
            $c.bucket,
            $c.pivot,
            $c.multi_pivot,
            $c.rebal,
            $c.n,
            $c.k,
            $c.cap,
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
                0,
                0,
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
            /* Equal Buckets ConfigurableSimdQH */
            (
                Buckets::Vec,
                Pivots::MedianOf3,
                Pivots::EqualMedianOf2,
                Rebalancing::NoRebal,
                16,
                0,
                0,
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
                0,
                0,
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
                0,
                0,
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
                Work::MostlyPush => {dispatch_heap!($e, $t, workloads::MostlyPushDecreasing, $simd, $writer);},
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
    println!("| - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - |");
    println!("| - - - - - - ENGINEERING QUICK HEAPS EXPERIMENTS - - - - - - |");
    println!("| - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - |");

    #[cfg(feature = "simd_qhs")]
    simd_qhs();

    #[cfg(feature = "implicit_equal_buckets")]
    implicit_equal_buckets();

    #[cfg(feature = "multiway")]
    multiway();

    // TODO: rebalancing();
    // TODO: finish/check multiway();
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
    let path = String::from(env!("CARGO_MANIFEST_DIR"))
        + &String::from("/evals/data/experiments/")
        + &e.output_name
        + &backend_name
        + ".csv";

    let file = File::create(path).unwrap();
    let mut writer = BufWriter::with_capacity(1 << 15, file);

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
