#![feature(where_clause_attrs)]

use clap::Parser;
use original_quickheap::OriginalQuickHeap;
use quickheap::dijkstra::DijkstraQuery;
use quickheap::graph::Graph;
use quickheap::pivot_strategies::MedianOfM;
use quickheap::prim::PrimMST;
use quickheap::scalar_quickheap::Search;
#[cfg(feature = "avx2")]
use quickheap::simd::Avx2;
#[cfg(feature = "avx512")]
use quickheap::simd::Avx512;

use quickheap::*;
use serde::Serialize;
use std::any::type_name;
use std::cell::RefCell;
use std::hint::black_box;
use std::path::PathBuf;

const REPEATS: usize = 3;

trait GraphWorkload {
    fn setup<H: Heap<u64>>(graph: &Graph<u32>) -> impl FnOnce();
}

// One-to-all dijkstra query from a random starting vertex
struct DijkstraWorkload;

// Construct mst using the prim algorithm from a random starting vertex
struct PrimWorkload;

impl GraphWorkload for DijkstraWorkload {
    fn setup<H: Heap<u64>>(graph: &Graph<u32>) -> impl FnOnce() {
        let mut dijkstra = DijkstraQuery::<H>::new(graph);
        move || {
            dijkstra.run_all(0);
            black_box(dijkstra);
        }
    }
}

impl GraphWorkload for PrimWorkload {
    fn setup<H: Heap<u64>>(graph: &Graph<u32>) -> impl FnOnce() {
        let mut prim = PrimMST::<H>::new(graph);
        move || {
            prim.compute_mst_from_vertex(0);
            black_box(prim);
        }
    }
}

#[derive(Serialize)]
struct Result {
    heap: &'static str,
    graph: String,
    workload: &'static str,
    vertices: usize,
    edges: usize,
    repeat: usize,
    nanos: f64,
}

thread_local! {
    static CSV_WRITER: RefCell<csv::Writer<std::io::Stdout>> =
        RefCell::new(csv::Writer::from_writer(std::io::stdout()));
}

/// Runs the workload `REPEATS` times, writes each run as a CSV row, and
/// returns the median nanos (used to decide whether to skip larger `n`).
fn time_workload<H: Heap<u64>, W: GraphWorkload>(instance: &str, graph: &Graph<u32>) -> f64 {
    // Warmup: run once to populate instruction cache, page tables, and branch predictors.
    let f = W::setup::<H>(graph);
    f();

    let mut all_nanos = vec![];

    for repeat in 0..REPEATS {
        let f = W::setup::<H>(graph);

        let start = std::time::Instant::now();
        f();
        let nanos = start.elapsed().as_nanos() as f64;

        let result = Result {
            heap: type_name::<H>(),
            graph: instance.to_string(),
            workload: type_name::<W>(),
            vertices: graph.num_vertices(),
            edges: graph.num_edges(),
            repeat,
            nanos,
        };

        CSV_WRITER.with(|w| w.borrow_mut().serialize(&result).unwrap());
        all_nanos.push(result.nanos);
    }

    all_nanos.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_nanos[all_nanos.len() / 2]
}

pub fn bench<H: Heap<u64>>(graphs: &Vec<(String, Graph<u32>)>) {
    for (instance, graph) in graphs {
        eprint!("{:<90} {:<30}", type_name::<H>(), instance);

        let t = time_workload::<H, DijkstraWorkload>(instance, graph);
        eprint!(" {t:>12.0}");
        if !H::MONOTONE {
            let t = time_workload::<H, PrimWorkload>(instance, graph);
            eprint!(" {t:>12.0}");
        }

        eprintln!();
    }
}

#[derive(clap::Parser)]
struct Args {
    /// Single .gr file, or directory containing .gr files.
    path: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();

    let mut graphs: Vec<(String, Graph<u32>)> = vec![];

    let path = args.path.unwrap_or_else(|| PathBuf::from("input/"));
    // Scan `path` for all input files.
    let paths = if path.is_file() {
        vec![path]
    } else {
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
    };

    // Load all the graphs into memory
    for path in paths {
        let graph = Graph::from_dimacs_instance(&path);
        eprintln!(
            "Loaded graph {} with {} vertices and {} edges",
            path.to_string_lossy(),
            graph.num_vertices(),
            graph.num_edges()
        );
        graphs.push((path.to_string_lossy().to_string(), graph));
    }

    // QUICKHEAP
    #[cfg(feature = "avx2")]
    bench::<simd_quickheap::SimdQuickHeap<u64, Avx2, MedianOfM<3>, 16, true>>(&graphs);
    #[cfg(feature = "avx512")]
    bench::<simd_quickheap::SimdQuickHeap<u64, Avx512<true>, MedianOfM<3>, 16, true>>(&graphs);

    // SCALAR QUICKHEAP
    bench::<scalar_quickheap::ScalarQuickHeap<u64, 1, false, { Search::LinearScan }>>(&graphs);

    // ENGINEERED
    #[cfg(feature = "ffi")]
    bench::<sequence_heap::SequenceHeapU64>(&graphs);
    #[cfg(feature = "ffi")]
    bench::<s3q::S3qHeapU64>(&graphs);
    #[cfg(feature = "ffi")]
    bench::<boost_heap::BoostDary4HeapU64>(&graphs);
    #[cfg(feature = "ffi")]
    bench::<boost_heap::BoostFibHeapU64>(&graphs);
    #[cfg(feature = "ffi")]
    bench::<boost_heap::BoostPairingHeapU64>(&graphs);
    #[cfg(feature = "ffi")]
    bench::<boost_heap::BoostBinomialHeapU64>(&graphs);
    #[cfg(feature = "ffi")]
    bench::<boost_heap::BoostSkewHeapU64>(&graphs);

    // REIMPLS
    // bench::<binary_heap::CustomBinaryHeap<u64>>(&graphs);
    // bench::<dary_heap::CustomDaryHeap<u64, 2>>(&graphs);
    // bench::<dary_heap::CustomDaryHeap<u64, 3>>(&graphs);
    // bench::<dary_heap::CustomDaryHeap<u64, 4>>(&graphs);
    // bench::<dary_heap::CustomDaryHeap<u64, 8>>(&graphs);
    // bench::<dary_heap::CustomDaryHeap<u64, 16>>(&graphs);
    bench::<original_quickheap::OriginalQuickHeap<u64>>(&graphs);

    // BASELINE
    //* bench::<impls::BinaryHeap<u64>>(&graphs);

    // DARY
    // bench::<impls::DaryHeap<u64, 2>>(&graphs);
    // bench::<impls::DaryHeap<u64, 4>>(&graphs);
    // bench::<impls::DaryHeap<u64, 8>>(&graphs);
    // bench::<impls::DaryHeap<u64, 16>>(&graphs);
    // bench::<impls::OrxDaryHeap<u64, 2>>(&graphs);
    //* bench::<impls::OrxDaryHeap<u64, 4>>(&graphs);
    //* bench::<impls::OrxDaryHeap<u64, 8>>(&graphs);
    // bench::<impls::OrxDaryHeap<u64, 16>>(&graphs);

    // AMORTIZED
    bench::<impls::PairingHeap<u64>>(&graphs);
    bench::<impls::FibonacciHeap<u64>>(&graphs);
    bench::<impls::WeakHeap<u64>>(&graphs);

    // MONOTONE
    //* bench::<impls::RadixHeap<u64>>(&graphs);

    // SET
    // bench::<impls::BTreeSet<u64>>(&graphs);
    // bench::<impls::RevBTreeSet<u64>>(&graphs);
    // bench::<impls::IndexSetBTreeSet<u64>>(&graphs);
    // bench::<impls::IndexSetRevBTreeSet<u64>>(&graphs);

    CSV_WRITER.with(|w| w.borrow_mut().flush().unwrap());
}
