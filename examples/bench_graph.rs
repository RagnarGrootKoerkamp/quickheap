#![feature(where_clause_attrs)]

use quickheap::dijkstra::DijkstraQuery;
use quickheap::graph::Graph;
use quickheap::prim::PrimMST;
#[cfg(feature = "avx2")]
use quickheap::simd::Avx2;
#[cfg(feature = "avx512")]
use quickheap::simd::Avx512;

use quickheap::*;
use serde::Serialize;
use std::any::type_name;
use std::cell::RefCell;
use std::hint::black_box;

const REPEATS: usize = 10;

const BASE_PATH: &'static str = "./input/";
const SUFF: &'static str = ".gr";
const GRAPH_INSTANCES: &[&str] = &["GER_graph"]; // TODO: Add more graph instances

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
        eprint!("{:<70} {}", type_name::<H>(), instance);

        let t = time_workload::<H, DijkstraWorkload>(instance, graph);
        eprint!(" {t:>8.2}");
        if !H::MONOTONE {
            let t = time_workload::<H, PrimWorkload>(instance, graph);
            eprint!(" {t:>8.2}");
        }

        eprintln!();
    }
}

fn main() {
    let mut graphs: Vec<(String, Graph<u32>)> = vec![];

    // Load all the graphs into memory
    for instance in GRAPH_INSTANCES {
        let path = format!("{}{}{}", BASE_PATH, instance, SUFF);
        graphs.push((instance.to_string(), Graph::from_dimacs_instance(&path)));
    }

    eprintln!("QUICKHEAP");

    #[cfg(feature = "avx2")]
    bench::<simd_quickheap::SimdQuickHeap<u64, Avx2, 16, 1>>(&graphs);
    #[cfg(feature = "avx512")]
    bench::<simd_quickheap::SimdQuickHeap<u64, Avx512<false>, 16, 1>>(&graphs);
    #[cfg(feature = "avx512")]
    bench::<simd_quickheap::SimdQuickHeap<u64, Avx512<true>, 16, 1>>(&graphs);

    // bench::<simd_quickheap::SimdQuickHeap<u64, 8, 1>>(&graphs);
    // bench::<simd_quickheap::SimdQuickHeap<u64, 8, 3>>(&graphs);

    eprintln!("Engineered");
    #[cfg(feature = "ffi")]
    bench::<sequence_heap::SequenceHeapU64>(&graphs);
    #[cfg(feature = "ffi")]
    bench::<s3q::S3qHeapU64>(&graphs);

    eprintln!("BASELINE");
    bench::<impls::BinaryHeap<u64>>(&graphs);

    eprintln!("DARY");
    // bench::<impls::DaryHeap<u64, 2>>(&graphs);
    // bench::<impls::DaryHeap<u64, 4>>(&graphs);
    // bench::<impls::DaryHeap<u64, 8>>(&graphs);
    // bench::<impls::DaryHeap<u64, 16>>(&graphs);
    // bench::<impls::OrxDaryHeap<u64, 2>>(&graphs);
    bench::<impls::OrxDaryHeap<u64, 4>>(&graphs);
    bench::<impls::OrxDaryHeap<u64, 8>>(&graphs);
    // bench::<impls::OrxDaryHeap<u64, 16>>(&graphs);

    // eprintln!("Amortized");
    // bench::<impls::PairingHeap<u64>>(&graphs);
    // bench::<impls::WeakHeap<u64>>(&graphs);

    eprintln!("Monotone");
    bench::<impls::RadixHeap<u64>>(&graphs);

    // eprintln!("Set");
    // bench::<impls::BTreeSet<u64>>(&graphs);
    // bench::<impls::RevBTreeSet<u64>>(&graphs);
    // bench::<impls::IndexSetBTreeSet<u64>>(&graphs);
    // bench::<impls::IndexSetRevBTreeSet<u64>>(&graphs);

    CSV_WRITER.with(|w| w.borrow_mut().flush().unwrap());
}
