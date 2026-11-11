use std::hint::black_box;

use clap::builder::styling::AnsiColor::BrightBlack;

use crate::dijkstra::{DijkstraQuery, KvDijkstraQuery};
use crate::graph::Graph;
use crate::prim::PrimMST;
use crate::{Heap, KvHeap};

pub trait GraphWorkload {
    fn setup<H: Heap<u64>>(graph: &Graph<u32>) -> impl FnOnce() -> (usize, usize);
}

pub trait KvGraphWorkload {
    fn setup_kv<H: KvHeap<u64>>(graph: &Graph<u64>) -> impl FnOnce() -> (H, usize, usize);
}

// One-to-all dijkstra query from a random starting vertex
pub struct DijkstraWorkload;

// Construct mst using the prim algorithm from a random starting vertex
pub struct PrimWorkload;

impl GraphWorkload for DijkstraWorkload {
    fn setup<H: Heap<u64>>(graph: &Graph<u32>) -> impl FnOnce() -> (usize, usize) {
        let mut dijkstra = DijkstraQuery::<H>::new(graph);
        move || dijkstra.run_all(0)
    }
}

impl KvGraphWorkload for DijkstraWorkload {
    fn setup_kv<H: KvHeap<u64>>(graph: &Graph<u64>) -> impl FnOnce() -> (H, usize, usize) {
        let mut dijkstra = KvDijkstraQuery::<H>::new(graph);
        move || {
            let (push, pop) = dijkstra.run_all(0);
            (black_box(dijkstra.get_heap()), push, pop)
        }
    }
}

impl GraphWorkload for PrimWorkload {
    fn setup<H: Heap<u64>>(graph: &Graph<u32>) -> impl FnOnce() -> (usize, usize) {
        let mut prim = PrimMST::<H>::new(graph);
        move || prim.compute_mst_from_vertex(0)
    }
}
