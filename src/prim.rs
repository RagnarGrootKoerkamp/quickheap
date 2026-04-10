use std::fmt::Debug;

use crate::graph::{Edge, Graph};
use crate::graph_util::{pack_id_key_tuple_to_u64, unpack_id_key_tuple_from_u64};
use crate::Heap;

pub struct MST<WeightT: Ord + Debug + Copy> {
    edges: Vec<Edge<WeightT>>,
    parents: Vec<usize>,
}

pub trait MSTAlgorithm<'g, WeightT: Ord + Debug + Copy> {
    fn new(graph: &'g Graph<u32>) -> Self;
    fn compute_mst_from_vertex(&mut self, v: usize);
    fn get_tree(&self) -> MST<WeightT>;
}

pub struct PrimMST<'g, HeapT: Heap<u64>> {
    heap: HeapT,
    graph: &'g Graph<u32>,
    contained: Vec<bool>,
    mst_edges: Vec<Edge<u32>>,
    parents: Vec<usize>,
}

impl<'g, HeapT: Heap<u64>> MSTAlgorithm<'g, u32> for PrimMST<'g, HeapT> {
    fn new(graph_: &'g Graph<u32>) -> Self {
        Self {
            graph: graph_,
            heap: HeapT::default(),
            contained: vec![false; graph_.num_vertices()],
            mst_edges: vec![],
            parents: vec![usize::MAX; graph_.num_vertices()],
        }
    }

    fn compute_mst_from_vertex(&mut self, v: usize) {
        self.init();

        self.relax(v);
        self.contained[v] = true;

        let mut next_elem = self.heap.pop();
        while next_elem.is_some() {
            let tup = next_elem.unwrap();
            let (id, weight_) = unpack_id_key_tuple_from_u64(tup);

            let tail = self.graph.tail(id as usize);
            let head = self.graph.head(id as usize);

            if self.contained[head] {
                next_elem = self.heap.pop();
                continue;
            }

            self.mst_edges.push(Edge {
                from: tail,
                to: head,
                weight: weight_,
            });

            self.contained[head] = true;
            self.relax(head);

            next_elem = self.heap.pop();
        }

        self.parents[v] = v;
        for edge in &self.mst_edges {
            self.parents[edge.to] = edge.from;
        }
    }

    fn get_tree(&self) -> MST<u32> {
        MST {
            edges: self.mst_edges.clone(),
            parents: self.parents.clone(),
        }
    }
}

impl<'g, HeapT: Heap<u64>> PrimMST<'g, HeapT> {
    fn init(&mut self) {
        self.heap = HeapT::default();
        self.contained = vec![false; self.graph.num_vertices()];
        self.mst_edges.clear();
        self.parents = vec![usize::MAX; self.graph.num_vertices()];
    }

    fn relax(&mut self, v: usize) {
        let deg = self.graph.degree(v);
        let first = self.graph.first_edge(v);

        for i in 0..deg {
            let head = self.graph.head(first + i);

            if self.contained[head] {
                continue;
            }

            let weight = self.graph.weight(first + i);
            let tup = pack_id_key_tuple_to_u64((first + i) as u32, weight);
            self.heap.push(tup);
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        binary_heap::CustomBinaryHeap,
        graph::Graph,
        graph_util::convert_directed_graph_to_undirected,
        prim::{MSTAlgorithm, PrimMST},
        scalar_quickheap::ScalarQuickHeap,
    };

    #[test]
    fn test_prim() {
        let fo = vec![0, 2, 3, 4, 7, 10, 13];
        let heads = vec![1, 3, 0, 5, 1, 2, 4, 2, 3, 5, 0, 2, 4];
        let tails = vec![0, 0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5];
        let weights = vec![2, 3, 3, 6, 6, 9, 10, 1, 11, 6, 12, 3, 2];

        let graph = Graph::new(fo, heads, tails, weights);

        let undirected_graph = convert_directed_graph_to_undirected(&graph);

        let mut prim_algo = PrimMST::<CustomBinaryHeap<u64>>::new(&undirected_graph);

        let mut prim_algo2 = PrimMST::<ScalarQuickHeap<u64, 3, false>>::new(&undirected_graph);

        prim_algo.compute_mst_from_vertex(4);
        prim_algo2.compute_mst_from_vertex(4);

        let tree = prim_algo.get_tree();
        let tree2 = prim_algo.get_tree();

        assert_eq!(4, tree.parents[4]);
        assert_eq!(4, tree.parents[5]);
        assert_eq!(4, tree.parents[2]);
        assert_eq!(2, tree.parents[3]);
        assert_eq!(3, tree.parents[0]);
        assert_eq!(0, tree.parents[1]);

        assert_eq!(4, tree2.parents[4]);
        assert_eq!(4, tree2.parents[5]);
        assert_eq!(4, tree2.parents[2]);
        assert_eq!(2, tree2.parents[3]);
        assert_eq!(3, tree2.parents[0]);
        assert_eq!(0, tree2.parents[1]);
    }
}
