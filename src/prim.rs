use crate::graph::{Edge, Graph};
use crate::graph_util::{pack_id_key_tuple_to_u64, unpack_id_key_tuple_from_u64};
use crate::Heap;

pub struct PrimMST<'g, HeapT: Heap<u64>> {
    graph: &'g Graph<u32>,
    heap: HeapT,
    visited: Vec<bool>,
    mst_edges: Vec<Edge<u32>>,
}

impl<'g, HeapT: Heap<u64>> PrimMST<'g, HeapT> {
    pub fn new(graph: &'g Graph<u32>) -> Self {
        Self {
            heap: HeapT::default(),
            visited: vec![false; graph.num_vertices()],
            mst_edges: vec![],
            graph,
        }
    }

    pub fn compute_mst_from_vertex(&mut self, v: usize) {
        self.relax(v);
        self.visited[v] = true;

        while let Some(tup) = self.heap.pop() {
            let (id, _weight) = unpack_id_key_tuple_from_u64(tup);

            let edge @ Edge { to: v, .. } = self.graph.edge(id);

            if !self.visited[v] {
                self.mst_edges.push(edge);
                self.visited[v] = true;

                self.relax(v);
            }
        }
    }

    #[cfg(test)]
    fn parents(&mut self, v: usize) -> Vec<usize> {
        let mut parents = vec![usize::MAX; self.graph.num_vertices()];
        parents[v] = v;
        for edge in &self.mst_edges {
            parents[edge.to] = edge.from;
        }
        parents
    }

    fn relax(&mut self, v: usize) {
        for (id, Edge { to, weight, .. }) in self.graph.outgoing_edges(v) {
            if !self.visited[to] {
                self.heap.push(pack_id_key_tuple_to_u64(id, weight));
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        binary_heap::CustomBinaryHeap, graph::Graph,
        graph_util::convert_directed_graph_to_undirected, prim::PrimMST,
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

        let tree = prim_algo.parents(4);
        let tree2 = prim_algo2.parents(4);

        assert_eq!(4, tree[4]);
        assert_eq!(4, tree[5]);
        assert_eq!(4, tree[2]);
        assert_eq!(2, tree[3]);
        assert_eq!(3, tree[0]);
        assert_eq!(0, tree[1]);

        assert_eq!(4, tree2[4]);
        assert_eq!(4, tree2[5]);
        assert_eq!(4, tree2[2]);
        assert_eq!(2, tree2[3]);
        assert_eq!(3, tree2[0]);
        assert_eq!(0, tree2[1]);
    }
}
