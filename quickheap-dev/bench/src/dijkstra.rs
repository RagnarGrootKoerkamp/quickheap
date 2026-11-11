use quickheap::ElemT;
use rand_distr::num_traits::ToPrimitive;

use crate::graph::{Edge, Graph};
use crate::graph_util::{pack_id_key_tuple_to_u64, unpack_id_key_tuple_from_u64};
use crate::workloads::Elem;
use crate::{Heap, KvHeap};

pub struct DijkstraQuery<'g, HeapT: Heap<u64>> {
    heap: HeapT,
    graph: &'g Graph<u32>,
    distances: Vec<u32>,
}

impl<'g, HeapT: Heap<u64>> DijkstraQuery<'g, HeapT> {
    pub fn new(graph: &'g Graph<u32>) -> Self {
        Self {
            heap: HeapT::default(),
            distances: vec![u32::MAX; graph.num_vertices()],
            graph,
        }
    }

    /// Return (#push, #pop)
    pub fn run_all(&mut self, s: usize) -> (usize, usize) {
        let mut push_count = 0;
        let mut pop_count = 0;
        self.heap = HeapT::default();
        self.distances[s] = 0;

        push_count += 1;
        self.heap.push(pack_id_key_tuple_to_u64(s, 0));

        while let Some(next_elem) = self.heap.pop() {
            pop_count += 1;
            let (v, dist_to_v) = unpack_id_key_tuple_from_u64(next_elem);

            if dist_to_v == self.distances[v] {
                for (_id, Edge { to, weight, .. }) in self.graph.outgoing_edges(v) {
                    let to_dist = dist_to_v + weight;
                    if to_dist < self.distances[to] {
                        self.distances[to] = to_dist;
                        push_count += 1;
                        self.heap.push(pack_id_key_tuple_to_u64(to, to_dist));
                    }
                }
            }
        }
        (push_count, pop_count)
    }
}

pub struct KvDijkstraQuery<'g, HeapT: KvHeap<u64>> {
    heap: HeapT,
    graph: &'g Graph<u64>,
    distances: Vec<u64>,
}

impl<'g, HeapT: KvHeap<u64>> KvDijkstraQuery<'g, HeapT> {
    pub fn new(graph: &'g Graph<u64>) -> Self {
        Self {
            heap: HeapT::default(),
            distances: vec![u64::MAX; graph.num_vertices()],
            graph,
        }
    }

    pub fn get_heap(self) -> HeapT {
        self.heap
    }

    /// Return (#push, #pop)
    pub fn run_all(&mut self, s: u64) -> (usize, usize) {
        let mut push_count = 0;
        let mut pop_count = 0;
        self.heap = HeapT::default();
        self.distances[s as usize] = 0;

        push_count += 1;
        self.heap.push(0, s);

        while let Some((dist_to_v, v)) = self.heap.pop() {
            pop_count += 1;

            for (_id, Edge { to, weight, .. }) in self
                .graph
                .outgoing_edges(v.to_usize().expect("Conversion to usize failed."))
            {
                let to_dist = dist_to_v + weight;

                if self.distances[to] == u64::MAX {
                    self.distances[to] = to_dist;
                    self.heap
                        .push(to_dist, to.to_u64().expect("Conversion to u64 failed."));
                    continue;
                }

                if to_dist < self.distances[to] {
                    let old_dist = self.distances[to];
                    self.distances[to] = to_dist;
                    push_count += 1;

                    self.heap.decrease_key(
                        to_dist,
                        old_dist,
                        to.to_u64().expect("Conversion to u64 failed."),
                    );
                }
            }
        }
        (push_count, pop_count)
    }
}

#[cfg(test)]
mod test {
    use quickheap::{Avx2, KvSimdQuickHeap, decrease_key::SimpleDK};

    use super::*;
    use crate::{binary_heap::CustomBinaryHeap, graph::Graph};

    #[test]
    fn test_dijkstra() {
        let fo = vec![0, 2, 3, 4, 7, 10, 13];
        let heads = vec![1, 3, 0, 5, 1, 2, 4, 2, 3, 5, 0, 2, 4];
        let tails = vec![0, 0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5];
        let weights1 = vec![2, 3, 3, 6, 6, 9, 10, 1, 11, 6, 12, 3, 2];
        let weights2 = vec![2, 3, 3, 6, 6, 9, 10, 1, 11, 6, 12, 3, 2];

        let graph1 = Graph::<u32>::new(fo.clone(), heads.clone(), tails.clone(), weights1);
        let graph2 = Graph::<u64>::new(fo, heads, tails, weights2);

        let mut query = DijkstraQuery::<CustomBinaryHeap<u64>>::new(&graph1);
        let mut query2 = KvDijkstraQuery::<KvSimdQuickHeap<u64, Avx2, SimpleDK<u64>>>::new(&graph2);

        query.run_all(2);
        query2.run_all(2);

        assert_eq!(query.distances[2], 0);
        assert_eq!(query.distances[5], 6);
        assert_eq!(query.distances[0], 18);
        assert_eq!(query.distances[3], 19);

        assert_eq!(query2.distances[2], 0);
        assert_eq!(query2.distances[5], 6);
        assert_eq!(query2.distances[0], 18);
        assert_eq!(query2.distances[3], 19);
    }
}
