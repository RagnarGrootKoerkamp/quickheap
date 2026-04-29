use crate::Heap;
use crate::graph::{Edge, Graph};
use crate::graph_util::{pack_id_key_tuple_to_u64, unpack_id_key_tuple_from_u64};

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

    pub fn run_all(&mut self, s: usize) {
        self.heap = HeapT::default();
        self.distances[s] = 0;

        self.heap.push(pack_id_key_tuple_to_u64(s, 0));

        while let Some(next_elem) = self.heap.pop() {
            let (v, dist_to_v) = unpack_id_key_tuple_from_u64(next_elem);

            if dist_to_v == self.distances[v] {
                for (_id, Edge { to, weight, .. }) in self.graph.outgoing_edges(v) {
                    let to_dist = dist_to_v + weight;
                    if to_dist < self.distances[to] {
                        self.distances[to] = to_dist;
                        self.heap.push(pack_id_key_tuple_to_u64(to, to_dist));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{binary_heap::CustomBinaryHeap, graph::Graph};

    #[test]
    fn test_dijkstra() {
        let fo = vec![0, 2, 3, 4, 7, 10, 13];
        let heads = vec![1, 3, 0, 5, 1, 2, 4, 2, 3, 5, 0, 2, 4];
        let tails = vec![0, 0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5];
        let weights = vec![2, 3, 3, 6, 6, 9, 10, 1, 11, 6, 12, 3, 2];

        let graph = Graph::new(fo, heads, tails, weights);
        let mut query = DijkstraQuery::<CustomBinaryHeap<u64>>::new(&graph);

        query.run_all(2);

        assert_eq!(query.distances[2], 0);
        assert_eq!(query.distances[5], 6);
        assert_eq!(query.distances[0], 18);
        assert_eq!(query.distances[3], 19);
    }
}
