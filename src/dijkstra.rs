use crate::Heap;
use crate::graph::Graph;
use crate::graph_util::{pack_id_key_tuple_to_u64, unpack_id_key_tuple_from_u64};

use min_max_traits::Max;

pub struct FastResetDistanceLabel<DistT> {
    distances: Vec<DistT>,
    epoch: usize,
    epochs: Vec<usize>,
}

impl<DistT: Max + Clone> DistanceLabel<DistT> for FastResetDistanceLabel<DistT> {
    fn new(cap: usize) -> Self {
        Self {
            distances: vec![DistT::MAX; cap],
            epoch: 0,
            epochs: vec![0; cap],
        }
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.epoch += 1;
    }

    #[inline(always)]
    fn get(&self, id: usize) -> DistT {
        if self.epochs[id] == self.epoch {
            return self.distances[id].clone();
        }

        DistT::MAX
    }

    #[inline(always)]
    fn set(&mut self, id: usize, dist: DistT) {
        self.distances[id] = dist;
        self.epochs[id] = self.epoch;
    }
}

pub trait DistanceLabel<DistT> {
    fn new(cap: usize) -> Self;
    fn clear(&mut self);
    fn get(&self, id: usize) -> DistT;
    fn set(&mut self, id: usize, dist: DistT);
}

pub struct DijkstraQuery<'g, HeapT: Heap<u64>, DistanceT: DistanceLabel<u32>, GraphT: Graph<u32>> {
    heap: HeapT,
    distances: DistanceT,
    graph: &'g GraphT,
    best_distances: Vec<u32>,
}

impl<'g, HeapT: Heap<u64>, DistanceT: DistanceLabel<u32>, GraphT: Graph<u32>>
    DijkstraQuery<'g, HeapT, DistanceT, GraphT>
{
    pub fn new(graph_: &'g GraphT) -> Self {
        Self {
            heap: HeapT::default(),
            distances: DistanceT::new(graph_.num_vertices()),
            graph: graph_,
            best_distances: vec![u32::MAX; graph_.num_vertices()],
        }
    }

    pub fn get_distance(&self, v: usize) -> u32 {
        self.distances.get(v)
    }

    // fn run(&mut self, s: usize, t: usize) {}

    pub fn run_all(&mut self, s: u32) {
        self.distances.clear();
        self.heap = HeapT::default();
        self.best_distances = vec![u32::MAX; self.graph.num_vertices()];

        self.distances.set(s as usize, 0);

        let packed = pack_id_key_tuple_to_u64(s, 0);
        self.heap.push(packed);

        let mut next_elem: Option<u64> = self.heap.pop();
        while next_elem.is_some() {
            let tup = next_elem.unwrap();
            let (v, dist) = unpack_id_key_tuple_from_u64(tup);

            if self.best_distances[v as usize] <= dist {
                next_elem = self.heap.pop();
                continue;
            }

            self.best_distances[v as usize] = dist;
            self.relax_edges(v);

            next_elem = self.heap.pop();
        }
    }

    fn relax_edges(&mut self, v: u32) {
        let deg = self.graph.degree(v as usize);
        let dist_to_v = self.distances.get(v as usize);
        let first_edge = self.graph.first_edge(v as usize);

        for curr_edge in first_edge..first_edge + deg {
            let head = self.graph.head(curr_edge) as u32;
            let weight = self.graph.weight(curr_edge);

            if self.distances.get(head as usize) <= dist_to_v + weight {
                continue;
            }

            self.distances.set(head as usize, dist_to_v + weight);

            let tup = pack_id_key_tuple_to_u64(head, dist_to_v + weight);
            self.heap.push(tup);
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        binary_heap::CustomBinaryHeap,
        dijkstra::{DijkstraQuery, FastResetDistanceLabel},
        graph::StaticGraph,
    };

    #[test]
    fn test_dijkstra() {
        let fo = vec![0, 2, 3, 4, 7, 10, 13];
        let heads = vec![1, 3, 0, 5, 1, 2, 4, 2, 3, 5, 0, 2, 4];
        let tails = vec![0, 0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5];
        let weights = vec![2, 3, 3, 6, 6, 9, 10, 1, 11, 6, 12, 3, 2];

        let graph = StaticGraph::new(fo, heads, tails, weights);
        let mut query = DijkstraQuery::<
            CustomBinaryHeap<u64>,
            FastResetDistanceLabel<u32>,
            StaticGraph<u32>,
        >::new(&graph);

        query.run_all(2);

        assert_eq!(query.get_distance(2), 0);
        assert_eq!(query.get_distance(5), 6);
        assert_eq!(query.get_distance(0), 18);
        assert_eq!(query.get_distance(3), 19);
    }
}
