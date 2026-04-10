use std::fmt::Debug;

use itertools::izip;

use crate::graph_util::construct_graph_from_dimacs_file;

#[derive(Clone, Copy, Debug)]
pub struct Edge<WeightT: Debug + Copy> {
    pub from: usize,
    pub to: usize,
    pub weight: WeightT,
}

pub struct Graph<WeightT: Copy + Debug> {
    num_vertices: usize,
    num_edges: usize,
    first_out: Vec<usize>,
    edges: Vec<Edge<WeightT>>,
}

impl Graph<u32> {
    pub fn from_dimacs_instance(path: &str) -> Self {
        construct_graph_from_dimacs_file(path)
    }
}

impl<WeightT: Copy + Debug> Graph<WeightT> {
    pub fn new(
        first_out: Vec<usize>,
        to: Vec<usize>,
        from: Vec<usize>,
        weights: Vec<WeightT>,
    ) -> Self {
        Self {
            num_vertices: first_out.len() - 1,
            num_edges: to.len(),
            first_out,
            edges: izip!(from, to, weights)
                .map(|(from, to, weight)| Edge { from, to, weight })
                .collect(),
        }
    }

    pub fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    pub fn weight(&self, e: usize) -> WeightT {
        self.edges[e].weight
    }

    pub fn degree(&self, v: usize) -> usize {
        debug_assert!(v < self.num_vertices);
        self.first_out[v + 1] - self.first_out[v]
    }

    pub fn first_edge(&self, v: usize) -> usize {
        debug_assert!(v < self.num_vertices);
        self.first_out[v]
    }

    pub fn to(&self, e: usize) -> usize {
        debug_assert!(e < self.num_edges);
        self.edges[e].to
    }

    pub fn from(&self, e: usize) -> usize {
        debug_assert!(e < self.num_edges);
        self.edges[e].from
    }

    pub fn edge(&self, e: usize) -> Edge<WeightT> {
        debug_assert!(e < self.num_edges);
        self.edges[e]
    }

    pub fn print(&self) {
        let mut curr_edge = 0;
        for v in 0..self.num_vertices {
            let deg = self.degree(v);
            for _ in 0..deg {
                let head = self.to(curr_edge);
                let weight = self.weight(curr_edge);
                curr_edge += 1;
                println!("{} -> {} w = {:?}", v, head, weight);
            }
        }
    }

    pub fn outgoing_edges(&self, v: usize) -> impl Iterator<Item = (usize, Edge<WeightT>)> {
        let first_edge = self.first_out[v];
        let last_edge = self.first_out[v + 1];
        (first_edge..last_edge).map(move |id| (id, self.edges[id]))
    }
}
