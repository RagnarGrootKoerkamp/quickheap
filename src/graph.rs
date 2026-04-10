use std::fmt::Debug;

#[derive(Clone, Copy, Debug)]
pub struct Edge<WeightT: Ord + Debug + Copy> {
    pub from: usize,
    pub to: usize,
    pub weight: WeightT,
}

pub struct Graph<WeightT: Clone + Debug> {
    num_vertices: usize,
    num_edges: usize,
    first_out: Vec<usize>,
    heads: Vec<usize>,
    tails: Vec<usize>,
    weights: Vec<WeightT>,
}

impl<WeightT: Clone + Debug> Graph<WeightT> {
    pub fn new(
        first_out: Vec<usize>,
        heads: Vec<usize>,
        tails: Vec<usize>,
        weights: Vec<WeightT>,
    ) -> Self {
        Self {
            num_vertices: first_out.len() - 1,
            num_edges: heads.len(),
            weights,
            first_out,
            heads,
            tails,
        }
    }

    pub fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    pub fn weight(&self, e: usize) -> WeightT {
        self.weights[e].clone()
    }

    pub fn degree(&self, v: usize) -> usize {
        debug_assert!(v < self.num_vertices);
        self.first_out[v + 1] - self.first_out[v]
    }

    pub fn first_edge(&self, v: usize) -> usize {
        debug_assert!(v < self.num_vertices);
        self.first_out[v]
    }

    pub fn head(&self, e: usize) -> usize {
        debug_assert!(e < self.num_edges);
        self.heads[e]
    }

    pub fn tail(&self, e: usize) -> usize {
        debug_assert!(e < self.num_edges);
        self.tails[e]
    }

    pub fn print(&self) {
        let mut curr_edge = 0;
        for v in 0..self.num_vertices {
            let deg = self.degree(v);
            for _ in 0..deg {
                let head = self.head(curr_edge);
                let weight = self.weight(curr_edge);
                curr_edge += 1;
                println!("{} -> {} w = {:?}", v, head, weight);
            }
        }
    }
}
