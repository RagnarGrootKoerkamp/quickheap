use std::fmt::Debug;

pub trait Graph<WeightT> {
    fn degree(&self, v: usize) -> usize;
    fn head(&self, e: usize) -> usize;
    fn tail(&self, e: usize) -> usize;
    fn weight(&self, e: usize) -> WeightT;

    fn first_edge(&self, v: usize) -> usize;

    fn num_vertices(&self) -> usize;
    fn num_edges(&self) -> usize;

    fn print(&self);
}

#[derive(Clone, Copy, Debug)]
pub struct Edge<WeightT: Ord + Debug + Copy> {
    pub from: usize,
    pub to: usize,
    pub weight: WeightT,
}

pub struct StaticGraph<WeightT: Clone + Debug> {
    num_vertices: usize,
    num_edges: usize,
    first_out: Vec<usize>,
    heads: Vec<usize>,
    tails: Vec<usize>,
    weights: Vec<WeightT>,
}

impl<WeightT: Clone + Debug> StaticGraph<WeightT> {
    pub fn new(
        first_out_: Vec<usize>,
        heads_: Vec<usize>,
        tails_: Vec<usize>,
        weights_: Vec<WeightT>,
    ) -> Self {
        Self {
            num_vertices: first_out_.len() - 1,
            num_edges: heads_.len(),
            weights: weights_,
            first_out: first_out_,
            heads: heads_,
            tails: tails_,
        }
    }
}

impl<WeightT: Clone + Debug> Graph<WeightT> for StaticGraph<WeightT> {
    fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    fn num_edges(&self) -> usize {
        self.num_edges
    }

    fn weight(&self, e: usize) -> WeightT {
        self.weights[e].clone()
    }

    fn degree(&self, v: usize) -> usize {
        debug_assert!(v < self.num_vertices);
        self.first_out[v + 1] - self.first_out[v]
    }

    fn first_edge(&self, v: usize) -> usize {
        debug_assert!(v < self.num_vertices);
        self.first_out[v]
    }

    fn head(&self, e: usize) -> usize {
        debug_assert!(e < self.num_edges);
        self.heads[e]
    }

    fn tail(&self, e: usize) -> usize {
        debug_assert!(e < self.num_edges);
        self.tails[e]
    }

    fn print(&self) {
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
