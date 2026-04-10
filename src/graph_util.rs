use std::{
    fmt::Debug,
    fs::File,
    io::{BufRead, BufReader},
};

use min_max_traits::Max;

use crate::graph::{Edge, Graph};

pub fn convert_directed_graph_to_undirected<WeightT: Max + Ord + Copy + Debug>(
    graph: &Graph<WeightT>,
) -> Graph<WeightT> {
    let mut edges: Vec<Edge<WeightT>> = vec![];

    for v in 0..graph.num_vertices() {
        for (_id, Edge { from, to, weight }) in graph.outgoing_edges(v) {
            // Insert the edge in both directions
            edges.push(Edge { from, to, weight });
            edges.push(Edge {
                from: to,
                to: from,
                weight,
            });
        }
    }

    edges.sort_by_key(|e| e.weight);
    edges.sort_by_key(|e| e.to);
    edges.sort_by_key(|e| e.from);

    let mut last_edge = Edge::<WeightT> {
        from: usize::MAX,
        to: usize::MAX,
        weight: WeightT::MAX,
    };
    let mut deduped_edges: Vec<Edge<WeightT>> = vec![];
    for edge in &edges {
        if edge.from == last_edge.from && edge.to == last_edge.to {
            continue;
        }

        deduped_edges.push(*edge);
        last_edge = *edge;
    }

    construct_graph_from_edge_list(&deduped_edges, graph.num_vertices())
}

pub fn construct_graph_from_edge_list<WeightT: Max + Ord + Clone + Copy + Debug>(
    edges: &Vec<Edge<WeightT>>,
    num_vertices: usize,
) -> Graph<WeightT> {
    let mut new_first_out = vec![];
    let mut new_heads = vec![];
    let mut new_tails = vec![];
    let mut weights = vec![];

    new_first_out.push(0);
    let mut current_edge = 0;
    let mut current_tail: usize = 0;
    for edge in edges {
        let new_head = edge.to;
        let new_tail = edge.from;

        while current_tail < new_tail {
            new_first_out.push(current_edge);
            current_tail += 1;
        }

        new_heads.push(new_head);
        new_tails.push(new_tail);
        weights.push(edge.weight);
        current_edge += 1;
    }

    while current_tail < num_vertices {
        new_first_out.push(current_edge);
        current_tail += 1;
    }

    Graph::new(new_first_out, new_heads, new_tails, weights)
}

pub fn construct_graph_from_dimacs_file(path: &str) -> Graph<u32> {
    let error_msg = &format!("File with path {} could not be opened.", path);
    let file = File::open(path).expect(error_msg);
    let file_reader = BufReader::new(file);

    let mut edges: Vec<Edge<u32>> = vec![];
    let mut num_vertices: usize = 0;
    let mut problem_defined: bool = false;

    for line in file_reader.lines() {
        match line {
            Ok(str) => {
                let parts: Vec<&str> = str.split(" ").collect();

                match parts[0] {
                    "c" => {
                        continue;
                    }
                    "p" => {
                        assert!(parts.len() == 4);
                        assert!(parts[1] == "sp");
                        let error_v = format!(
                            "Someting went wrong parsing the number of vertices {}",
                            parts[2]
                        );
                        let error_e = format!(
                            "Someting went wrong parsing the number of edges {}",
                            parts[3]
                        );

                        num_vertices = parts[2].parse::<usize>().expect(&error_v);
                        parts[3].parse::<usize>().expect(&error_e);
                        problem_defined = true;
                    }
                    "a" => {
                        assert!(problem_defined);
                        assert!(parts.len() == 4);

                        let error_from =
                            format!("Someting went wrong parsing the tail {}", parts[1]);

                        let error_to = format!("Someting went wrong parsing the head {}", parts[2]);
                        let error_weight =
                            format!("Someting went wrong parsing the weight {}", parts[3]);

                        let from = parts[2].parse::<usize>().expect(&error_from);
                        let to = parts[3].parse::<usize>().expect(&error_to);
                        let weight = parts[3].parse::<u32>().expect(&error_weight);

                        edges.push(Edge { from, to, weight });
                    }
                    _ => println!("Unknown match arm: {}", parts[0]),
                }
            }
            Err(e) => eprintln!("Someting went wrong when reading the file. {e}"),
        }
    }

    edges.sort_by_key(|a| (a.from, a.to, a.weight));
    construct_graph_from_edge_list(&edges, num_vertices)
}

#[inline(always)]
pub fn pack_id_key_tuple_to_u64(id: usize, key: u32) -> u64 {
    assert!(id < (1 << 32));
    (key as u64) << 32 | id as u64
}

#[inline(always)]
pub fn unpack_id_key_tuple_from_u64(tup: u64) -> (usize, u32) {
    let id = tup as u32 as usize;
    let key = (tup >> 32) as u32;
    (id, key)
}
