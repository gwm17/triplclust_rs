use rustc_hash::FxHashSet;
use std::collections::VecDeque;
use std::num::NonZero;

use kiddo::{ImmutableKdTree, SquaredEuclidean};
use numpy::ndarray::ArrayView2;

struct Edge {
    begin: usize,
    end: usize,
    distance: f64,
}

struct Node {
    index: usize,
    row: usize,
    in_edges: FxHashSet<usize>,
    out_edges: FxHashSet<usize>,
}

pub struct DiGraph {
    edges: Vec<Edge>,
    nodes: Vec<Node>,
    roots: Vec<usize>,
}

impl DiGraph {
    pub fn new(point_cloud: &ArrayView2<f64>, graph_points: &[usize]) -> Self {
        let mut digraph = Self {
            edges: vec![],
            nodes: vec![],
            roots: vec![],
        };
        for (point_idx, row) in graph_points.iter().enumerate() {
            digraph.nodes.push(Node {
                index: point_idx,
                row: *row,
                in_edges: FxHashSet::default(),
                out_edges: FxHashSet::default(),
            });
        }
        digraph.roots.push(digraph.nodes.len() - 1);
        digraph.miniumum_spanning_arboresence(point_cloud);
        digraph
    }

    pub fn split_into_subtrees(&mut self, min_depth: usize) -> Option<Vec<Vec<usize>>> {
        if !self.split_by_depth(min_depth) {
            return None;
        }
        self.split_by_weight();

        let mut subtrees = vec![];

        for root_idx in self.roots.iter() {
            let mut nodes_to_check = VecDeque::new();
            let mut visited_nodes = FxHashSet::<usize>::default();
            visited_nodes.insert(*root_idx);
            nodes_to_check.push_front(*root_idx);
            subtrees.push(vec![]);
            let this_subtree = subtrees.last_mut().expect("Somehow no last???");
            this_subtree.push(self.nodes[*root_idx].index);
            while !nodes_to_check.is_empty() {
                if let Some(node_idx) = nodes_to_check.pop_front() {
                    for edge_idx in self.nodes[node_idx].in_edges.iter() {
                        let begin_idx = &self.edges[*edge_idx].begin;
                        if !visited_nodes.contains(begin_idx) {
                            this_subtree.push(self.nodes[*begin_idx].index);
                            visited_nodes.insert(*begin_idx);
                            nodes_to_check.push_back(*begin_idx);
                        }
                    }
                    for edge_idx in self.nodes[node_idx].out_edges.iter() {
                        let end_idx = &self.edges[*edge_idx].end;
                        if !visited_nodes.contains(end_idx) {
                            this_subtree.push(self.nodes[*end_idx].index);
                            visited_nodes.insert(*end_idx);
                            nodes_to_check.push_back(*end_idx);
                        }
                    }
                }
            }
        }

        return Some(subtrees);
    }

    fn miniumum_spanning_arboresence(&mut self, point_cloud: &ArrayView2<f64>) {
        let explicit_layout: Vec<[f64; 3]> = self
            .nodes
            .iter()
            .map(|node| {
                [
                    point_cloud[(node.row, 0)],
                    point_cloud[(node.row, 1)],
                    point_cloud[(node.row, 2)],
                ]
            })
            .collect();
        let tree = ImmutableKdTree::<f64, 3>::new_from_slice(&explicit_layout);
        let n_neigh = NonZero::<usize>::new(1).expect("Some how 1 is 0??");

        for node_idx in 0..(self.nodes.len() - 1) {
            let nearest = tree.nearest_n::<SquaredEuclidean>(&explicit_layout[node_idx], n_neigh);
            assert!(nearest.len() != 0);
            let nearest_idx = nearest[0].item as usize;
            self.edges.push(Edge {
                begin: node_idx,
                end: nearest_idx,
                distance: nearest[0].distance,
            });
            let edge_idx = self.edges.len() - 1;
            self.nodes[node_idx].out_edges.insert(edge_idx);
            self.nodes[nearest_idx].in_edges.insert(edge_idx);
        }
    }

    fn split_by_depth(&mut self, min_depth: usize) -> bool {
        let mut changed = false;
        for node_idx in 0..self.nodes.len() {
            if self.nodes[node_idx].in_edges.len() <= 1 {
                continue;
            }
            let mut edges_to_remove = FxHashSet::default();
            for edge in self.nodes[node_idx].in_edges.iter() {
                if self.depth_search(node_idx, min_depth) {
                    edges_to_remove.insert(*edge);
                }
            }

            if edges_to_remove.len() > 1 {
                changed = true;
                let mut min_dist_edge = 0;
                let mut min_dist = f64::INFINITY;
                for edge in edges_to_remove.iter() {
                    if self.edges[*edge].distance < min_dist {
                        min_dist_edge = *edge;
                        min_dist = self.edges[*edge].distance;
                    }
                }
                edges_to_remove.remove(&min_dist_edge);

                for edge in edges_to_remove.into_iter() {
                    self.nodes[node_idx].in_edges.remove(&edge);
                    self.nodes[self.edges[edge].begin].out_edges.remove(&edge);
                    self.roots.push(self.edges[edge].begin);
                }
            }
        }

        changed
    }

    fn split_by_weight(&mut self) {
        for node_idx in 0..self.nodes.len() {
            let (n_nodes, total_distance) = self.total_distance_to_depth4(node_idx);
            let edges_to_scan = self.nodes[node_idx].in_edges.clone();
            for edge_idx in edges_to_scan {
                let exlusive_mean =
                    (total_distance - self.edges[edge_idx].distance) / ((n_nodes - 1) as f64);
                if self.edges[edge_idx].distance > (4.5 * exlusive_mean) {
                    self.roots.push(self.edges[edge_idx].begin);
                    self.nodes[self.edges[edge_idx].begin]
                        .out_edges
                        .remove(&edge_idx);
                    self.nodes[node_idx].in_edges.remove(&edge_idx);
                }
            }
        }
    }

    fn depth_search(&self, node: usize, depth: usize) -> bool {
        if depth > 0 {
            for edge in self.nodes[node].in_edges.iter() {
                if self.depth_search(self.edges[*edge].begin, depth - 1) {
                    return true;
                }
            }
            return false;
        } else {
            return false;
        }
    }

    fn total_distance_to_depth4(&self, node: usize) -> (usize, f64) {
        let mut n_nodes = 0;
        let mut total_distance = 0.0;
        let mut nodes_to_check = VecDeque::new();
        let mut visited_nodes = FxHashSet::<usize>::default();
        let mut depth = 0;
        nodes_to_check.push_front(Some(node));
        nodes_to_check.push_back(None);
        while depth < 4 {
            match nodes_to_check.pop_front() {
                Some(node) => match node {
                    Some(idx) => {
                        visited_nodes.insert(idx);
                        let node_data = &self.nodes[idx];
                        for edge_idx in node_data.in_edges.iter() {
                            if !visited_nodes.contains(&self.edges[*edge_idx].begin) {
                                nodes_to_check.push_back(Some(self.edges[*edge_idx].begin));
                                n_nodes += 1;
                                total_distance += self.edges[*edge_idx].distance;
                            }
                        }
                        for edge_idx in node_data.out_edges.iter() {
                            if !visited_nodes.contains(&self.edges[*edge_idx].end) {
                                nodes_to_check.push_back(Some(self.edges[*edge_idx].end));
                                n_nodes += 1;
                                total_distance += self.edges[*edge_idx].distance;
                            }
                        }
                    }
                    None => {
                        depth += 1;
                        nodes_to_check.push_back(None);
                    }
                },
                None => {
                    break;
                }
            }
        }
        return (n_nodes, total_distance);
    }
}
