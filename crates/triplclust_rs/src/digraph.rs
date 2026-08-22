/// This is a nearly identical implementation of the C++ triplclust digraph system
/// for splitting clusters. This includes some of the quirks associated with that
/// implementation.
///
/// TODO: Take another pass at this
use rustc_hash::FxHashSet;
use std::collections::VecDeque;
use std::num::NonZero;

use kiddo::{ImmutableKdTree, SquaredEuclidean};
use numpy::ndarray::ArrayView2;

/// Definition of a directed, weighted edge
#[derive(Debug)]
struct Edge {
    /// Node the edge starts at
    begin: usize,
    /// Node the edge ends at
    end: usize,
    /// Distance between nodes (points); aka the weight
    distance: f64,
}

/// A node in our graph, aka a point in the cluster
#[derive(Debug)]
struct Node {
    /// Index (i.e. id) of this node
    index: usize,
    /// Index of this node in the original point cloud
    row: usize,
    /// Edges flowing into this node
    in_edges: FxHashSet<usize>,
    /// Edges flowing out of this node
    out_edges: FxHashSet<usize>,
}

/// Our directed graph implementation
/// Graphs are a little different in Rust
/// since we can't use mutable shared references
pub struct DiGraph {
    /// The edges, which we id/access by index
    edges: Vec<Edge>,
    /// The nodes, which we id/access by index
    nodes: Vec<Node>,
    /// Roots of sub-graphs identified by our algorithms
    roots: Vec<usize>,
}

impl DiGraph {
    /// Create the graph and populate all of the nodes and edges.
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

    /// Ask the graph to split the cluster into sub-trees.
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

    /// In original code this was called minimum_spanning_tree. Since this is a digraph,
    /// it's an miniumum_spanning_arboresence, as there is no one single tree. But...
    /// ours isn't a true digraph, as all edges are really bi-directional...
    ///
    /// Either way, the edges are the nearest neighbor which is not at an earlier index
    /// in the cluster.
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
        let n_neigh = NonZero::<usize>::new(2).expect("Some how 1 is 0??");
        for node_idx in 0..(self.nodes.len() - 1) {
            let tree = ImmutableKdTree::<f64, 3>::new_from_slice(&explicit_layout[node_idx..]);
            let nearest = tree.nearest_n::<SquaredEuclidean>(&explicit_layout[node_idx], n_neigh);
            assert!(nearest.len() > 1);
            let nearest_idx = node_idx + nearest[1].item as usize;
            self.edges.push(Edge {
                begin: node_idx,
                end: nearest_idx,
                distance: nearest[1].distance.sqrt(),
            });
            let edge_idx = self.edges.len() - 1;
            self.nodes[node_idx].out_edges.insert(edge_idx);
            self.nodes[nearest_idx].in_edges.insert(edge_idx);
        }
    }

    /// Split the graph into sub-trees by depth. Given a minimum depth,
    /// do a depth first search for each node and if an edge leads to a path  which
    /// exceeds min_depth, the node is a sub-tree root and the edge is removed from the
    /// graph.
    fn split_by_depth(&mut self, min_depth: usize) -> bool {
        let mut changed = false;
        for node_idx in 0..self.nodes.len() {
            if self.nodes[node_idx].in_edges.len() <= 1 {
                continue;
            }
            let mut edges_to_remove = FxHashSet::default();
            for edge in self.nodes[node_idx].in_edges.iter() {
                if self.depth_search(self.edges[*edge].begin, min_depth) {
                    edges_to_remove.insert(*edge);
                }
            }
            if edges_to_remove.len() > 1 {
                changed = true;
                let mut min_dist_edge = self.edges.len() + 1;
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

    /// Split the graph into sub-trees by weight. For each node, peform a breadth-first
    /// search up to depth 4, and calculate the total weight of all edges to that depth.
    /// Then see if any edge exceeds the non-inclusive average weight of that total. If
    /// it does, the node is a root of a sub-tree, and the edge is removed from the graph.
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

    /// A depth first search
    fn depth_search(&self, node: usize, depth: usize) -> bool {
        if depth > 0 {
            for edge in self.nodes[node].in_edges.iter() {
                if self.depth_search(self.edges[*edge].begin, depth - 1) {
                    return true;
                }
            }
            return false;
        } else {
            return true;
        }
    }

    /// Get the total weight of all edges to depth 4.
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
