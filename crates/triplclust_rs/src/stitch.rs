use super::digraph::DiGraph;
use numpy::ndarray::{Array1, ArrayView1, ArrayView2};

pub fn stitch_pointcloud(
    point_cloud: &ArrayView2<f64>,
    labels: &ArrayView1<i32>,
    unique_labels: &ArrayView1<i32>,
) -> (Array1<i32>, Array1<i32>) {
    todo!();
}

fn stitch_cluster(point_cloud: Vec<[f64; 3]>) -> Vec<i32> {
    todo!();
}

fn correct_over_segmentation(
    point_cloud: &[[f64; 3]],
    subtrees: Vec<Vec<usize>>,
) -> Vec<Vec<usize>> {
    todo!();
}

fn expand_start(point_cloud: &[[f64; 3]], subtrees: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    todo!();
}

fn renormalize_cluster_ids(labels: Vec<i32>) -> Vec<i32> {
    todo!();
}
