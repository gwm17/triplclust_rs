use super::digraph::DiGraph;
use super::error::StitchError;

use faer;
use numpy::ndarray::{Array1, ArrayView1, ArrayView2};
use rustc_hash::FxHashSet;

pub fn stitch_pointcloud(
    point_cloud: &ArrayView2<f64>,
    labels: &ArrayView1<i32>,
    unique_labels: &ArrayView1<i32>,
    min_depth: usize,
) -> Result<(Array1<i32>, Array1<i32>), StitchError> {
    let mut max_id = match labels.iter().max() {
        Some(id) => *id,
        None => return Err(StitchError::NoInitialClusters),
    };
    let mut new_labels = labels.to_owned();
    for cluster in unique_labels.iter() {
        let mut cluster_points = vec![];
        for (idx, label) in labels.iter().enumerate() {
            if label == cluster {
                cluster_points.push(idx);
            }
        }

        let updated_labels =
            match stitch_cluster(point_cloud, &cluster_points, *cluster, max_id, min_depth) {
                Some(up) => up,
                None => continue,
            };
        let updated_max = *updated_labels
            .iter()
            .max()
            .expect("Somehow no labels after stitch");
        for (idx, updated) in updated_labels.iter().enumerate() {
            new_labels[cluster_points[idx]] = *updated;
        }
        max_id = updated_max;
    }

    let new_unique = new_labels
        .iter()
        .cloned()
        .collect::<FxHashSet<i32>>()
        .into_iter()
        .collect();

    Ok((new_labels, new_unique))
}

fn stitch_cluster(
    point_cloud: &ArrayView2<f64>,
    cluster_points: &[usize],
    current_id: i32,
    mut max_id: i32,
    min_depth: usize,
) -> Option<Vec<i32>> {
    let mut graph = DiGraph::new(point_cloud, cluster_points);
    let sub_clusters = match graph.split_into_subtrees(min_depth) {
        Some(trees) => trees,
        None => return None,
    };
    let sub_clusters = match correct_over_segmentation(point_cloud, cluster_points, sub_clusters) {
        Some(clust) => clust,
        None => return None,
    };
    let sub_clusters = match expand_start(point_cloud, cluster_points, sub_clusters) {
        Some(clust) => clust,
        None => return None,
    };
    if sub_clusters.len() <= 1 {
        return None;
    }

    let mut new_labels = vec![current_id; cluster_points.len()];
    for cluster in sub_clusters.into_iter() {
        max_id += 1;
        for idx in cluster.into_iter() {
            new_labels[idx] = max_id;
        }
    }

    Some(new_labels)
}

fn correct_over_segmentation(
    point_cloud: &ArrayView2<f64>,
    cluster_points: &[usize],
    mut sub_clusters: Vec<Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    if sub_clusters.len() <= 1 {
        return None;
    }

    let mut c_count = 0;
    let tolerance = 10;
    let mut cluster_to_join: Option<(usize, usize)> = None;
    let mut min_cluster_dist: f64;
    while sub_clusters.len() > 1 && c_count < sub_clusters.len() {
        c_count = 0;
        for (orig_idx, cluster) in sub_clusters.iter().enumerate() {
            cluster_to_join = None;
            min_cluster_dist = f64::INFINITY;
            let start_idx = cluster.len() * 3 / 4;
            let sub_len = cluster.len() - start_idx;
            let cloud = faer::Mat::from_fn(sub_len, 3, |i, j| {
                point_cloud[(cluster_points[i + start_idx], j)]
            });

            let (a, b) = match pca_ols(cloud.as_ref()) {
                Ok(values) => values,
                Err(e) => {
                    println!("Failed PCA analysis: {}", e);
                    return None;
                }
            };

            // Original was different here... did mean of same value multiplied by 20?
            let max_distance = ols_distance(a.as_ref(), b.as_ref(), cloud.row(0));
            for (comp_idx, comp_cluster) in sub_clusters.iter().enumerate() {
                if comp_idx == orig_idx
                    || (cluster.last().expect("cluster has no points?") + tolerance)
                        < *comp_cluster.first().expect("cluster has no points??")
                {
                    continue;
                }
                let leading_point = faer::Row::from_fn(3, |i| point_cloud[(comp_cluster[0], i)]);
                let dist = ols_distance(a.as_ref(), b.as_ref(), leading_point.as_ref());
                if dist < max_distance && dist < min_cluster_dist {
                    cluster_to_join = Some((orig_idx, comp_idx));
                    min_cluster_dist = dist;
                }
            }
            if cluster_to_join.is_none() {
                c_count += 1;
            } else {
                break;
            }
        }

        match &cluster_to_join {
            Some((origin, join)) => {
                let mut joiner = sub_clusters[*join].clone();
                sub_clusters[*origin].append(&mut joiner);
                sub_clusters[*origin].sort();
                // As part of the original impl, the first cluster is unique
                // So we maintain that if it is absorbed in another cluster,
                // that cluster becomes the first in the set...
                // What first really means here is unclear though. It may make more
                // sense to sort on indicies...
                if *join == 0 {
                    sub_clusters.rotate_left(*join);
                }
                sub_clusters.remove(*join);
            }
            None => (),
        }
        cluster_to_join = None;
    }

    Some(sub_clusters)
}

fn expand_start(
    point_cloud: &ArrayView2<f64>,
    cluster_points: &[usize],
    sub_clusters: Vec<Vec<usize>>,
) -> Option<Vec<Vec<usize>>> {
    if sub_clusters.len() <= 1 {
        return None;
    }
    todo!();
}

fn pca_ols(data: faer::MatRef<f64>) -> Result<(faer::Col<f64>, faer::Col<f64>), StitchError> {
    // Calculate column-wise mean
    let mean_point: faer::Col<f64> = data
        .col_iter()
        .map(|c| c.sum() / (c.nrows() as f64))
        .collect();
    // Center the data
    let mut cdata = data.clone().to_owned();
    cdata
        .col_iter_mut()
        .zip(mean_point.iter())
        .for_each(|(col, &mean)| col.iter_mut().for_each(|value| *value -= mean));
    // Singular-value decomposition
    let decomp = cdata.svd()?;
    // Get principal axis (largest singular value, I think 0th column?)
    let max_component = decomp.V().col(0).to_owned();
    return Ok((mean_point, max_component));
}

fn ols_distance(a: faer::ColRef<f64>, b: faer::ColRef<f64>, point: faer::RowRef<f64>) -> f64 {
    let lambda = b.transpose() * (a - point.transpose());
    (point - (a + lambda * b).transpose()).norm_l2()
}
