use super::error::{ClusterError, TriplclustError};
use super::params::ClusterParams;
use super::triplet::{Triplet, triplet_metric};
use kodama::{Dendrogram, Step, linkage};
use numpy::ndarray::Array1;
use rustc_hash::{FxHashMap, FxHashSet};

pub struct ClusterResult {
    pub unique_labels: Array1<i32>,
    pub labels: Array1<i32>,
    pub n_clusters_removed: usize,
    pub optimal_cdt: f64,
}

impl ClusterResult {
    pub fn new(
        cloud_size: usize,
        triplet_labels: Array1<i32>,
        triplets: &Vec<Triplet>,
        n_clusters_removed: usize,
        optimal_cdt: f64,
    ) -> Self {
        assert!(triplet_labels.len() == triplets.len());
        let mut label_map = vec![FxHashMap::default(); cloud_size];

        for (triplet, label) in triplets.iter().zip(triplet_labels.iter()) {
            Self::assign_label(&mut label_map, triplet.index_a, *label);
            Self::assign_label(&mut label_map, triplet.index_b, *label);
            Self::assign_label(&mut label_map, triplet.index_c, *label);
        }

        let labels = Self::normalize(Self::collapse_by_max(label_map));
        let unique_labels = labels
            .iter()
            .cloned()
            .collect::<FxHashSet<i32>>()
            .into_iter()
            .collect();
        Self {
            unique_labels,
            labels,
            n_clusters_removed,
            optimal_cdt,
        }
    }

    fn assign_label(map: &mut Vec<FxHashMap<i32, i32>>, point: usize, label: i32) {
        match map[point].get_mut(&label) {
            Some(count) => *count += 1,
            None => {
                map[point].insert(label, 0);
            }
        }
    }

    fn collapse_by_max(map: Vec<FxHashMap<i32, i32>>) -> Array1<i32> {
        let mut point_labels = Array1::<i32>::zeros(map.len());
        for (idx, labels) in map.iter().enumerate() {
            if labels.len() == 0 {
                point_labels[idx] = -1;
            } else if labels.len() == 1 {
                point_labels[idx] = labels.iter().fold(0, |x, y| x + y.0);
            } else {
                point_labels[idx] = *labels.iter().max_by(|x, y| x.1.cmp(y.1)).unwrap().0;
            }
        }
        point_labels
    }

    fn normalize(raw_labels: Array1<i32>) -> Array1<i32> {
        let mut output = Array1::<i32>::ones(raw_labels.len()) * -1;
        let uniques = raw_labels.iter().cloned().collect::<FxHashSet<i32>>();
        let mut uni_map = FxHashMap::<i32, i32>::default();
        uni_map.insert(-1, -1); // noise maps to noise
        let mut cluster_val = 0;
        for uni in uniques.iter() {
            if *uni == -1 {
                continue;
            } else {
                uni_map.insert(*uni, cluster_val);
                cluster_val += 1;
            }
        }

        for idx in 0..raw_labels.len() {
            output[idx] = *uni_map
                .get(&raw_labels[idx])
                .expect("Somehow a label wasn't in the uni map...");
        }

        output
    }
}

fn distance_matrix(triplets: &[Triplet], scale: &f64) -> Vec<f64> {
    let mut matrix = Vec::with_capacity(triplets.len() * (triplets.len() - 1) / 2);
    matrix.fill(0.0);
    let n = triplets.len();
    for i in 0..(n - 1) {
        for j in (i + 1)..n {
            matrix.push(triplet_metric(&triplets[i], &triplets[j], scale));
        }
    }
    matrix
}

fn step_std_dev(steps: &[Step<f64>]) -> f64 {
    let n = steps.len() as f64;
    let mean = steps
        .iter()
        .fold(0.0, |mean: f64, step: &Step<f64>| mean + step.dissimilarity)
        / n;

    return (steps.iter().fold(0.0, |sigma: f64, step: &Step<f64>| {
        sigma + (mean - step.dissimilarity).powf(2.0)
    }) / (n - 1.0))
        .sqrt();
}

fn compute_triplet_labels(
    n_triplets: usize,
    dendrogram: &Dendrogram<f64>,
    cluster_distance_threshold: &Option<f64>,
) -> (Array1<i32>, f64) {
    let steps = dendrogram.steps();
    let mut triplet_labels = Array1::<i32>::zeros(n_triplets);
    triplet_labels += -1;
    let mut stop_index = 0;
    let mut opt_cdt = 0.0;
    match cluster_distance_threshold {
        None => {
            for (idx, step) in steps[steps.len() / 2..].iter().enumerate() {
                let index = idx + steps.len() / 2;
                let prev_step = &steps[index - 1];
                if (step.dissimilarity > 1.0e-8 || prev_step.dissimilarity > 0.0)
                    && step.dissimilarity
                        > (prev_step.dissimilarity + 2.0 * step_std_dev(&steps[..index + 1]))
                {
                    stop_index = index;
                    opt_cdt = (step.dissimilarity + prev_step.dissimilarity) * 0.5;
                    break;
                }
            }
        }
        Some(cdt) => {
            for (idx, step) in steps.iter().enumerate() {
                if step.dissimilarity > *cdt {
                    stop_index = idx;
                    break;
                }
            }
            opt_cdt = *cdt;
        }
    }

    // In kodama, the Step always has the smaller label in cluster1
    // So we only have the case where cluster1 and cluster2 are < n_triplets
    // and the case when cluster1 < n_triplets.
    let cluster_base = dendrogram.observations() as i32;
    for (idx, step) in steps[..stop_index].iter().enumerate() {
        let cluster_label = idx as i32 + cluster_base;
        if step.cluster1 < n_triplets && step.cluster2 < n_triplets {
            triplet_labels[step.cluster1] = cluster_label;
            triplet_labels[step.cluster2] = cluster_label;
        } else if step.cluster1 < n_triplets {
            triplet_labels[step.cluster1] = cluster_label;
            for label in triplet_labels.iter_mut() {
                if *label == step.cluster2 as i32 {
                    *label = cluster_label;
                }
            }
        } else {
            for label in triplet_labels.iter_mut() {
                if *label == step.cluster1 as i32 || *label == step.cluster2 as i32 {
                    *label = cluster_label;
                }
            }
        }
    }

    (triplet_labels, opt_cdt)
}

fn apply_min_cluster_size(
    triplet_labels: &mut Array1<i32>,
    dendrogram: &Dendrogram<f64>,
    min_cluster_size: usize,
) -> usize {
    let uniques: FxHashSet<i32> = triplet_labels.iter().cloned().collect();
    let mut removed: usize = 0;
    for uni in uniques.iter() {
        if *uni == -1 {
            continue;
        } else if dendrogram.cluster_size(*uni as usize) < min_cluster_size {
            removed += 1;
            for label in triplet_labels.iter_mut() {
                if *label == *uni {
                    *label = -1;
                }
            }
        }
    }
    return removed;
}

pub fn cluster(
    smooth_cloud_size: usize,
    triplets: &Vec<Triplet>,
    params: &ClusterParams,
) -> Result<ClusterResult, TriplclustError> {
    if triplets.len() == 0 {
        return Err(ClusterError::EmptyTriplets.into());
    }
    let mut d_matrix = distance_matrix(&triplets, &params.scale);
    let dendrogram = linkage(&mut d_matrix, triplets.len(), params.linkage.clone().into());
    let (mut triplet_labels, opt_cdt) = compute_triplet_labels(
        triplets.len(),
        &dendrogram,
        &params.cluster_distance_threshold,
    );
    let n_removed =
        apply_min_cluster_size(&mut triplet_labels, &dendrogram, params.min_cluster_size);
    Ok(ClusterResult::new(
        smooth_cloud_size,
        triplet_labels,
        triplets,
        n_removed,
        opt_cdt,
    ))
}
