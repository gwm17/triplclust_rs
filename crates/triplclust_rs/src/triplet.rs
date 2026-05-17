//! Triplet algorithims, including a *lot* of 3d vector math.
//!
//! ## Why did I use static arrays over numpy for the vector math?
//!
//! It is fair to question why this is done this way when we have crates like
//! numpy included in the workspace. Moving the data out of numpy dynamic, heap
//! allocated arrays to fixed size stack allocated arrays is critical for performance
//! in this algorithm. Additionally, these are (mostly) formulated to avoid unnecessary
//! allocation, which can sneak up on you when using libraries like numpy.
//!
//! This has been benchmarked against using numpy implementations and has an order of
//! magnitude performance gain.
use super::params::TripletParams;
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use numpy::ndarray::ArrayView2;

/// Implementation of a Triplet as defined by C. Dalitz et al. in the
/// TriplClust paper.
#[derive(Debug, Clone)]
pub struct Triplet {
    /// Index of point a in the point cloud
    pub index_a: usize,
    /// Index of point b in the point cloud
    pub index_b: usize,
    /// Index of point c in the point cloud
    pub index_c: usize,
    /// The centroid position of the triplet (mean of the constituent points)
    pub centroid: [f64; 3],
    /// The direction of the triplet
    pub direction: [f64; 3],
    /// The error (1 - cos(alpha) in the paper)
    pub error: f64,
}

impl PartialEq for Triplet {
    /// Triplets are defined as equal if they have the exact same
    /// points in the exact same order
    fn eq(&self, other: &Self) -> bool {
        self.index_a == other.index_a
            && self.index_b == other.index_b
            && self.index_c == other.index_c
    }
}

impl PartialOrd for Triplet {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.error.partial_cmp(&other.error)
    }
}

/// 3d vector norm
fn norm_l2(x: &[f64]) -> f64 {
    return f64::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
}

/// 3d vector dot product
fn dot(x: &[f64], y: &[f64]) -> f64 {
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

/// 3d vector subtraction
fn sub(x: &[f64], y: &[f64]) -> [f64; 3] {
    return [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
}

/// 3d vector addition
fn add(x: &[f64], y: &[f64]) -> [f64; 3] {
    return [x[0] + y[0], x[1] + y[1], x[2] + y[2]];
}

/// 3d vector division by a scalar
fn div(x: &[f64], s: f64) -> [f64; 3] {
    return [x[0] / s, x[1] / s, x[2] / s];
}

/// 3d vector multiplication by a scalar
fn mult(x: &[f64], s: f64) -> [f64; 3] {
    return [x[0] * s, x[1] * s, x[2] * s];
}

/// Evaluate the set of valid triplets for the points in a point cloud
///
/// Walk through the points in each core point's neighborhood and evaluate
/// all triplet combintation and keep only those which meet our desired error
/// constraint. Parameters also control upper limits on how many triplets may be
/// kept for each core point. In the case where the number of candidates exceeds the
/// limit, the lowest error triplets within the limit are kept.
///
/// See TriplClust paper for more details.
pub fn evaluate_triplets(cloud: &ArrayView2<f64>, params: &TripletParams) -> Vec<Triplet> {
    let explicit_layout: Vec<[f64; 3]> = cloud
        .rows()
        .into_iter()
        .map(|point| [point[0], point[1], point[2]])
        .collect();
    let tree = ImmutableKdTree::<f64, 3>::new_from_slice(&explicit_layout);
    let mut triplets = vec![];

    for (index_b, point_b) in explicit_layout.iter().enumerate() {
        let neighbors =
            tree.nearest_n::<SquaredEuclidean>(&explicit_layout[index_b], params.neighborhood_size);
        let mut candidates = vec![];
        for nindex_a in 1..neighbors.len() {
            if neighbors[nindex_a].distance == 0.0 {
                continue;
            }
            let point_a = explicit_layout[neighbors[nindex_a].item as usize];
            let mut dir_ab = sub(point_b, &point_a);
            dir_ab = div(&dir_ab, norm_l2(&dir_ab));

            for nindex_c in (nindex_a + 1)..neighbors.len() {
                if neighbors[nindex_c].distance == 0.0 {
                    continue;
                }
                let point_c = explicit_layout[neighbors[nindex_c].item as usize];
                let mut dir_bc = sub(&point_c, point_b);
                dir_bc = div(&dir_bc, norm_l2(&dir_bc));
                let angle: f64 = dot(&dir_ab, &dir_bc);
                let error = 1.0 - angle;

                if error <= params.error_cutoff {
                    // Note: This matches triplclust *code* not the *paper*
                    // as far as I can tell
                    let mut dir = sub(&point_c, point_b);
                    dir = div(&dir, norm_l2(&dir));
                    candidates.push(Triplet {
                        index_a: neighbors[nindex_a].item as usize,
                        index_b: index_b,
                        index_c: neighbors[nindex_c].item as usize,
                        // Don't use add() here as this will cause extra alloc
                        centroid: [
                            (point_a[0] + point_b[0] + point_c[0]) / 3.0,
                            (point_a[1] + point_b[1] + point_c[1]) / 3.0,
                            (point_a[2] + point_b[2] + point_c[2]) / 3.0,
                        ],
                        direction: dir,
                        error: error,
                    });
                }
            }
        }
        candidates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for candidate in
            candidates[..std::cmp::min(params.max_candidates, candidates.len())].into_iter()
        {
            triplets.push(candidate.clone());
        }
    }

    return triplets;
}

/// The triplet distance metric defined in Daltiz et al. TriplClust paper
pub fn triplet_metric(lhs: &Triplet, rhs: &Triplet, scale: &f64) -> f64 {
    let rl_distance = sub(&rhs.centroid, &lhs.centroid);
    let lr_distance = mult(&rl_distance, -1.0);
    let distance_a: f64 = norm_l2(&add(
        &rl_distance,
        &mult(&lhs.direction, dot(&lhs.direction, &lr_distance)),
    ));

    let distance_b: f64 = norm_l2(&add(
        &lr_distance,
        &mult(&rhs.direction, dot(&rhs.direction, &rl_distance)),
    ));

    let cos_angle = dot(&rhs.direction, &lhs.direction).clamp(-1.0, 1.0);

    if cos_angle.abs() < 1.0e-8 {
        1.0e8
    } else {
        distance_a.max(distance_b) / scale + f64::abs(f64::tan(f64::acos(cos_angle)))
    }
}
