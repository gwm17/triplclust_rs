use super::params::TripletParams;
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use numpy::ndarray::ArrayView2;

#[derive(Debug, Clone)]
pub struct Triplet {
    pub index_a: usize,
    pub index_b: usize,
    pub index_c: usize,
    pub centroid: [f64; 3],
    pub direction: [f64; 3],
    pub error: f64,
}

impl PartialEq for Triplet {
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

fn norm_l2(x: &[f64]) -> f64 {
    return f64::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

fn sub(x: &[f64], y: &[f64]) -> [f64; 3] {
    return [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
}

fn add(x: &[f64], y: &[f64]) -> [f64; 3] {
    return [x[0] + y[0], x[1] + y[1], x[2] + y[2]];
}

fn div(x: &[f64], s: f64) -> [f64; 3] {
    return [x[0] / s, x[1] / s, x[2] / s];
}

fn mult(x: &[f64], s: f64) -> [f64; 3] {
    return [x[0] * s, x[1] * s, x[2] * s];
}

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
                    let mut dir = sub(&point_c, point_b);
                    dir = div(&dir, norm_l2(&dir));
                    candidates.push(Triplet {
                        index_a: neighbors[nindex_a].item as usize,
                        index_b: index_b,
                        index_c: neighbors[nindex_c].item as usize,
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
