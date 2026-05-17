use kiddo::{ImmutableKdTree, SquaredEuclidean};
use numpy::ndarray::ArrayView2;
use std::num::NonZero;

pub fn mean_sq_distance(explicit_cloud: Vec<[f64; 3]>, neighbors: NonZero<usize>) -> Vec<f64> {
    let tree = ImmutableKdTree::<f64, 3>::new_from_slice(&explicit_cloud);
    let mut msd = vec![0.0; explicit_cloud.len()];

    for (idx, point) in explicit_cloud.iter().enumerate() {
        let neighbors = tree.nearest_n::<SquaredEuclidean>(&point, neighbors);
        let count = neighbors.iter().fold((0.0, 0), |mut count, neighbor| {
            if neighbor.distance > 0.0 {
                count.0 += neighbor.distance;
                count.1 += 1;
            }
            count
        });
        msd[idx] = count.0 / (count.1 as f64)
    }

    return msd;
}

pub fn dnn_first_quartile(cloud: &ArrayView2<f64>) -> f64 {
    let explicit_layout: Vec<[f64; 3]> = cloud
        .rows()
        .into_iter()
        .map(|point| [point[0], point[1], point[2]])
        .collect();
    let mut msd = mean_sq_distance(explicit_layout, NonZero::new(2).unwrap());
    msd.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let quartile_index = msd.len() / 4;
    return msd[quartile_index].sqrt();
}
