use super::error::{SmoothingError, TriplclustError};
use super::params::SmoothParams;
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use numpy::ndarray::{Array2, ArrayView2, array, s};

/// Smooth a 3D point cloud using nearest neighbors.
///
/// Search for the neearest neighbors of each point within a given distance radius
/// and then average the position of those points.
///
/// Returns a new, smoothed point cloud of the same dimensions as the original.
pub fn smooth_pointcloud(
    cloud: &ArrayView2<f64>,
    params: &SmoothParams,
) -> Result<Array2<f64>, TriplclustError> {
    if cloud.len() == 0 {
        return Err(SmoothingError::EmptyPointCloud)?;
    }

    let mut output_array: Array2<f64> = Array2::zeros(cloud.raw_dim());
    let explicit_layout: Vec<[f64; 3]> = cloud
        .rows()
        .into_iter()
        .map(|point| [point[0], point[1], point[2]])
        .collect();
    let tree = ImmutableKdTree::<f64, 3>::new_from_slice(&explicit_layout);

    for (idx, point) in explicit_layout.iter().enumerate() {
        let neighbors = tree.within::<SquaredEuclidean>(
            &point,
            params.neighborhood_radius * params.neighborhood_radius,
        );
        let flen = neighbors.len() as f64;
        output_array.slice_mut(s![idx, ..]).assign(
            &array!(neighbors.iter().fold([0.0, 0.0, 0.0], |mut acc, point| {
                acc[0] += explicit_layout[point.item as usize][0] / flen;
                acc[1] += explicit_layout[point.item as usize][1] / flen;
                acc[2] += explicit_layout[point.item as usize][2] / flen;
                acc
            }))
            .flatten(),
        );
    }
    Ok(output_array)
}
