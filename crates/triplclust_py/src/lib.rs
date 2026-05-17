mod error;

use error::PyTriplclustError;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use triplclust_rs::cluster::cluster;
use triplclust_rs::dnn::dnn_first_quartile;
use triplclust_rs::params::{ClusterParams, SmoothParams, TripletParams};
use triplclust_rs::smooth::smooth_pointcloud as rs_smooth_pointcloud;
use triplclust_rs::triplet::evaluate_triplets;

#[pyfunction]
pub fn calculate_dnn<'py>(_py: Python<'py>, cloud: PyReadonlyArray2<f64>) -> f64 {
    dnn_first_quartile(&cloud.as_array())
}

#[pyfunction]
pub fn smooth_pointcloud<'py>(
    py: Python<'py>,
    cloud: PyReadonlyArray2<f64>,
    dnn: Option<f64>,
    neighborhood_radius: Option<f64>,
) -> Result<Bound<'py, PyArray2<f64>>, PyTriplclustError> {
    let params = match dnn {
        Some(val) => SmoothParams::default_with_dnn(val),
        None => match neighborhood_radius {
            Some(nval) => SmoothParams {
                neighborhood_radius: nval,
            },
            None => SmoothParams::default_with_dnn(dnn_first_quartile(&cloud.as_array())),
        },
    };
    let smoothed = rs_smooth_pointcloud(&cloud.as_array(), &params)?;
    Ok(smoothed.to_pyarray(py))
}

#[pyfunction]
pub fn triplet_clustering<'py>(
    py: Python<'py>,
    smoothed_point_cloud: PyReadonlyArray2<f64>,
    triplet_neighborhood_size: i32,
    triplet_max_candidates: i32,
    triplet_error_cutoff: f64,
    dnn: Option<f64>,
    cluster_distance_threshold: Option<f64>,
    cluster_scale: Option<f64>,
    min_cluster_size: i32,
    linkage: &str,
) -> Result<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>), PyTriplclustError> {
    let triplet_params = TripletParams::from_fullargs(
        triplet_neighborhood_size,
        triplet_max_candidates,
        triplet_error_cutoff,
    )?;

    let cluster_params = if dnn.is_some() && cluster_scale.is_none() {
        ClusterParams::from_fullargs(
            dnn,
            cluster_scale,
            cluster_distance_threshold,
            min_cluster_size,
            linkage,
        )?
    } else {
        ClusterParams::from_fullargs(
            Some(dnn_first_quartile(&smoothed_point_cloud.as_array())),
            cluster_scale,
            cluster_distance_threshold,
            min_cluster_size,
            linkage,
        )?
    };
    let cloud_array = smoothed_point_cloud.as_array();
    let triplets = evaluate_triplets(&cloud_array, &triplet_params);
    let result = cluster(cloud_array.nrows(), &triplets, &cluster_params)?;

    Ok((
        result.labels.to_pyarray(py),
        result.unique_labels.to_pyarray(py),
    ))
}

#[pymodule]
fn triplclust_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_dnn, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_pointcloud, m)?)?;
    m.add_function(wrap_pyfunction!(triplet_clustering, m)?)?;
    Ok(())
}
