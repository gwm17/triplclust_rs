mod error;

use error::PyTriplclustError;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use triplclust_rs::cluster::cluster;
use triplclust_rs::dnn::dnn_first_quartile;
use triplclust_rs::params::{ClusterParams, SmoothParams, TripletParams};
use triplclust_rs::smooth::smooth_pointcloud as rs_smooth_pointcloud;
use triplclust_rs::split;
use triplclust_rs::triplet::evaluate_triplets;

#[pyfunction]
pub fn calculate_dnn<'py>(_py: Python<'py>, cloud: PyReadonlyArray2<f64>) -> f64 {
    dnn_first_quartile(&cloud.as_array())
}

#[pyfunction]
#[pyo3(signature = (cloud, dnn, neighborhood_radius = 2.0))]
pub fn smooth_pointcloud<'py>(
    py: Python<'py>,
    cloud: PyReadonlyArray2<f64>,
    dnn: Option<f64>,
    neighborhood_radius: f64,
) -> Result<Bound<'py, PyArray2<f64>>, PyTriplclustError> {
    let params = SmoothParams::new(neighborhood_radius, dnn);
    let smoothed = rs_smooth_pointcloud(&cloud.as_array(), &params)?;
    Ok(smoothed.to_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (
        smoothed_point_cloud,
        dnn,
        triplet_neighborhood_size=19,
        triplet_max_candidates=2,
        triplet_error_cutoff=0.03,
        cluster_scale=0.3,
        min_cluster_size=5,
        linkage="single",
        cluster_distance_threshold=None
    )
)]
pub fn triplet_clustering<'py>(
    py: Python<'py>,
    smoothed_point_cloud: PyReadonlyArray2<f64>,
    dnn: Option<f64>,
    triplet_neighborhood_size: i32,
    triplet_max_candidates: i32,
    triplet_error_cutoff: f64,
    cluster_scale: f64,
    min_cluster_size: i32,
    linkage: &str,
    cluster_distance_threshold: Option<f64>,
) -> Result<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>), PyTriplclustError> {
    let triplet_params = TripletParams::new(
        triplet_neighborhood_size,
        triplet_max_candidates,
        triplet_error_cutoff,
    )?;

    let cluster_params = ClusterParams::new(
        dnn,
        cluster_scale,
        cluster_distance_threshold,
        min_cluster_size,
        linkage,
    )?;
    let cloud_array = smoothed_point_cloud.as_array();
    let triplets = evaluate_triplets(&cloud_array, &triplet_params);
    let result = cluster(cloud_array.nrows(), &triplets, &cluster_params)?;

    Ok((
        result.labels.to_pyarray(py),
        result.unique_labels.to_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (point_cloud, labels, unqiue_labels, min_depth=25))]
pub fn split_clusters<'py>(
    py: Python<'py>,
    point_cloud: PyReadonlyArray2<f64>,
    labels: PyReadonlyArray1<i32>,
    unqiue_labels: PyReadonlyArray1<i32>,
    min_depth: i32,
) -> Result<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>), PyTriplclustError> {
    let (labels, unis) = split::split_clusters(
        &point_cloud.as_array(),
        &labels.as_array(),
        &unqiue_labels.as_array(),
        min_depth as usize,
    )?;
    Ok((labels.to_pyarray(py), unis.to_pyarray(py)))
}

#[pymodule]
fn triplclust_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_dnn, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_pointcloud, m)?)?;
    m.add_function(wrap_pyfunction!(triplet_clustering, m)?)?;
    m.add_function(wrap_pyfunction!(split_clusters, m)?)?;
    Ok(())
}
