//! # triplclust_rs
//!
//! triplclust_rs is a Rust version of the [triplclust](https://github.com/cdalitz/triplclust)
//! algorithm developed by C. Dalitz and described in the following paper
//!
//! > C. Dalitz, J. Wilberg, L. Aymans: "TriplClust: An Algorithm for Curve Detection in
//! > 3D Point Clouds." Image Processing Online 9, pp. 26-46 (2019).
//! > https://doi.org/10.5201/ipol.2019.234
//!
//! This version of the algorithm is tailored to interact well with data produced during
//! the [Spyral](https://github.com/attpc/Spyral) analysis framework, with first class
//! support for numpy array point clouds.
//!
//! ## Install
//!
//! To add it to your rust project use
//!
//! ```bash
//! cargo add triplclust_rs
//! ```
//!
//! ## How it works
//!
//! There are two primary stages to the analysis, smoothing and clustering. Smoothing
//! is a data preparation stage where the point cloud is de-noised using a nearest-neighbors
//! approach. The second stage is the clustering, where each point in the cloud is composed
//! into it valid set of triplets (3 points aligned in a direction) and those triplets
//! are clustered by a distance metric outlined in the paper. In general, this library
//! follows the original implementation with one exception:
//!
//! The original implementation allowed for overlapping labels between clusters. This
//! can occur due to the fact that points are not uniquely assigned to triplets. `triplclust_rs`
//! tries to handle this better by using metrics to disentangle overlapping labels. Currently,
//! only one method is available, and that is to collapse by the most frequently assigned label
//! for a point. For example, if a point recieved the labels [1, 1, 1, 2, 2] from it's parent
//! triplets, it would resut in the label 1 as 1 was assigned most frequently.
//!
//! ## Python bindings
//!
//! Python bindings are provided via the triplclust_py crate, and the Python package is
//! hosted on PyPI as [triplclust_py](https://pypi.org/project/triplclust_py).
//!
//! ## Peformance
//!
//! triplclust_rs was written with performance in mind. Benchmarks are included and can
//! be run using
//!
//! ```bash
//! cargo bench
//! ```
//!
//! Early testing has shown that triplclust_rs is ~2x faster than the original implementation.
//! This has *only* been performed on test data, and a small point cloud at that, so should
//! be taken with a big grain of salt. Additionally, the original implementation is tricky
//! to benchmark 1-to-1 as it was written more as a test application. More testing is needed
//! to validate these early results.
//!
//! Benchmarks are run using the [criterion](https://docs.rs/criterion) crate.

pub mod cluster;
pub mod dnn;
pub mod error;
pub mod params;
pub mod smooth;
pub mod triplet;
pub mod utils;

#[cfg(test)]
mod tests {

    use super::*;
    static PRECISION: f64 = 1.0e-4;

    #[test]
    fn full_clustering() {
        let test_point_cloud = match utils::load_test_data() {
            Ok(pc) => pc,
            Err(err) => {
                println!("Failed to load test data with error {err}");
                panic!();
            }
        };
        let (
            test_dnn,
            test_radius,
            test_scale,
            test_thresh,
            _,
            test_smooth,
            test_labels,
            unique_test_labels,
        ) = match utils::load_test_results() {
            Ok(res) => res,
            Err(err) => {
                println!("Failed to load test results with error {err}");
                panic!();
            }
        };

        let int_scale = dnn::dnn_first_quartile(&test_point_cloud.view());
        assert!((test_dnn - int_scale).abs() < PRECISION);
        let smooth_params = params::SmoothParams::default_with_dnn(int_scale);
        assert!((test_radius - smooth_params.neighborhood_radius).abs() < PRECISION);
        let triplet_params =
            params::TripletParams::from_fullargs(19, 2, 0.03).expect("Invalid triplet parameters");
        let cluster_params = params::ClusterParams::default_with_dnn(int_scale, "single")
            .expect("Invalid cluster parameters!");
        assert!((test_scale - cluster_params.scale).abs() < PRECISION);

        let cloud_view = test_point_cloud.view();
        let smooth_cloud =
            smooth::smooth_pointcloud(&cloud_view, &smooth_params).expect("Smoothing failed!");
        assert_eq!(smooth_cloud.len(), test_smooth.len());
        for ridx in 0..smooth_cloud.nrows() {
            for cidx in 0..smooth_cloud.ncols() {
                let smooth_val = smooth_cloud[(ridx, cidx)];
                let test_val = test_smooth[(ridx, cidx)];
                let delta = (smooth_val - test_val).abs();
                assert!(delta < PRECISION);
            }
        }
        let triplets = triplet::evaluate_triplets(&smooth_cloud.view(), &triplet_params);
        let clusters = cluster::cluster(smooth_cloud.nrows(), &triplets, &cluster_params)
            .expect("Clustering failed!");

        assert_ne!(clusters.labels.len(), 0);
        assert!(clusters.unique_labels.len() > 1);
        assert!((clusters.optimal_cdt - test_thresh).abs() < PRECISION);
        assert_eq!(clusters.labels.len(), test_labels.len());
        assert!(unique_test_labels.len() == clusters.unique_labels.len());
        for label in clusters.unique_labels.iter() {
            assert!(unique_test_labels.contains(label));
        }
        for (idx, label) in clusters.labels.iter().enumerate() {
            println!("{}:{} -- {:?}", idx, label, test_labels[idx]);
            assert!(test_labels[idx].contains(label));
        }
    }
}
