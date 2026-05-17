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
