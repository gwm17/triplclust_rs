use criterion::{Criterion, criterion_group, criterion_main};
use numpy::ndarray::Array2;
use triplclust_rs::{
    cluster::cluster, dnn::dnn_first_quartile, params::ClusterParams, params::SmoothParams,
    params::TripletParams, smooth::smooth_pointcloud, triplet::evaluate_triplets,
};

fn load_data_and_params() -> (Array2<f64>, SmoothParams, TripletParams, ClusterParams) {
    let test_point_cloud = match triplclust_rs::utils::load_test_data() {
        Ok(pc) => pc,
        Err(err) => {
            println!("Failed to load test data with error {err}");
            panic!();
        }
    };
    let int_scale = dnn_first_quartile(&test_point_cloud.view());
    let smooth_params = SmoothParams::default_with_dnn(int_scale);
    let triplet_params =
        TripletParams::from_fullargs(19, 2, 0.03).expect("Invalid triplet parameters");
    let cluster_params =
        ClusterParams::default_with_dnn(int_scale, "single").expect("Invalid cluster parameters");
    (
        test_point_cloud,
        smooth_params,
        triplet_params,
        cluster_params,
    )
}

fn complete_pass(
    data: &Array2<f64>,
    sparams: &SmoothParams,
    tparams: &TripletParams,
    cparams: &ClusterParams,
) {
    let smooth_cloud = triplclust_rs::smooth::smooth_pointcloud(&data.view(), &sparams)
        .expect("Smoothing failed!");
    let triplets = evaluate_triplets(&smooth_cloud.view(), &tparams);
    let _ = cluster(smooth_cloud.nrows(), &triplets, &cparams);
}

fn criterion_benchmark(c: &mut Criterion) {
    let (test_point_cloud, smooth_params, triplet_params, cluster_params) = load_data_and_params();
    let mut group = c.benchmark_group("triplclust_rs");

    group.bench_function("dnn", |b| {
        b.iter(|| dnn_first_quartile(&test_point_cloud.view()));
    });

    group.bench_function("smoothing", |b| {
        b.iter_with_large_drop(|| {
            let _ = smooth_pointcloud(&test_point_cloud.view(), &smooth_params);
        })
    });
    let smooth_cloud =
        triplclust_rs::smooth::smooth_pointcloud(&test_point_cloud.view(), &smooth_params)
            .expect("Smoothing failed!");
    group.bench_function("triplets", |b| {
        b.iter_with_large_drop(|| evaluate_triplets(&smooth_cloud.view(), &triplet_params))
    });
    let triplets = evaluate_triplets(&smooth_cloud.view(), &triplet_params);
    group.bench_function("clustering", |b| {
        b.iter_with_large_drop(|| cluster(smooth_cloud.nrows(), &triplets, &cluster_params))
    });
    group.bench_function("complete_pass", |b| {
        b.iter(|| {
            complete_pass(
                &test_point_cloud,
                &smooth_params,
                &triplet_params,
                &cluster_params,
            )
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
