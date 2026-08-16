import h5py as h5
import numpy as np
from triplclust_py import (
    smooth_pointcloud,
    triplet_clustering,
    calculate_dnn,
    split_clusters,
)
import matplotlib.pyplot as plt
from tripclust import tripclust

EVENT = 41470  # good one
# EVENT = 38036


def load_test_pc() -> np.ndarray:
    file = h5.File("data/run_0054.h5")
    clusters = []
    return file["cloud"][f"cloud_{EVENT}"][:, :3]


def pc2csv(cloud):
    with open("cloud.csv", "w") as file:
        lines = []
        for row in cloud:
            lines.append(f"{row[0]} {row[1]} {row[2]}\n")
        file.writelines(lines)


def main():
    t_pc = load_test_pc()
    t_pc = t_pc[np.argsort(t_pc[:, 2])]
    pc2csv(t_pc)
    print("------------RUST-----------")
    dnn = calculate_dnn(t_pc)
    smooth_cloud = smooth_pointcloud(t_pc, dnn, 4.1)
    cluster_labels, unique_labels = triplet_clustering(
        smooth_cloud, 20, 2, 0.01, dnn, 13.0, 0.3, 5, "single"
    )
    cluster_labels, unique_labels = split_clusters(
        t_pc, cluster_labels, unique_labels, 25
    )
    print("---------C++---------------")

    tc = tripclust()
    tc.set_r(4.1)
    tc.set_rdnn(True)
    tc.set_k(20)
    tc.set_n(2)
    tc.set_a(0.01)
    tc.set_s(0.3)
    tc.set_sdnn(True)
    tc.set_t(13.0)
    tc.set_tauto(False)
    tc.set_dmax(0.0)
    tc.set_dmax_dnn(False)
    tc.set_ordered(True)
    # tc.set_link(params.tripclust_parameters.link)
    tc.set_m(5)
    tc.set_postprocess(True)
    tc.set_min_depth(25)

    # Perform tripclust (Dalitz) clustering
    tc.fill_pointcloud(t_pc)
    tc.perform_clustering()
    tc_labels: np.ndarray = tc.get_labels()
    tc_ulabels = np.unique(tc_labels)

    fig, ax = plt.subplots(
        1, 2, subplot_kw={"projection": "3d"}, constrained_layout=True
    )

    for label in unique_labels:
        if label == -1:
            continue
        cluster = t_pc[cluster_labels == label]
        ax[0].scatter(
            cluster[:, 0], cluster[:, 1], cluster[:, 2], label=f"RS Cluster {label}"
        )
    ax[0].legend()

    for label in tc_ulabels:
        if label == -1:
            continue
        cluster = t_pc[tc_labels == label]
        ax[1].scatter(
            cluster[:, 0], cluster[:, 1], cluster[:, 2], label=f"S-U Cluster {label}"
        )
    ax[1].legend()
    plt.show(block=True)


if __name__ == "__main__":
    main()
