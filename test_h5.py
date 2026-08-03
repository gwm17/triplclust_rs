import h5py as h5
import numpy as np
from triplclust_py import (
    smooth_pointcloud,
    triplet_clustering,
    calculate_dnn,
    split_clusters,
)
import matplotlib.pyplot as plt

EVENT = 40037


def load_test_data() -> list[np.ndarray]:
    file = h5.File("data/o16_data/clusters/run_0054.h5")
    clusters = []
    cgroup = file["cluster"][f"event_{EVENT}"]
    for id in cgroup:
        clusters.append(cgroup[id]["cloud"][:])

    return clusters


def load_test_pc() -> np.ndarray:
    file = h5.File("data/o16_data/point_clouds/run_0054.h5")
    clusters = []
    return file["cloud"][f"cloud_{EVENT}"][:, :3]


def pc2csv(cloud):
    with open("cloud.csv", "w") as file:
        lines = []
        for row in cloud:
            lines.append(f"{row[0]} {row[1]} {row[2]}\n")
        file.writelines(lines)


def main():
    t_clusters = load_test_data()
    t_pc = load_test_pc()
    t_pc = t_pc[np.argsort(t_pc[:, 2])]
    pc2csv(t_pc)
    dnn = calculate_dnn(t_pc)
    smooth_cloud = smooth_pointcloud(t_pc, dnn, 4.1)
    cluster_labels, unique_labels = triplet_clustering(
        smooth_cloud, 20, 2, 0.01, dnn, 13.0, 0.3, 5, "single"
    )
    cluster_labels, unique_labels = split_clusters(
        t_pc, cluster_labels, unique_labels, 25
    )

    fig, ax = plt.subplots(
        1, 3, subplot_kw={"projection": "3d"}, constrained_layout=True
    )

    ax[2].scatter(t_pc[:, 0], t_pc[:, 1], t_pc[:, 2])
    for idx, cluster in enumerate(t_clusters):
        ax[0].scatter(
            cluster[:, 0], cluster[:, 1], cluster[:, 2], label=f"Cluster {idx}"
        )
    ax[0].legend()
    for label in unique_labels:
        if label == -1:
            continue
        cluster = t_pc[cluster_labels == label]
        ax[1].scatter(
            cluster[:, 0], cluster[:, 1], cluster[:, 2], label=f"RS Cluster {label}"
        )
    ax[1].legend()
    plt.show(block=True)


if __name__ == "__main__":
    main()
