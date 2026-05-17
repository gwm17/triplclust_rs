import numpy as np
import matplotlib.pyplot as plt
from triplclust_py import smooth_pointcloud, calculate_dnn, triplet_clustering


def load_test_data() -> np.ndarray:
    with open("data/test.dat") as dfile:
        lines = dfile.readlines()[3:]
        cloud = np.zeros((len(lines), 3))
        for row, line in enumerate(lines):
            elems = line.split(" ")
            cloud[row, 0] = float(elems[0])
            cloud[row, 1] = float(elems[1])
            cloud[row, 2] = float(elems[2])
    return cloud


def main():
    data = load_test_data()
    dnn = calculate_dnn(data)
    smooth_cloud = smooth_pointcloud(data, dnn, None)
    cluster_labels, unique_labels = triplet_clustering(
        smooth_cloud, 19, 2, 0.03, dnn, None, None, 5, "single"
    )

    fig, ax = plt.subplots(
        1, 1, subplot_kw={"projection": "3d"}, constrained_layout=True
    )
    print(len(cluster_labels))
    print(unique_labels)
    for label in unique_labels:
        mask = cluster_labels == label
        ax.scatter(
            data[mask, 0], data[mask, 1], data[mask, 2], label=f"Cluster {label}"
        )
    ax.legend()
    plt.show(block=True)


if __name__ == "__main__":
    main()
