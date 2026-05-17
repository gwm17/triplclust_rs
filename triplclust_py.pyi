import numpy as np

def calculate_dnn(cloud: np.ndarray) -> float:
    """Calculate the characteristic length scale of the point cloud

    Parameters
    ----------
    cloud: numpy.ndarray
        The point cloud

    Returns
    -------
    float:
        The characteristic length scale
    """
    ...

def smooth_pointcloud(
    cloud: np.ndarray, dnn: float | None, neighborhood_radius: float | None
) -> np.ndarray:
    """Smooth a point cloud which has already been sorted in z

    Perform nearest neighbor smoothing on a pointcloud. If dnn is given,
    the neighborhood radius is calculated from that scale. If dnn is not given
    and neighborhood_radius is given, neighborhood_radius is used. If neither
    are given, dnn is calculated from the cloud. This last option should *only*
    be used if dnn is not going to be used in subsequent calculations (i.e. clustering).

    Parameters
    ----------
    cloud: numpy.ndarray
        The pointcloud to smooth
    dnn: float | None
        The characteristic length scale
    neighborhood_radius: float | None
        The maximum raidal distance between neighbors

    Returns
    -------
    numpy.ndarray
        A new smoothed pointcloud
    """
    ...

def triplet_clustering(
    smoothed_cloud: np.ndarray,
    triplet_neighborhood_size: int,
    triplet_max_candidates: int,
    triplet_error_cutoff: float,
    dnn: float | None,
    cluster_distance_threshold: float | None,
    cluster_scale: float | None,
    min_cluster_size: int,
    linkage: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the triplclust algorithm to a smoothed point cloud

    Cluster the points by the triplclust triplet metric

    Parameters
    ----------
    smoothed_cloud: numpy.ndarray
        The smoothed, sorted pointcloud
    triplet_neighborhood_size: int
        The size of the triplet search neighborhood in points
    triplet_max_candidates: int
        The maximum of triplet candidates to consider for a point
    triplet_error_cutoff: float
        The error cutoff for evaluating triplet candidates
    dnn: float | None
        If given, dnn is used to calculate the comparison scale
        instead of the cluster scale parameter. See cluster_scale
        for case where both dnn and cluster_scale are None.
    cluster_distance_threshold: float | None
        The cluster distance used as a stopping criterion in the
        hierarchical clustering. If None, the appropriate threshold
        is calculated from the data.
    cluster_scale: float | None
        The scale factor used in the triplet distance metric. If None and dnn is
        None, dnn is calculated from the point cloud. This is not recommended,
        as dnn is typically used in multiple places and is expensive to calculate.
    min_cluster_size: int
        The minimum number of points required for a cluster to be valid.
    linkage: str
        The type of linkage to use in the agglometrive clustering.
        Valid values are "single", "complete", "average", and "median".

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A set of 1-D integer arrays, where the first is the set of cluster
        labels for each point in the point cloud and the second is the list
        of unique cluster labels. A label of -1 indicates that the point was
        not included in any *valid* cluster.
    """
    ...
