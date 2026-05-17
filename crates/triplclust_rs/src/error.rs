#[derive(Debug, Clone)]
pub enum SmoothingError {
    InvalidMaxNeighbors(i32),
    EmptyPointCloud,
}

impl std::fmt::Display for SmoothingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMaxNeighbors(val) => write!(
                f,
                "Invalid max neighbors parameter: {}, value must be > 0",
                val
            ),
            Self::EmptyPointCloud => {
                write!(f, "Smoothing operation resulted in an empty point cloud")
            }
        }
    }
}

impl std::error::Error for SmoothingError {}

#[derive(Debug, Clone)]
pub enum TripletError {
    InvalidNeighborhoodSize(i32),
    InvalidMaxCandidates(i32),
    EmptyPointCloud,
    PointCloudNotSorted,
}

impl std::fmt::Display for TripletError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidNeighborhoodSize(size) => write!(
                f,
                "Invalid neighborhood size parameter: {}, value must be > 0",
                size
            ),
            Self::InvalidMaxCandidates(size) => write!(
                f,
                "Invalid max candidates parameter: {}, value must be > 0",
                size
            ),
            Self::EmptyPointCloud => write!(f, "Empty point cloud was given to triplet operation"),
            Self::PointCloudNotSorted => write!(
                f,
                "Point cloud given to triplet operation was not sorted in Z"
            ),
        }
    }
}

impl std::error::Error for TripletError {}

#[derive(Debug, Clone)]
pub enum ClusterError {
    InvalidMinClusterSize(i32),
    InvalidLinkage(String),
    InvalidArguments,
    EmptyTriplets,
}

impl std::fmt::Display for ClusterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMinClusterSize(size) => write!(
                f,
                "Invalid min cluster size parameter: {}, value must be > 0",
                size
            ),
            Self::InvalidLinkage(val) => write!(
                f,
                "Invalid linkage parameter: {}, value must be one of [single, complete, average, median]",
                val
            ),
            Self::InvalidArguments => write!(
                f,
                "Both dnn and scale were None when constructing cluster parameters; at least one must be given"
            ),
            Self::EmptyTriplets => write!(
                f,
                "An empty triplet array was given to the clustering operation"
            ),
        }
    }
}

impl std::error::Error for ClusterError {}

#[derive(Debug, Clone)]
pub enum TriplclustError {
    Smoothing(SmoothingError),
    Triplet(TripletError),
    Cluster(ClusterError),
}

impl std::fmt::Display for TriplclustError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Smoothing(err) => write!(
                f,
                "Triplclust encoutered an error during smoothing: {}",
                err
            ),
            Self::Triplet(err) => write!(
                f,
                "Triplclust encoutered an error during evaluting triplets: {}",
                err
            ),
            Self::Cluster(err) => write!(
                f,
                "Triplclust encoutered an error during clustering: {}",
                err
            ),
        }
    }
}

impl std::error::Error for TriplclustError {}

impl From<SmoothingError> for TriplclustError {
    fn from(value: SmoothingError) -> Self {
        Self::Smoothing(value)
    }
}

impl From<TripletError> for TriplclustError {
    fn from(value: TripletError) -> Self {
        Self::Triplet(value)
    }
}

impl From<ClusterError> for TriplclustError {
    fn from(value: ClusterError) -> Self {
        Self::Cluster(value)
    }
}
