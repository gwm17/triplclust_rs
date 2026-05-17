//! Structs wrapping parameter sets used throughout the pacakge.
//! These structs have default implementations for basic use, but
//! in general are highly tuneable.
use super::error::{ClusterError, TriplclustError, TripletError};
use kodama::Method;
use std::num::NonZero;

/// Helper function to validate that conversions from signed to unsigned
/// are good.
fn validate_i32_to_nonzero_usize(val: i32) -> Option<NonZero<usize>> {
    if val < 0 {
        None
    } else {
        NonZero::new(val as usize)
    }
}

/// Parameters used in the point cloud nearest-neighbor smoothing
#[derive(Debug, Clone)]
pub struct SmoothParams {
    /// The neighborhood radius, in the same units as the point cloud distances.
    pub neighborhood_radius: f64,
}

impl SmoothParams {
    /// Create a default set of parameters from a given value
    /// of the intrinsic length scale dNN
    /// neighborhood_radius = 2.0 * dNN
    pub fn default_with_dnn(dnn: f64) -> Self {
        Self {
            neighborhood_radius: dnn * 2.0,
        }
    }
}

/// Parameters used in triplet formation
#[derive(Debug, Clone)]
pub struct TripletParams {
    /// The neighborhood size in number of  points
    pub neighborhood_size: NonZero<usize>,
    /// The maximum number of triplet candidates to keep for each point
    pub max_candidates: usize,
    /// The error cutoff for determining if a triplet is valid or not.
    /// I.e. 1 - cos(alpha) must be less than this value. (See Dalitz paper for details)
    pub error_cutoff: f64,
}

impl Default for TripletParams {
    /// Create a default set of triplet parameters
    /// neighborhood_size = 19
    /// max_candidates = 2
    /// error_cutoff = 0.03
    fn default() -> Self {
        Self {
            neighborhood_size: NonZero::new(19).unwrap(),
            max_candidates: 2,
            error_cutoff: 0.03,
        }
    }
}

impl TripletParams {
    /// Helper function for creating parameters when marshalling from Python
    pub fn from_fullargs(n_size: i32, max_c: i32, e_c: f64) -> Result<Self, TriplclustError> {
        let valid_n_size = match validate_i32_to_nonzero_usize(n_size) {
            Some(val) => val,
            None => return Err(TripletError::InvalidNeighborhoodSize(n_size).into()),
        };
        let valid_max_c = match validate_i32_to_nonzero_usize(max_c) {
            Some(val) => val.get(),
            None => return Err(TripletError::InvalidMaxCandidates(max_c).into()),
        };

        Ok(Self {
            neighborhood_size: valid_n_size,
            max_candidates: valid_max_c,
            error_cutoff: e_c,
        })
    }
}

/// Enum representing types of clustering linkages to be used
/// This type implements translation from string values for use
/// with Python.
#[derive(Debug, Clone)]
pub enum Linkage {
    Single,
    Complete,
    Average,
    Median,
}

/// Translate triplclust_rs::Linkage to kodama::Linkage
impl Into<Method> for Linkage {
    fn into(self) -> Method {
        match self {
            Self::Single => Method::Single,
            Self::Complete => Method::Complete,
            Self::Average => Method::Average,
            Self::Median => Method::Median,
        }
    }
}

/// Translate string to Linkage
impl TryFrom<&str> for Linkage {
    type Error = TriplclustError;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        if value == "single" {
            Ok(Self::Single)
        } else if value == "complete" {
            Ok(Self::Complete)
        } else if value == "average" {
            Ok(Self::Average)
        } else if value == "median" {
            Ok(Self::Median)
        } else {
            Err(ClusterError::InvalidLinkage(value.to_string()).into())
        }
    }
}

/// Parameters for use with clustering
#[derive(Debug, Clone)]
pub struct ClusterParams {
    /// Optional cluster distance threshold
    pub cluster_distance_threshold: Option<f64>,
    /// Scale factor used in triplet distance calculation. See Dalitz paper.
    pub scale: f64,
    /// Minimum cluster size
    pub min_cluster_size: usize,
    /// Clustering linkage type
    pub linkage: Linkage,
}

impl ClusterParams {
    /// Create a default set of clustering parameters from a value for the intrisinc
    /// length scale dNN and a Linkage string
    /// cluster_distance_threshold = None
    /// scale = dNN * 0.3 -- Note this matches the triplclust *code* but not the *paper*
    /// min_cluster_size = 5
    /// linkage = Linkage::from(string)
    pub fn default_with_dnn(dnn: f64, linkage: &str) -> Result<Self, TriplclustError> {
        Ok(Self {
            cluster_distance_threshold: None,
            scale: dnn * 0.3,
            min_cluster_size: 5,
            linkage: linkage.try_into()?,
        })
    }

    /// Helper function to marshall a full set of Python arguments into
    /// Rust parameters
    pub fn from_fullargs(
        dnn: Option<f64>,
        scale: Option<f64>,
        cdt: Option<f64>,
        mcs: i32,
        linkage: &str,
    ) -> Result<Self, TriplclustError> {
        let valid_mcs = match validate_i32_to_nonzero_usize(mcs) {
            Some(val) => val.get(),
            None => return Err(ClusterError::InvalidMinClusterSize(mcs))?,
        };
        if let Some(dnn_val) = dnn {
            Ok(Self {
                cluster_distance_threshold: cdt,
                scale: dnn_val / 3.0,
                min_cluster_size: valid_mcs,
                linkage: linkage.try_into()?,
            })
        } else {
            match scale {
                Some(val) => Ok(Self {
                    cluster_distance_threshold: cdt,
                    scale: val,
                    min_cluster_size: valid_mcs,
                    linkage: linkage.try_into()?,
                }),
                None => Err(ClusterError::InvalidArguments)?,
            }
        }
    }
}
