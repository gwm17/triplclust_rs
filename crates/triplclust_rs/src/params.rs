use super::error::{ClusterError, TriplclustError, TripletError};
use kodama::Method;
use std::num::NonZero;

fn validate_i32_to_nonzero_usize(val: i32) -> Option<NonZero<usize>> {
    if val < 0 {
        None
    } else {
        NonZero::new(val as usize)
    }
}

#[derive(Debug, Clone)]
pub struct SmoothParams {
    pub neighborhood_radius: f64,
}

impl SmoothParams {
    pub fn default_with_dnn(dnn: f64) -> Self {
        Self {
            neighborhood_radius: dnn * 2.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TripletParams {
    pub neighborhood_size: NonZero<usize>,
    pub max_candidates: usize,
    pub error_cutoff: f64,
}

impl Default for TripletParams {
    fn default() -> Self {
        Self {
            neighborhood_size: NonZero::new(19).unwrap(),
            max_candidates: 2,
            error_cutoff: 0.03,
        }
    }
}

impl TripletParams {
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

#[derive(Debug, Clone)]
pub enum Linkage {
    Single,
    Complete,
    Average,
    Median,
}

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

#[derive(Debug, Clone)]
pub struct ClusterParams {
    pub cluster_distance_threshold: Option<f64>,
    pub scale: f64,
    pub min_cluster_size: usize,
    pub linkage: Linkage,
}

impl ClusterParams {
    pub fn default_with_dnn(dnn: f64, linkage: &str) -> Result<Self, TriplclustError> {
        Ok(Self {
            cluster_distance_threshold: None,
            scale: dnn * 0.3,
            min_cluster_size: 5,
            linkage: linkage.try_into()?,
        })
    }

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
