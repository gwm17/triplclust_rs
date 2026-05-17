use pyo3::PyErr;
use triplclust_rs::error::TriplclustError;

#[derive(Debug, Clone)]
pub enum PyTriplclustError {
    Triplclust(TriplclustError),
}

impl From<TriplclustError> for PyTriplclustError {
    fn from(value: TriplclustError) -> Self {
        Self::Triplclust(value)
    }
}

impl std::fmt::Display for PyTriplclustError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Triplclust(val) => write!(f, "{}", val),
        }
    }
}

impl std::error::Error for PyTriplclustError {}

impl Into<PyErr> for PyTriplclustError {
    fn into(self) -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(format!("{self}"))
    }
}
