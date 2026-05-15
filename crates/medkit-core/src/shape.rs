use crate::{MedkitCoreError, Result};

/// Non-empty n-dimensional image or array shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Creates a shape and rejects empty or zero-valued dimensions.
    pub fn new(dims: impl Into<Vec<usize>>) -> Result<Self> {
        let dims = dims.into();
        if dims.is_empty() {
            return Err(MedkitCoreError::EmptyShape);
        }
        if let Some(index) = dims.iter().position(|dim| *dim == 0) {
            return Err(MedkitCoreError::ZeroDimension { index });
        }
        Ok(Self { dims })
    }

    /// Returns the rank.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Returns the dimension at `index`.
    pub fn dim(&self, index: usize) -> Option<usize> {
        self.dims.get(index).copied()
    }

    /// Returns all dimensions.
    pub fn as_slice(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the total number of elements.
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }
}
