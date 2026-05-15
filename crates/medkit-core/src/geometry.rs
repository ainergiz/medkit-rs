use crate::{MedkitCoreError, Result, Shape};

/// Patient/world coordinate system convention.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CoordinateSystem {
    /// Right-anterior-superior convention.
    RAS,
    /// Left-posterior-superior convention.
    LPS,
    /// Unknown or named external convention.
    Other(String),
}

/// Spatial geometry of an image grid.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialGeometry {
    shape: Shape,
    spacing: Vec<f64>,
    origin: Vec<f64>,
    direction: Vec<f64>,
    coordinate_system: CoordinateSystem,
}

impl SpatialGeometry {
    /// Creates a spatial geometry with explicit spacing, origin, and direction.
    pub fn new(
        shape: Shape,
        spacing: impl Into<Vec<f64>>,
        origin: impl Into<Vec<f64>>,
        direction: impl Into<Vec<f64>>,
        coordinate_system: CoordinateSystem,
    ) -> Result<Self> {
        let rank = shape.rank();
        let spacing = spacing.into();
        let origin = origin.into();
        let direction = direction.into();
        if spacing.len() != rank {
            return Err(MedkitCoreError::SpacingRankMismatch {
                spacing: spacing.len(),
                rank,
            });
        }
        if origin.len() != rank {
            return Err(MedkitCoreError::OriginRankMismatch {
                origin: origin.len(),
                rank,
            });
        }
        let expected = rank * rank;
        if direction.len() != expected {
            return Err(MedkitCoreError::DirectionSizeMismatch {
                values: direction.len(),
                expected,
            });
        }
        if let Some((index, value)) = spacing
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite() || **value <= 0.0)
        {
            return Err(MedkitCoreError::InvalidSpacing {
                index,
                value: *value,
            });
        }
        Ok(Self {
            shape,
            spacing,
            origin,
            direction,
            coordinate_system,
        })
    }

    /// Creates geometry with identity direction and zero origin.
    pub fn identity(
        shape: Shape,
        spacing: impl Into<Vec<f64>>,
        coordinate_system: CoordinateSystem,
    ) -> Result<Self> {
        let rank = shape.rank();
        let mut direction = vec![0.0; rank * rank];
        for index in 0..rank {
            direction[index * rank + index] = 1.0;
        }
        Self::new(
            shape,
            spacing,
            vec![0.0; rank],
            direction,
            coordinate_system,
        )
    }

    /// Returns the image shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns voxel spacing.
    pub fn spacing(&self) -> &[f64] {
        &self.spacing
    }

    /// Returns physical origin.
    pub fn origin(&self) -> &[f64] {
        &self.origin
    }

    /// Returns row-major direction matrix values.
    pub fn direction(&self) -> &[f64] {
        &self.direction
    }

    /// Returns the coordinate system.
    pub fn coordinate_system(&self) -> &CoordinateSystem {
        &self.coordinate_system
    }

    /// Returns a row-major homogeneous affine matrix.
    pub fn affine(&self) -> Vec<f64> {
        let rank = self.shape.rank();
        let width = rank + 1;
        let mut affine = vec![0.0; width * width];
        for row in 0..rank {
            for col in 0..rank {
                affine[row * width + col] = self.direction[row * rank + col] * self.spacing[col];
            }
            affine[row * width + rank] = self.origin[row];
        }
        affine[rank * width + rank] = 1.0;
        affine
    }
}
