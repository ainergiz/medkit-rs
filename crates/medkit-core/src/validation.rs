use crate::{MedkitCoreError, Result, SpatialGeometry};

/// Tolerances used when comparing physical geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometryTolerance {
    /// Absolute spacing tolerance.
    pub spacing: f64,
    /// Absolute origin tolerance.
    pub origin: f64,
    /// Absolute direction-cosine tolerance.
    pub direction: f64,
}

impl GeometryTolerance {
    /// Creates a geometry tolerance.
    pub fn new(spacing: f64, origin: f64, direction: f64) -> Result<Self> {
        validate_tolerance("spacing", spacing)?;
        validate_tolerance("origin", origin)?;
        validate_tolerance("direction", direction)?;
        Ok(Self {
            spacing,
            origin,
            direction,
        })
    }
}

impl Default for GeometryTolerance {
    fn default() -> Self {
        Self {
            spacing: 1e-5,
            origin: 1e-4,
            direction: 1e-6,
        }
    }
}

fn validate_tolerance(field: &'static str, value: f64) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(MedkitCoreError::InvalidTolerance { field, value });
    }
    Ok(())
}

/// A single reason two geometries are not compatible.
#[derive(Debug, Clone, PartialEq)]
pub enum GeometryMismatch {
    /// Shape values differ.
    Shape {
        /// Left shape dimensions.
        left: Vec<usize>,
        /// Right shape dimensions.
        right: Vec<usize>,
    },
    /// Coordinate system differs.
    CoordinateSystem {
        /// Left coordinate system.
        left: String,
        /// Right coordinate system.
        right: String,
    },
    /// Spacing differs beyond tolerance.
    Spacing {
        /// Axis index.
        index: usize,
        /// Left value.
        left: f64,
        /// Right value.
        right: f64,
    },
    /// Origin differs beyond tolerance.
    Origin {
        /// Axis index.
        index: usize,
        /// Left value.
        left: f64,
        /// Right value.
        right: f64,
    },
    /// Direction value differs beyond tolerance.
    Direction {
        /// Row-major value index.
        index: usize,
        /// Left value.
        left: f64,
        /// Right value.
        right: f64,
    },
}

/// Compatibility report for two spatial geometries.
#[derive(Debug, Clone, PartialEq)]
pub struct GeometryCompatibilityReport {
    mismatches: Vec<GeometryMismatch>,
}

impl GeometryCompatibilityReport {
    /// Returns true if no incompatibilities were found.
    pub fn is_compatible(&self) -> bool {
        self.mismatches.is_empty()
    }

    /// Returns all mismatches.
    pub fn mismatches(&self) -> &[GeometryMismatch] {
        &self.mismatches
    }
}

/// Geometry compatibility checks.
pub trait GeometryCompatibility {
    /// Compares two geometries with explicit tolerances.
    fn compatibility_with(
        &self,
        other: &SpatialGeometry,
        tolerance: GeometryTolerance,
    ) -> GeometryCompatibilityReport;

    /// Returns true if two geometries are compatible under default tolerances.
    fn is_compatible_with(&self, other: &SpatialGeometry) -> bool {
        self.compatibility_with(other, GeometryTolerance::default())
            .is_compatible()
    }
}

impl GeometryCompatibility for SpatialGeometry {
    fn compatibility_with(
        &self,
        other: &SpatialGeometry,
        tolerance: GeometryTolerance,
    ) -> GeometryCompatibilityReport {
        let mut mismatches = Vec::new();
        if self.shape().as_slice() != other.shape().as_slice() {
            mismatches.push(GeometryMismatch::Shape {
                left: self.shape().as_slice().to_vec(),
                right: other.shape().as_slice().to_vec(),
            });
        }
        if self.coordinate_system() != other.coordinate_system() {
            mismatches.push(GeometryMismatch::CoordinateSystem {
                left: format!("{:?}", self.coordinate_system()),
                right: format!("{:?}", other.coordinate_system()),
            });
        }
        compare_values(
            self.spacing(),
            other.spacing(),
            tolerance.spacing,
            |index, left, right| GeometryMismatch::Spacing { index, left, right },
            &mut mismatches,
        );
        compare_values(
            self.origin(),
            other.origin(),
            tolerance.origin,
            |index, left, right| GeometryMismatch::Origin { index, left, right },
            &mut mismatches,
        );
        compare_values(
            self.direction(),
            other.direction(),
            tolerance.direction,
            |index, left, right| GeometryMismatch::Direction { index, left, right },
            &mut mismatches,
        );
        GeometryCompatibilityReport { mismatches }
    }
}

fn compare_values(
    left: &[f64],
    right: &[f64],
    tolerance: f64,
    mismatch: impl Fn(usize, f64, f64) -> GeometryMismatch,
    mismatches: &mut Vec<GeometryMismatch>,
) {
    for (index, (left_value, right_value)) in left.iter().zip(right).enumerate() {
        if (left_value - right_value).abs() > tolerance {
            mismatches.push(mismatch(index, *left_value, *right_value));
        }
    }
}
