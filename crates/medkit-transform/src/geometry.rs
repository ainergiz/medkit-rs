use serde::{Deserialize, Serialize};

use crate::{BoundingBox3, Result, TransformError};

/// Physical geometry for a dense 3D voxel grid.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VolumeGeometry {
    /// Shape in x, y, z order.
    pub shape: [usize; 3],
    /// Voxel spacing in physical units for x, y, z axes.
    pub spacing: [f64; 3],
    /// Physical coordinate of voxel index `[0, 0, 0]`.
    pub origin: [f64; 3],
    /// Row-major direction cosine matrix.
    pub direction: [[f64; 3]; 3],
}

impl VolumeGeometry {
    /// Creates geometry with explicit shape, spacing, origin, and direction.
    pub fn new(
        shape: [usize; 3],
        spacing: [f64; 3],
        origin: [f64; 3],
        direction: [[f64; 3]; 3],
    ) -> Result<Self> {
        if shape.contains(&0) {
            return Err(TransformError::InvalidSize { size: shape });
        }
        validate_spacing(spacing)?;
        if origin.iter().any(|value| !value.is_finite()) {
            return Err(TransformError::InvalidOrigin { origin });
        }
        if direction.iter().flatten().any(|value| !value.is_finite()) {
            return Err(TransformError::InvalidDirection {
                determinant: f64::NAN,
            });
        }
        let determinant = determinant3(direction);
        if !determinant.is_finite() || determinant.abs() <= 1e-12 {
            return Err(TransformError::InvalidDirection { determinant });
        }
        Ok(Self {
            shape,
            spacing,
            origin,
            direction,
        })
    }

    /// Creates identity-direction geometry with zero origin.
    pub fn identity(shape: [usize; 3], spacing: [f64; 3]) -> Result<Self> {
        Self::new(
            shape,
            spacing,
            [0.0, 0.0, 0.0],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
    }

    /// Returns true when two geometries match within an absolute tolerance.
    pub fn approximately_eq(&self, other: &Self, tolerance: f64) -> bool {
        self.shape == other.shape
            && all_close(&self.spacing, &other.spacing, tolerance)
            && all_close(&self.origin, &other.origin, tolerance)
            && self
                .direction
                .iter()
                .flatten()
                .zip(other.direction.iter().flatten())
                .all(|(left, right)| (*left - *right).abs() <= tolerance)
    }

    /// Converts continuous voxel indices to physical coordinates.
    pub fn voxel_to_world(&self, index: [f64; 3]) -> [f64; 3] {
        let matrix = self.voxel_matrix();
        [
            self.origin[0]
                + matrix[0][0] * index[0]
                + matrix[0][1] * index[1]
                + matrix[0][2] * index[2],
            self.origin[1]
                + matrix[1][0] * index[0]
                + matrix[1][1] * index[1]
                + matrix[1][2] * index[2],
            self.origin[2]
                + matrix[2][0] * index[0]
                + matrix[2][1] * index[1]
                + matrix[2][2] * index[2],
        ]
    }

    /// Converts physical coordinates to continuous voxel indices.
    pub fn world_to_voxel(&self, world: [f64; 3]) -> Result<[f64; 3]> {
        let inverse = self.inverse_voxel_matrix()?;
        let delta = [
            world[0] - self.origin[0],
            world[1] - self.origin[1],
            world[2] - self.origin[2],
        ];
        Ok(mat_vec_mul(inverse, delta))
    }

    /// Returns geometry after cropping by a half-open bounding box.
    pub fn crop(&self, bbox: BoundingBox3) -> Result<Self> {
        Self::new(
            bbox.size(),
            self.spacing,
            self.voxel_to_world([
                bbox.start[0] as f64,
                bbox.start[1] as f64,
                bbox.start[2] as f64,
            ]),
            self.direction,
        )
    }

    /// Returns geometry after center pad/crop to `target`.
    pub fn pad_crop_center(&self, target: [usize; 3]) -> Result<Self> {
        if target.contains(&0) {
            return Err(TransformError::InvalidSize { size: target });
        }
        let mut src_start = [0_usize; 3];
        let mut dst_start = [0_usize; 3];
        for axis in 0..3 {
            if self.shape[axis] > target[axis] {
                src_start[axis] = (self.shape[axis] - target[axis]) / 2;
            } else {
                dst_start[axis] = (target[axis] - self.shape[axis]) / 2;
            }
        }
        let output_zero_in_source = [
            src_start[0] as f64 - dst_start[0] as f64,
            src_start[1] as f64 - dst_start[1] as f64,
            src_start[2] as f64 - dst_start[2] as f64,
        ];
        Self::new(
            target,
            self.spacing,
            self.voxel_to_world(output_zero_in_source),
            self.direction,
        )
    }

    /// Returns geometry resampled to a target spacing while preserving endpoints.
    pub fn resampled_to_spacing(&self, spacing: [f64; 3]) -> Result<Self> {
        validate_spacing(spacing)?;
        let mut shape = [1_usize; 3];
        for axis in 0..3 {
            if self.shape[axis] == 1 {
                shape[axis] = 1;
                continue;
            }
            let extent = (self.shape[axis] - 1) as f64 * self.spacing[axis];
            shape[axis] = (extent / spacing[axis]).round().max(0.0) as usize + 1;
        }
        Self::new(shape, spacing, self.origin, self.direction)
    }

    pub(crate) fn voxel_matrix(&self) -> [[f64; 3]; 3] {
        [
            [
                self.direction[0][0] * self.spacing[0],
                self.direction[0][1] * self.spacing[1],
                self.direction[0][2] * self.spacing[2],
            ],
            [
                self.direction[1][0] * self.spacing[0],
                self.direction[1][1] * self.spacing[1],
                self.direction[1][2] * self.spacing[2],
            ],
            [
                self.direction[2][0] * self.spacing[0],
                self.direction[2][1] * self.spacing[1],
                self.direction[2][2] * self.spacing[2],
            ],
        ]
    }

    pub(crate) fn inverse_voxel_matrix(&self) -> Result<[[f64; 3]; 3]> {
        invert3(self.voxel_matrix())
    }
}

fn validate_spacing(spacing: [f64; 3]) -> Result<()> {
    if spacing
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(TransformError::InvalidSpacing { spacing });
    }
    Ok(())
}

fn all_close(left: &[f64; 3], right: &[f64; 3], tolerance: f64) -> bool {
    left.iter()
        .zip(right)
        .all(|(left, right)| (*left - *right).abs() <= tolerance)
}

pub(crate) fn mat_vec_mul(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}

pub(crate) fn mat_mul(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = left[row][0] * right[0][col]
                + left[row][1] * right[1][col]
                + left[row][2] * right[2][col];
        }
    }
    out
}

fn determinant3(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn invert3(matrix: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3]> {
    let determinant = determinant3(matrix);
    if !determinant.is_finite() || determinant.abs() <= 1e-12 {
        return Err(TransformError::InvalidDirection { determinant });
    }
    let inv_det = 1.0 / determinant;
    Ok([
        [
            (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * inv_det,
            (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * inv_det,
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * inv_det,
        ],
        [
            (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * inv_det,
            (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * inv_det,
            (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * inv_det,
        ],
        [
            (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * inv_det,
            (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) * inv_det,
            (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * inv_det,
        ],
    ])
}
