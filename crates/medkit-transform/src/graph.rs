use serde::{Deserialize, Serialize};

/// Interpolation policy used for image or label transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Interpolation {
    /// Nearest-neighbor interpolation, appropriate for labels.
    Nearest,
    /// Linear interpolation, appropriate for scalar images.
    Linear,
}

/// A single lazy preprocessing operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "op")]
pub enum TransformOp {
    /// Clamp CT intensities into a window.
    CtWindow {
        /// Minimum window value.
        min: f32,
        /// Maximum window value.
        max: f32,
    },
    /// Normalize image intensities after deterministic preprocessing.
    Normalize {
        /// Output mean.
        mean: f32,
        /// Output standard deviation.
        std: f32,
    },
    /// Crop image and label to the foreground label bounding box.
    CropForeground {
        /// Margin in voxels around the foreground box.
        margin: usize,
    },
    /// Center pad or crop to a fixed size.
    PadCrop {
        /// Output size in x, y, z order.
        size: [usize; 3],
    },
    /// Geometry-aware resampling to target spacing.
    Resample {
        /// Target spacing in x, y, z order.
        spacing: [f64; 3],
    },
}

/// Lazy transform graph: ordered operations plus image/label interpolation policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LazyTransformGraph {
    /// Ordered lazy operations.
    pub operations: Vec<TransformOp>,
    /// Interpolation policy for scalar image data.
    pub image_interpolation: Interpolation,
    /// Interpolation policy for label data.
    pub label_interpolation: Interpolation,
}

impl LazyTransformGraph {
    /// Returns a stable operation summary.
    pub fn operation_names(&self) -> Vec<&'static str> {
        self.operations
            .iter()
            .map(|operation| match operation {
                TransformOp::CtWindow { .. } => "ct_window",
                TransformOp::Normalize { .. } => "normalize",
                TransformOp::CropForeground { .. } => "crop_foreground",
                TransformOp::PadCrop { .. } => "pad_crop",
                TransformOp::Resample { .. } => "resample",
            })
            .collect()
    }
}
