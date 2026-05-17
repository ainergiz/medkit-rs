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
    /// Linearly maps the per-volume intensity range into an output range.
    MinMaxNormalize {
        /// Output value for the minimum input intensity.
        #[serde(default = "default_min_max_output_min")]
        output_min: f32,
        /// Output value for the maximum input intensity.
        #[serde(default = "default_min_max_output_max")]
        output_max: f32,
    },
    /// Normalizes each volume with its own mean and standard deviation.
    ZScoreNormalize {
        /// Minimum standard deviation used for nearly constant images.
        #[serde(default = "default_normalize_epsilon")]
        epsilon: f32,
    },
    /// Clips intensities to per-volume percentile bounds.
    PercentileClip {
        /// Lower percentile in `[0, 100]`.
        lower: f32,
        /// Upper percentile in `[0, 100]`.
        upper: f32,
    },
    /// Normalizes with externally computed dataset-level mean and std.
    DatasetMeanStdNormalize {
        /// Dataset-level mean.
        mean: f32,
        /// Dataset-level standard deviation.
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
                TransformOp::MinMaxNormalize { .. } => "min_max_normalize",
                TransformOp::ZScoreNormalize { .. } => "z_score_normalize",
                TransformOp::PercentileClip { .. } => "percentile_clip",
                TransformOp::DatasetMeanStdNormalize { .. } => "dataset_mean_std_normalize",
                TransformOp::CropForeground { .. } => "crop_foreground",
                TransformOp::PadCrop { .. } => "pad_crop",
                TransformOp::Resample { .. } => "resample",
            })
            .collect()
    }
}

fn default_min_max_output_min() -> f32 {
    0.0
}

fn default_min_max_output_max() -> f32 {
    1.0
}

fn default_normalize_epsilon() -> f32 {
    1.0e-6
}
