use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    resample::{resample_f32, resample_u16},
    BoundingBox3, Interpolation, LazyTransformGraph, Result, TransformError, TransformOp, Volume3D,
    VolumeGeometry,
};

/// Transform plan parsed from TOML and used as a lazy preprocessing graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransformPlan {
    /// Human-readable plan name.
    pub name: String,
    /// Ordered lazy graph operations.
    pub operations: Vec<TransformOp>,
    /// Interpolation policy for scalar images.
    pub image_interpolation: Interpolation,
    /// Interpolation policy for labels.
    pub label_interpolation: Interpolation,
}

impl TransformPlan {
    /// Parses a transform plan from TOML.
    pub fn from_toml_str(input: &str) -> Result<Self> {
        toml::from_str(input).map_err(|error| TransformError::PlanParse {
            message: error.to_string(),
        })
    }

    /// Returns a default CT segmentation plan.
    pub fn ct_segmentation_default() -> Self {
        Self {
            name: "ct-segmentation".to_string(),
            operations: vec![
                TransformOp::Resample {
                    spacing: [1.0, 1.0, 1.0],
                },
                TransformOp::CtWindow {
                    min: -1000.0,
                    max: 1000.0,
                },
                TransformOp::MinMaxNormalize {
                    output_min: 0.0,
                    output_max: 1.0,
                },
                TransformOp::CropForeground { margin: 2 },
                TransformOp::PadCrop { size: [32, 32, 32] },
            ],
            image_interpolation: Interpolation::Linear,
            label_interpolation: Interpolation::Nearest,
        }
    }

    /// Returns this plan as a lazy transform graph.
    pub fn lazy_graph(&self) -> LazyTransformGraph {
        LazyTransformGraph {
            operations: self.operations.clone(),
            image_interpolation: self.image_interpolation,
            label_interpolation: self.label_interpolation,
        }
    }

    /// Serializes the plan to stable JSON for hashing and cache provenance.
    pub fn canonical_json(&self) -> Result<String> {
        Ok(serde_json::to_string(self).expect("transform plans contain only serializable fields"))
    }

    /// Returns a content hash for the transform plan.
    pub fn plan_hash(&self) -> Result<String> {
        let mut hasher = Sha256::new();
        hasher.update(self.canonical_json()?.as_bytes());
        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Applies deterministic preprocessing to an image/label pair.
    pub fn apply_pair(&self, image: Volume3D<f32>, label: Volume3D<u16>) -> Result<PreparedPair> {
        if image.shape != label.shape {
            return Err(TransformError::ShapeMismatch {
                image: image.shape,
                label: label.shape,
            });
        }
        let geometry = VolumeGeometry::identity(image.shape, [1.0, 1.0, 1.0])?;
        self.apply_pair_with_geometry(image, label, geometry)
    }

    /// Applies deterministic preprocessing with an explicit physical geometry.
    pub fn apply_pair_with_geometry(
        &self,
        mut image: Volume3D<f32>,
        mut label: Volume3D<u16>,
        mut geometry: VolumeGeometry,
    ) -> Result<PreparedPair> {
        if image.shape != label.shape {
            return Err(TransformError::ShapeMismatch {
                image: image.shape,
                label: label.shape,
            });
        }
        if image.shape != geometry.shape {
            return Err(TransformError::GeometryShapeMismatch {
                volume: image.shape,
                geometry: geometry.shape,
            });
        }
        if self.label_interpolation != Interpolation::Nearest
            && self
                .operations
                .iter()
                .any(|operation| matches!(operation, TransformOp::Resample { .. }))
        {
            return Err(TransformError::InvalidLabelInterpolation {
                reason: "segmentation labels must use nearest-neighbor interpolation for resample operations".to_string(),
            });
        }
        let mut crop_origin = [0_usize; 3];
        let mut applied = Vec::new();
        for operation in &self.operations {
            match *operation {
                TransformOp::CtWindow { min, max } => {
                    validate_window(min, max)?;
                    ensure_finite_intensities(&image)?;
                    for value in &mut image.data {
                        *value = value.clamp(min, max);
                    }
                    applied.push("ct_window".to_string());
                }
                TransformOp::MinMaxNormalize {
                    output_min,
                    output_max,
                } => {
                    min_max_normalize(&mut image, output_min, output_max)?;
                    applied.push("min_max_normalize".to_string());
                }
                TransformOp::ZScoreNormalize { epsilon } => {
                    z_score_normalize(&mut image, epsilon)?;
                    applied.push("z_score_normalize".to_string());
                }
                TransformOp::PercentileClip { lower, upper } => {
                    percentile_clip(&mut image, lower, upper)?;
                    applied.push("percentile_clip".to_string());
                }
                TransformOp::DatasetMeanStdNormalize { mean, std } => {
                    dataset_mean_std_normalize(&mut image, mean, std)?;
                    applied.push("dataset_mean_std_normalize".to_string());
                }
                TransformOp::CropForeground { margin } => {
                    if let Some(bbox) = foreground_bbox(&label, margin) {
                        crop_origin = [
                            crop_origin[0] + bbox.start[0],
                            crop_origin[1] + bbox.start[1],
                            crop_origin[2] + bbox.start[2],
                        ];
                        image = image.crop(bbox)?;
                        label = label.crop(bbox)?;
                        geometry = geometry.crop(bbox)?;
                    }
                    applied.push("crop_foreground".to_string());
                }
                TransformOp::PadCrop { size } => {
                    image = image.pad_crop_center(size, 0.0)?;
                    label = label.pad_crop_center(size, 0)?;
                    geometry = geometry.pad_crop_center(size)?;
                    applied.push("pad_crop".to_string());
                }
                TransformOp::Resample { spacing } => {
                    let target_geometry = geometry.resampled_to_spacing(spacing)?;
                    let next_image = resample_f32(
                        &image,
                        &geometry,
                        &target_geometry,
                        self.image_interpolation,
                    );
                    let next_label = resample_u16(
                        &label,
                        &geometry,
                        &target_geometry,
                        self.label_interpolation,
                    );
                    image = next_image?;
                    label = next_label?;
                    geometry = target_geometry;
                    applied.push("resample".to_string());
                }
            }
        }
        Ok(PreparedPair {
            image,
            label,
            geometry,
            crop_origin,
            applied_operations: applied,
        })
    }
}

/// Result of applying deterministic preprocessing to an image/label pair.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedPair {
    /// Preprocessed scalar image.
    pub image: Volume3D<f32>,
    /// Preprocessed label map.
    pub label: Volume3D<u16>,
    /// Physical geometry after preprocessing.
    pub geometry: VolumeGeometry,
    /// Origin of the foreground crop in the voxel frame where the crop ran.
    pub crop_origin: [usize; 3],
    /// Applied operation names.
    pub applied_operations: Vec<String>,
}

fn validate_window(min: f32, max: f32) -> Result<()> {
    if !min.is_finite() || !max.is_finite() || min > max {
        return Err(TransformError::InvalidIntensityTransform {
            reason: format!("ct_window requires finite min <= max, got min={min} max={max}"),
        });
    }
    Ok(())
}

fn min_max_normalize(image: &mut Volume3D<f32>, output_min: f32, output_max: f32) -> Result<()> {
    if !output_min.is_finite() || !output_max.is_finite() || output_min >= output_max {
        return Err(TransformError::InvalidIntensityTransform {
            reason: format!(
                "min_max_normalize requires finite output_min < output_max, got output_min={output_min} output_max={output_max}"
            ),
        });
    }
    let (min, max) = finite_min_max(image)?;
    let range = max - min;
    if range <= f32::EPSILON {
        image.data.fill(output_min);
        return Ok(());
    }
    let output_range = output_max - output_min;
    for value in &mut image.data {
        let unit = (*value - min) / range;
        *value = output_min + unit * output_range;
    }
    Ok(())
}

fn z_score_normalize(image: &mut Volume3D<f32>, epsilon: f32) -> Result<()> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(TransformError::InvalidIntensityTransform {
            reason: format!("z_score_normalize requires positive finite epsilon, got {epsilon}"),
        });
    }
    ensure_finite_intensities(image)?;
    let mean = image
        .data
        .iter()
        .map(|value| f64::from(*value))
        .sum::<f64>()
        / image.data.len() as f64;
    let variance = image
        .data
        .iter()
        .map(|value| {
            let delta = f64::from(*value) - mean;
            delta * delta
        })
        .sum::<f64>()
        / image.data.len() as f64;
    let std = (variance.sqrt() as f32).max(epsilon);
    let mean = mean as f32;
    for value in &mut image.data {
        *value = (*value - mean) / std;
    }
    Ok(())
}

fn percentile_clip(image: &mut Volume3D<f32>, lower: f32, upper: f32) -> Result<()> {
    if !lower.is_finite()
        || !upper.is_finite()
        || !(0.0..=100.0).contains(&lower)
        || !(0.0..=100.0).contains(&upper)
        || lower > upper
    {
        return Err(TransformError::InvalidIntensityTransform {
            reason: format!(
                "percentile_clip requires 0 <= lower <= upper <= 100, got lower={lower} upper={upper}"
            ),
        });
    }
    ensure_finite_intensities(image)?;
    let mut sorted = image.data.clone();
    sorted.sort_by(f32::total_cmp);
    let low = percentile_value(&sorted, lower);
    let high = percentile_value(&sorted, upper);
    for value in &mut image.data {
        *value = value.clamp(low, high);
    }
    Ok(())
}

fn dataset_mean_std_normalize(image: &mut Volume3D<f32>, mean: f32, std: f32) -> Result<()> {
    if !mean.is_finite() || !std.is_finite() || std <= 0.0 {
        return Err(TransformError::InvalidIntensityTransform {
            reason: format!(
                "dataset_mean_std_normalize requires finite mean and positive finite std, got mean={mean} std={std}"
            ),
        });
    }
    ensure_finite_intensities(image)?;
    for value in &mut image.data {
        *value = (*value - mean) / std;
    }
    Ok(())
}

fn finite_min_max(image: &Volume3D<f32>) -> Result<(f32, f32)> {
    ensure_finite_intensities(image)?;
    Ok(image
        .data
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), value| {
            (min.min(*value), max.max(*value))
        }))
}

fn ensure_finite_intensities(image: &Volume3D<f32>) -> Result<()> {
    if image.data.iter().any(|value| !value.is_finite()) {
        return Err(TransformError::InvalidIntensityTransform {
            reason: "image intensities must be finite".to_string(),
        });
    }
    Ok(())
}

fn percentile_value(sorted: &[f32], percentile: f32) -> f32 {
    let position = f64::from(percentile) / 100.0 * (sorted.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    if low == high {
        return sorted[low];
    }
    let weight = (position - low as f64) as f32;
    sorted[low] * (1.0 - weight) + sorted[high] * weight
}

fn foreground_bbox(label: &Volume3D<u16>, margin: usize) -> Option<BoundingBox3> {
    let mut start = [usize::MAX; 3];
    let mut end = [0_usize; 3];
    for z in 0..label.shape[2] {
        for y in 0..label.shape[1] {
            for x in 0..label.shape[0] {
                if *label.get(x, y, z) == 0 {
                    continue;
                }
                start[0] = start[0].min(x);
                start[1] = start[1].min(y);
                start[2] = start[2].min(z);
                end[0] = end[0].max(x + 1);
                end[1] = end[1].max(y + 1);
                end[2] = end[2].max(z + 1);
            }
        }
    }
    if start[0] == usize::MAX {
        return None;
    }
    for axis in 0..3 {
        start[axis] = start[axis].saturating_sub(margin);
        end[axis] = (end[axis] + margin).min(label.shape[axis]);
    }
    Some(BoundingBox3::new(start, end))
}
