use std::{
    fs,
    path::{Path, PathBuf},
};

use medkit_dataset::{CaseStatus, DatasetManifest};
use medkit_transform::{TransformPlan, Volume3D, VolumeGeometry};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{nifti_pixels, CacheError, Result};

const CACHE_MANIFEST: &str = "cache_manifest.json";

/// Configuration for deterministic cache preparation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrepareConfig {
    /// Dataset root provided by the user.
    pub dataset_root: PathBuf,
    /// Validation manifest path.
    pub manifest_path: PathBuf,
    /// Transform plan TOML path.
    pub plan_path: PathBuf,
    /// Cache output directory.
    pub cache_dir: PathBuf,
    /// Optional chunk shape for patch-friendly fixed-size chunk files.
    pub chunk_shape: Option<[usize; 3]>,
}

/// Machine-readable cache manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheManifest {
    /// Cache format version.
    pub version: u32,
    /// Cache directory.
    pub cache_dir: String,
    /// Source validation manifest path.
    pub dataset_manifest_path: String,
    /// Transform plan hash.
    pub transform_plan_hash: String,
    /// Transform plan.
    pub transform_plan: TransformPlan,
    /// Aggregate cache summary.
    pub summary: CacheSummary,
    /// Cached cases.
    pub cases: Vec<CachedCase>,
}

/// Aggregate cache preparation summary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheSummary {
    /// Number of valid manifest cases considered.
    pub input_cases: usize,
    /// Number of cases cached successfully.
    pub cached_cases: usize,
    /// Number of failed cases.
    pub failed_cases: usize,
    /// Total foreground voxels after preprocessing.
    pub foreground_voxels: usize,
    /// Bytes written for image and label cache files.
    pub bytes_written: usize,
}

/// Metadata for one cached case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CachedCase {
    /// Case id.
    pub case_id: String,
    /// Content-addressed cache key.
    pub cache_key: String,
    /// Hash of source paths and metadata.
    pub source_metadata_hash: String,
    /// Hash of the transform plan.
    pub transform_plan_hash: String,
    /// Source image path.
    pub image_path: String,
    /// Source label path.
    pub label_path: String,
    /// Source image geometry.
    pub source_geometry: VolumeGeometry,
    /// Cached output geometry.
    pub output_geometry: VolumeGeometry,
    /// Cached image path.
    pub image_cache_path: String,
    /// Cached label path.
    pub label_cache_path: String,
    /// Cached foreground index path with little-endian u64 flat indices.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub foreground_indices_path: Option<String>,
    /// Cached foreground integral-volume path with little-endian u32 values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub foreground_prefix_path: Option<String>,
    /// Foreground prefix shape in x, y, z order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub foreground_prefix_shape: Option<[usize; 3]>,
    /// Optional fixed-size chunked image path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_chunk_cache_path: Option<String>,
    /// Optional fixed-size chunked label path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label_chunk_cache_path: Option<String>,
    /// Optional fixed-size chunk grid in x, y, z order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_grid: Option<[usize; 3]>,
    /// Cached shape in x, y, z order.
    pub shape: [usize; 3],
    /// Binary chunk shape in x, y, z order.
    pub chunk_shape: [usize; 3],
    /// Foreground crop origin in source voxel coordinates.
    pub crop_origin: [usize; 3],
    /// Applied deterministic transform operations.
    pub applied_operations: Vec<String>,
    /// Count of non-zero label voxels after preprocessing.
    pub foreground_voxels: usize,
    /// Bytes written for this case.
    pub bytes_written: usize,
}

/// Prepares a deterministic content-addressed cache.
pub fn prepare_cache(config: &PrepareConfig) -> Result<CacheManifest> {
    let manifest = read_dataset_manifest(&config.manifest_path)?;
    let plan_text = fs::read_to_string(&config.plan_path)
        .map_err(|source| CacheError::io(&config.plan_path, source))?;
    let plan = TransformPlan::from_toml_str(&plan_text)?;
    let plan_hash = plan.plan_hash()?;
    fs::create_dir_all(&config.cache_dir)
        .map_err(|source| CacheError::io(&config.cache_dir, source))?;

    let mut summary = CacheSummary::default();
    let mut cases = Vec::new();
    for case in manifest
        .cases
        .iter()
        .filter(|case| case.status == CaseStatus::Valid)
    {
        summary.input_cases += 1;
        match prepare_case(
            case,
            &plan,
            &plan_hash,
            &config.cache_dir,
            config.chunk_shape,
        ) {
            Ok(cached) => {
                summary.cached_cases += 1;
                summary.foreground_voxels += cached.foreground_voxels;
                summary.bytes_written += cached.bytes_written;
                cases.push(cached);
            }
            Err(_) => summary.failed_cases += 1,
        }
    }

    let cache_manifest = CacheManifest {
        version: 1,
        cache_dir: config.cache_dir.to_string_lossy().into_owned(),
        dataset_manifest_path: config.manifest_path.to_string_lossy().into_owned(),
        transform_plan_hash: plan_hash,
        transform_plan: plan,
        summary,
        cases,
    };
    write_cache_manifest(&cache_manifest, &config.cache_dir.join(CACHE_MANIFEST))?;
    Ok(cache_manifest)
}

/// Reads a cache manifest from a cache directory.
pub fn read_cache_manifest(cache_dir: impl AsRef<Path>) -> Result<CacheManifest> {
    let path = cache_dir.as_ref().join(CACHE_MANIFEST);
    let text = fs::read_to_string(&path).map_err(|source| CacheError::io(&path, source))?;
    serde_json::from_str(&text).map_err(|source| CacheError::json(path, source))
}

fn read_dataset_manifest(path: &Path) -> Result<DatasetManifest> {
    let text = fs::read_to_string(path).map_err(|source| CacheError::io(path, source))?;
    serde_json::from_str(&text).map_err(|source| CacheError::json(path, source))
}

fn write_cache_manifest(manifest: &CacheManifest, path: &Path) -> Result<()> {
    let text =
        serde_json::to_string_pretty(manifest).map_err(|source| CacheError::json(path, source))?;
    fs::write(path, text).map_err(|source| CacheError::io(path, source))
}

fn prepare_case(
    case: &medkit_dataset::CaseManifest,
    plan: &TransformPlan,
    plan_hash: &str,
    cache_dir: &Path,
    chunk_shape: Option<[usize; 3]>,
) -> Result<CachedCase> {
    let image_path = case.image_path.as_ref().ok_or_else(|| {
        CacheError::invalid_input(format!("case {} has no image path", case.case_id))
    })?;
    let label_path = case.label_path.as_ref().ok_or_else(|| {
        CacheError::invalid_input(format!("case {} has no label path", case.case_id))
    })?;
    let image = nifti_pixels::load_image_f32(Path::new(image_path))?;
    let label = nifti_pixels::load_label_u16(Path::new(label_path))?;
    if !image.geometry.approximately_eq(&label.geometry, 1e-6) {
        return Err(CacheError::invalid_input(format!(
            "case {} image and label source geometry differ",
            case.case_id
        )));
    }
    let source_geometry = image.geometry;
    let label_geometry = label.geometry;
    let prepared = plan.apply_pair_with_geometry(image.volume, label.volume, source_geometry)?;
    let foreground_indices = prepared
        .label
        .data
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value != 0).then_some(index))
        .collect::<Vec<_>>();
    let foreground_voxels = foreground_indices.len();
    let source_metadata_hash = source_hash(case, &source_geometry, &label_geometry)?;
    let cache_key = cache_key(&case.case_id, &source_metadata_hash, plan_hash);
    let case_dir = cache_dir.join(&cache_key);
    fs::create_dir_all(&case_dir).map_err(|source| CacheError::io(&case_dir, source))?;
    let image_cache_path = case_dir.join("image.f32.raw");
    let label_cache_path = case_dir.join("label.u16.raw");
    let foreground_indices_path = case_dir.join("foreground_indices.u64.raw");
    let foreground_prefix_path = case_dir.join("foreground_prefix.u32.raw");
    write_f32_volume(&prepared.image, &image_cache_path)?;
    write_u16_volume(&prepared.label, &label_cache_path)?;
    write_u64_indices(&foreground_indices, &foreground_indices_path)?;
    let foreground_prefix_shape = [
        prepared.label.shape[0] + 1,
        prepared.label.shape[1] + 1,
        prepared.label.shape[2] + 1,
    ];
    let foreground_prefix = foreground_prefix_values(&prepared.label);
    write_u32_values(&foreground_prefix, &foreground_prefix_path)?;
    let chunk_shape = chunk_shape.map(|shape| valid_chunk_shape(shape, prepared.image.shape));
    let chunk_paths = if let Some(chunk_shape) = chunk_shape {
        let image_chunk_cache_path = case_dir.join("image.chunks.f32.raw");
        let label_chunk_cache_path = case_dir.join("label.chunks.u16.raw");
        write_chunked_volume_f32(&prepared.image, chunk_shape, &image_chunk_cache_path)?;
        write_chunked_volume_u16(&prepared.label, chunk_shape, &label_chunk_cache_path)?;
        let chunk_grid = chunk_grid(prepared.image.shape, chunk_shape);
        Some((
            chunk_shape,
            chunk_grid,
            image_chunk_cache_path,
            label_chunk_cache_path,
        ))
    } else {
        None
    };
    let mut bytes_written = prepared.image.data.len() * 4
        + prepared.label.data.len() * 2
        + foreground_indices.len() * 8
        + foreground_prefix.len() * 4;
    if let Some((chunk_shape, chunk_grid, _, _)) = &chunk_paths {
        let chunk_voxels = chunk_shape[0] * chunk_shape[1] * chunk_shape[2];
        let chunks = chunk_grid[0] * chunk_grid[1] * chunk_grid[2];
        bytes_written += chunks * chunk_voxels * (4 + 2);
    }
    let (chunk_shape_out, chunk_grid, image_chunk_cache_path, label_chunk_cache_path) =
        if let Some((chunk_shape, chunk_grid, image_path, label_path)) = chunk_paths {
            (
                chunk_shape,
                Some(chunk_grid),
                Some(image_path.to_string_lossy().into_owned()),
                Some(label_path.to_string_lossy().into_owned()),
            )
        } else {
            (prepared.image.shape, None, None, None)
        };
    let cached = CachedCase {
        case_id: case.case_id.clone(),
        cache_key,
        source_metadata_hash,
        transform_plan_hash: plan_hash.to_string(),
        image_path: image_path.clone(),
        label_path: label_path.clone(),
        source_geometry,
        output_geometry: prepared.geometry,
        image_cache_path: image_cache_path.to_string_lossy().into_owned(),
        label_cache_path: label_cache_path.to_string_lossy().into_owned(),
        foreground_indices_path: Some(foreground_indices_path.to_string_lossy().into_owned()),
        foreground_prefix_path: Some(foreground_prefix_path.to_string_lossy().into_owned()),
        foreground_prefix_shape: Some(foreground_prefix_shape),
        image_chunk_cache_path,
        label_chunk_cache_path,
        shape: prepared.image.shape,
        chunk_shape: chunk_shape_out,
        chunk_grid,
        crop_origin: prepared.crop_origin,
        applied_operations: prepared.applied_operations,
        foreground_voxels,
        bytes_written,
    };
    let case_json = case_dir.join("case.json");
    let text = serde_json::to_string_pretty(&cached)
        .map_err(|source| CacheError::json(&case_json, source))?;
    fs::write(&case_json, text).map_err(|source| CacheError::io(case_json, source))?;
    Ok(cached)
}

fn write_f32_volume(volume: &Volume3D<f32>, path: &Path) -> Result<()> {
    let mut bytes = Vec::with_capacity(volume.data.len() * 4);
    for value in &volume.data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).map_err(|source| CacheError::io(path, source))
}

fn write_u16_volume(volume: &Volume3D<u16>, path: &Path) -> Result<()> {
    let mut bytes = Vec::with_capacity(volume.data.len() * 2);
    for value in &volume.data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).map_err(|source| CacheError::io(path, source))
}

fn write_u64_indices(indices: &[usize], path: &Path) -> Result<()> {
    let mut bytes = Vec::with_capacity(indices.len() * 8);
    for value in indices {
        bytes.extend_from_slice(&(*value as u64).to_le_bytes());
    }
    fs::write(path, bytes).map_err(|source| CacheError::io(path, source))
}

fn write_u32_values(values: &[u32], path: &Path) -> Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).map_err(|source| CacheError::io(path, source))
}

fn foreground_prefix_values(label: &Volume3D<u16>) -> Vec<u32> {
    let shape = label.shape;
    let prefix_shape = [shape[0] + 1, shape[1] + 1, shape[2] + 1];
    let mut values = vec![0_u32; prefix_shape[0] * prefix_shape[1] * prefix_shape[2]];
    let prefix_y_stride = prefix_shape[0];
    let prefix_z_stride = prefix_shape[0] * prefix_shape[1];
    let label_y_stride = shape[0];
    let label_z_stride = shape[0] * shape[1];
    for z in 1..prefix_shape[2] {
        let prefix_z_base = z * prefix_z_stride;
        let previous_prefix_z_base = (z - 1) * prefix_z_stride;
        let label_z_base = (z - 1) * label_z_stride;
        for y in 1..prefix_shape[1] {
            let mut row_sum = 0_u32;
            let row_base = prefix_z_base + y * prefix_y_stride;
            let above_base = prefix_z_base + (y - 1) * prefix_y_stride;
            let behind_base = previous_prefix_z_base + y * prefix_y_stride;
            let above_behind_base = previous_prefix_z_base + (y - 1) * prefix_y_stride;
            let label_row_base = label_z_base + (y - 1) * label_y_stride;
            for x in 1..prefix_shape[0] {
                row_sum += u32::from(label.data[label_row_base + x - 1] != 0);
                values[row_base + x] = row_sum + values[above_base + x] + values[behind_base + x]
                    - values[above_behind_base + x];
            }
        }
    }
    values
}

fn valid_chunk_shape(requested: [usize; 3], volume_shape: [usize; 3]) -> [usize; 3] {
    [
        requested[0].max(1).min(volume_shape[0]),
        requested[1].max(1).min(volume_shape[1]),
        requested[2].max(1).min(volume_shape[2]),
    ]
}

fn chunk_grid(shape: [usize; 3], chunk_shape: [usize; 3]) -> [usize; 3] {
    [
        shape[0].div_ceil(chunk_shape[0]),
        shape[1].div_ceil(chunk_shape[1]),
        shape[2].div_ceil(chunk_shape[2]),
    ]
}

fn write_chunked_volume_f32(
    volume: &Volume3D<f32>,
    chunk_shape: [usize; 3],
    path: &Path,
) -> Result<()> {
    let mut bytes = Vec::with_capacity(chunked_value_count(volume.shape, chunk_shape) * 4);
    write_chunked_values(
        volume,
        chunk_shape,
        0.0_f32,
        |value, out| {
            out.extend_from_slice(&value.to_le_bytes());
        },
        &mut bytes,
    );
    fs::write(path, bytes).map_err(|source| CacheError::io(path, source))
}

fn write_chunked_volume_u16(
    volume: &Volume3D<u16>,
    chunk_shape: [usize; 3],
    path: &Path,
) -> Result<()> {
    let mut bytes = Vec::with_capacity(chunked_value_count(volume.shape, chunk_shape) * 2);
    write_chunked_values(
        volume,
        chunk_shape,
        0_u16,
        |value, out| {
            out.extend_from_slice(&value.to_le_bytes());
        },
        &mut bytes,
    );
    fs::write(path, bytes).map_err(|source| CacheError::io(path, source))
}

fn write_chunked_values<T: Copy>(
    volume: &Volume3D<T>,
    chunk_shape: [usize; 3],
    fill: T,
    mut write: impl FnMut(T, &mut Vec<u8>),
    out: &mut Vec<u8>,
) {
    let grid = chunk_grid(volume.shape, chunk_shape);
    for chunk_z in 0..grid[2] {
        for chunk_y in 0..grid[1] {
            for chunk_x in 0..grid[0] {
                let start = [
                    chunk_x * chunk_shape[0],
                    chunk_y * chunk_shape[1],
                    chunk_z * chunk_shape[2],
                ];
                for local_z in 0..chunk_shape[2] {
                    let z = start[2] + local_z;
                    for local_y in 0..chunk_shape[1] {
                        let y = start[1] + local_y;
                        for local_x in 0..chunk_shape[0] {
                            let x = start[0] + local_x;
                            let value = if x < volume.shape[0]
                                && y < volume.shape[1]
                                && z < volume.shape[2]
                            {
                                *volume.get(x, y, z)
                            } else {
                                fill
                            };
                            write(value, out);
                        }
                    }
                }
            }
        }
    }
}

fn chunked_value_count(shape: [usize; 3], chunk_shape: [usize; 3]) -> usize {
    let grid = chunk_grid(shape, chunk_shape);
    grid[0] * grid[1] * grid[2] * chunk_shape[0] * chunk_shape[1] * chunk_shape[2]
}

fn source_hash(
    case: &medkit_dataset::CaseManifest,
    image_geometry: &VolumeGeometry,
    label_geometry: &VolumeGeometry,
) -> Result<String> {
    let text = serde_json::to_string(&(case, image_geometry, label_geometry))
        .map_err(|source| CacheError::json(PathBuf::from("<case-source-hash>"), source))?;
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

fn cache_key(case_id: &str, source_hash: &str, plan_hash: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(case_id.as_bytes());
    hasher.update(source_hash.as_bytes());
    hasher.update(plan_hash.as_bytes());
    format!("{}-{:x}", case_id, hasher.finalize())
}
