use std::{
    fs::{self, File},
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use medkit_dataset::{CaseStatus, DatasetManifest};
use medkit_transform::{TransformPlan, Volume3D, VolumeGeometry};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{nifti_pixels, CacheError, Result};

const CACHE_MANIFEST: &str = "cache_manifest.json";
const CACHE_SCHEMA_VERSION: u32 = 1;
const CACHE_WRITER_VERSION: &str = env!("CARGO_PKG_VERSION");

fn default_image_channel_count() -> usize {
    1
}

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
    /// Hash of source paths, source bytes, and metadata.
    pub source_metadata_hash: String,
    /// Hash of the transform plan.
    pub transform_plan_hash: String,
    /// Source image path.
    pub image_path: String,
    /// Source image channel paths in cache channel order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub image_paths: Vec<String>,
    /// Number of source image channels stacked into the cached image artifact.
    #[serde(default = "default_image_channel_count")]
    pub image_channel_count: usize,
    /// Source label path.
    pub label_path: String,
    /// Source image geometry.
    pub source_geometry: VolumeGeometry,
    /// Cached output geometry.
    pub output_geometry: VolumeGeometry,
    /// Cached image path.
    pub image_cache_path: String,
    /// SHA-256 of the cached image artifact.
    #[serde(default)]
    pub image_cache_sha256: String,
    /// Cached label path.
    pub label_cache_path: String,
    /// SHA-256 of the cached label artifact.
    #[serde(default)]
    pub label_cache_sha256: String,
    /// Cached foreground index path with little-endian u64 flat indices.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub foreground_indices_path: Option<String>,
    /// SHA-256 of the foreground index artifact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub foreground_indices_sha256: Option<String>,
    /// Cached foreground integral-volume path with little-endian u32 values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub foreground_prefix_path: Option<String>,
    /// SHA-256 of the foreground prefix artifact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub foreground_prefix_sha256: Option<String>,
    /// Foreground prefix shape in x, y, z order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub foreground_prefix_shape: Option<[usize; 3]>,
    /// Optional fixed-size chunked image path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_chunk_cache_path: Option<String>,
    /// SHA-256 of the optional fixed-size chunked image artifact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_chunk_cache_sha256: Option<String>,
    /// Optional fixed-size chunked label path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label_chunk_cache_path: Option<String>,
    /// SHA-256 of the optional fixed-size chunked label artifact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label_chunk_cache_sha256: Option<String>,
    /// Optional fixed-size chunk grid in x, y, z order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_grid: Option<[usize; 3]>,
    /// Cached shape in x, y, z order.
    pub shape: [usize; 3],
    /// Binary chunk shape in x, y, z order.
    pub chunk_shape: [usize; 3],
    /// Foreground crop origin in the voxel frame where the crop operation ran.
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
    let staging_dir = staging_dir(&config.cache_dir);
    fs::create_dir_all(&staging_dir).map_err(|source| CacheError::io(&staging_dir, source))?;

    let result = prepare_cache_in_staging(config, manifest, plan, plan_hash, &staging_dir);
    match result {
        Ok(cache_manifest) => {
            if let Err(error) =
                promote_staged_cache(&staging_dir, &config.cache_dir, &cache_manifest)
            {
                let _ = cleanup_staging(&staging_dir);
                Err(error)
            } else {
                Ok(cache_manifest)
            }
        }
        Err(error) => {
            cleanup_staging(&staging_dir)?;
            Err(error)
        }
    }
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
    let mut file = File::create(path).map_err(|source| CacheError::io(path, source))?;
    file.write_all(text.as_bytes())
        .map_err(|source| CacheError::io(path, source))?;
    file.sync_all()
        .map_err(|source| CacheError::io(path, source))
}

fn prepare_cache_in_staging(
    config: &PrepareConfig,
    manifest: DatasetManifest,
    plan: TransformPlan,
    plan_hash: String,
    staging_dir: &Path,
) -> Result<CacheManifest> {
    let mut summary = CacheSummary::default();
    let mut cases = Vec::new();
    for case in manifest
        .cases
        .iter()
        .filter(|case| case.status == CaseStatus::Valid)
    {
        summary.input_cases += 1;
        let cached = prepare_case(
            case,
            &config.dataset_root,
            &plan,
            &plan_hash,
            staging_dir,
            &config.cache_dir,
            config.chunk_shape,
        )
        .map_err(|error| {
            CacheError::invalid_input(format!("failed to prepare case {}: {error}", case.case_id))
        })?;
        summary.cached_cases += 1;
        summary.foreground_voxels += cached.foreground_voxels;
        summary.bytes_written += cached.bytes_written;
        cases.push(cached);
    }

    let cache_manifest = CacheManifest {
        version: CACHE_SCHEMA_VERSION,
        cache_dir: config.cache_dir.to_string_lossy().into_owned(),
        dataset_manifest_path: config.manifest_path.to_string_lossy().into_owned(),
        transform_plan_hash: plan_hash,
        transform_plan: plan,
        summary,
        cases,
    };
    write_cache_manifest(&cache_manifest, &staging_dir.join(CACHE_MANIFEST))?;
    Ok(cache_manifest)
}

fn staging_dir(cache_dir: &Path) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    cache_dir
        .join(".staging")
        .join(format!("prepare-{}-{nanos}", std::process::id()))
}

fn cleanup_staging(staging_dir: &Path) -> Result<()> {
    match fs::remove_dir_all(staging_dir) {
        Ok(()) => {
            if let Some(parent) = staging_dir.parent() {
                let _ = fs::remove_dir(parent);
            }
            Ok(())
        }
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(CacheError::io(staging_dir, source)),
    }
}

fn promote_staged_cache(
    staging_dir: &Path,
    cache_dir: &Path,
    manifest: &CacheManifest,
) -> Result<()> {
    for case in &manifest.cases {
        let staged_case_dir = staging_dir.join(&case.cache_key);
        let final_case_dir = cache_dir.join(&case.cache_key);
        if final_case_dir.exists() {
            if !final_case_dir.is_dir() {
                return Err(CacheError::invalid_input(format!(
                    "cache case path exists but is not a directory: {}",
                    final_case_dir.display()
                )));
            }
            match existing_case_status(&final_case_dir, case)? {
                ExistingCaseStatus::Valid => cleanup_staging(&staged_case_dir)?,
                ExistingCaseStatus::Corrupt => {
                    replace_case_dir(&staged_case_dir, &final_case_dir)?;
                }
            }
        } else {
            fs::rename(&staged_case_dir, &final_case_dir)
                .map_err(|source| CacheError::io(&final_case_dir, source))?;
        }
    }
    promote_cache_manifest(staging_dir, cache_dir)?;
    cleanup_staging(staging_dir)
}

enum ExistingCaseStatus {
    Valid,
    Corrupt,
}

fn existing_case_status(final_case_dir: &Path, case: &CachedCase) -> Result<ExistingCaseStatus> {
    let case_json = final_case_dir.join("case.json");
    let text = match fs::read_to_string(&case_json) {
        Ok(text) => text,
        Err(_) => return Ok(ExistingCaseStatus::Corrupt),
    };
    let existing: CachedCase = match serde_json::from_str(&text) {
        Ok(existing) => existing,
        Err(_) => return Ok(ExistingCaseStatus::Corrupt),
    };
    if existing != *case {
        return Err(CacheError::invalid_input(format!(
            "existing cache case {} at {} does not match staged metadata; remove the case directory or rebuild into a clean cache",
            case.case_id,
            final_case_dir.display()
        )));
    }
    if validate_existing_case_artifacts(case).is_ok() {
        Ok(ExistingCaseStatus::Valid)
    } else {
        Ok(ExistingCaseStatus::Corrupt)
    }
}

fn replace_case_dir(staged_case_dir: &Path, final_case_dir: &Path) -> Result<()> {
    let backup_dir = unique_sibling_path(final_case_dir, "replace-case");
    if backup_dir.exists() {
        fs::remove_dir_all(&backup_dir).map_err(|source| CacheError::io(&backup_dir, source))?;
    }
    fs::rename(final_case_dir, &backup_dir)
        .map_err(|source| CacheError::io(&backup_dir, source))?;
    match fs::rename(staged_case_dir, final_case_dir) {
        Ok(()) => {
            fs::remove_dir_all(&backup_dir)
                .map_err(|source| CacheError::io(&backup_dir, source))?;
            sync_parent_dir(final_case_dir);
            Ok(())
        }
        Err(source) => {
            let _ = fs::rename(&backup_dir, final_case_dir);
            Err(CacheError::io(final_case_dir, source))
        }
    }
}

fn promote_cache_manifest(staging_dir: &Path, cache_dir: &Path) -> Result<()> {
    let staged_manifest = staging_dir.join(CACHE_MANIFEST);
    let final_manifest = cache_dir.join(CACHE_MANIFEST);
    replace_manifest_file(&staged_manifest, &final_manifest)?;
    sync_parent_dir(&final_manifest);
    Ok(())
}

#[cfg(unix)]
fn replace_manifest_file(staged_manifest: &Path, final_manifest: &Path) -> Result<()> {
    fs::rename(staged_manifest, final_manifest)
        .map_err(|source| CacheError::io(final_manifest, source))
}

#[cfg(not(unix))]
fn replace_manifest_file(staged_manifest: &Path, final_manifest: &Path) -> Result<()> {
    if !final_manifest.exists() {
        return fs::rename(staged_manifest, final_manifest)
            .map_err(|source| CacheError::io(final_manifest, source));
    }
    let backup_manifest = unique_sibling_path(final_manifest, "replace-manifest");
    if backup_manifest.exists() {
        fs::remove_file(&backup_manifest)
            .map_err(|source| CacheError::io(&backup_manifest, source))?;
    }
    fs::rename(final_manifest, &backup_manifest)
        .map_err(|source| CacheError::io(&backup_manifest, source))?;
    match fs::rename(staged_manifest, final_manifest) {
        Ok(()) => {
            fs::remove_file(&backup_manifest)
                .map_err(|source| CacheError::io(&backup_manifest, source))?;
            Ok(())
        }
        Err(source) => {
            let _ = fs::rename(&backup_manifest, final_manifest);
            Err(CacheError::io(final_manifest, source))
        }
    }
}

fn validate_existing_case_artifacts(case: &CachedCase) -> std::result::Result<(), String> {
    let mut errors = Vec::new();
    check_artifact_hash(
        &case.image_cache_path,
        Some(case.image_cache_sha256.as_str()),
        "image cache",
        &mut errors,
    );
    check_artifact_hash(
        &case.label_cache_path,
        Some(case.label_cache_sha256.as_str()),
        "label cache",
        &mut errors,
    );
    if let Some(path) = &case.foreground_indices_path {
        check_artifact_hash(
            path,
            case.foreground_indices_sha256.as_deref(),
            "foreground indices",
            &mut errors,
        );
    }
    if let Some(path) = &case.foreground_prefix_path {
        check_artifact_hash(
            path,
            case.foreground_prefix_sha256.as_deref(),
            "foreground prefix",
            &mut errors,
        );
    }
    if let Some(path) = &case.image_chunk_cache_path {
        check_artifact_hash(
            path,
            case.image_chunk_cache_sha256.as_deref(),
            "image chunk cache",
            &mut errors,
        );
    }
    if let Some(path) = &case.label_chunk_cache_path {
        check_artifact_hash(
            path,
            case.label_chunk_cache_sha256.as_deref(),
            "label chunk cache",
            &mut errors,
        );
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

fn check_artifact_hash(path: &str, expected: Option<&str>, kind: &str, errors: &mut Vec<String>) {
    let Some(expected) = expected.filter(|value| !value.is_empty()) else {
        errors.push(format!("{kind} is missing SHA-256 metadata"));
        return;
    };
    match sha256_file(Path::new(path)) {
        Ok(actual) if actual == expected => {}
        Ok(actual) => errors.push(format!("{kind} has SHA-256 {actual}, expected {expected}")),
        Err(error) => errors.push(format!("{kind} could not be hashed: {error}")),
    }
}

fn unique_sibling_path(path: &Path, prefix: &str) -> PathBuf {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact");
    parent.join(format!(
        ".{prefix}-{name}-{}-{}",
        std::process::id(),
        unique_nanos()
    ))
}

fn unique_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos())
}

#[cfg(unix)]
fn sync_parent_dir(path: &Path) {
    if let Some(parent) = path.parent() {
        if let Ok(dir) = File::open(parent) {
            let _ = dir.sync_all();
        }
    }
}

#[cfg(not(unix))]
fn sync_parent_dir(_path: &Path) {}

#[derive(Debug)]
struct SourceImage {
    path: PathBuf,
    path_text: String,
    channel_index: Option<u16>,
}

#[derive(Debug)]
struct LoadedImageChannel {
    path_text: String,
    channel_index: Option<u16>,
    volume: Volume3D<f32>,
    geometry: VolumeGeometry,
    source_content_hash: String,
}

fn prepare_case(
    case: &medkit_dataset::CaseManifest,
    dataset_root: &Path,
    plan: &TransformPlan,
    plan_hash: &str,
    staging_cache_dir: &Path,
    final_cache_dir: &Path,
    chunk_shape: Option<[usize; 3]>,
) -> Result<CachedCase> {
    let image_sources = image_sources_for_case(case, dataset_root)?;
    let label_path = case.label_path.as_ref().ok_or_else(|| {
        CacheError::invalid_input(format!("case {} has no label path", case.case_id))
    })?;
    let label_path = resolve_manifest_source_path(dataset_root, label_path);
    let label_path_text = label_path.to_string_lossy().into_owned();
    let label = nifti_pixels::load_label_u16(&label_path)?;
    let mut image_channels = Vec::with_capacity(image_sources.len());
    for source in image_sources {
        let image = nifti_pixels::load_image_f32(&source.path)?;
        image_channels.push(LoadedImageChannel {
            path_text: source.path_text,
            channel_index: source.channel_index,
            volume: image.volume,
            geometry: image.geometry,
            source_content_hash: image.source_content_hash,
        });
    }
    let source_geometry = image_channels
        .first()
        .map(|channel| channel.geometry)
        .ok_or_else(|| {
            CacheError::invalid_input(format!("case {} has no image channels", case.case_id))
        })?;
    for channel in &image_channels {
        if !channel.geometry.approximately_eq(&source_geometry, 1e-6) {
            return Err(CacheError::invalid_input(format!(
                "case {} image channel {} geometry differs from the first image channel",
                case.case_id, channel.path_text
            )));
        }
        if !channel.geometry.approximately_eq(&label.geometry, 1e-6) {
            return Err(CacheError::invalid_input(format!(
                "case {} image and label source geometry differ for channel {}",
                case.case_id, channel.path_text
            )));
        }
    }
    let label_geometry = label.geometry;
    let image_path_texts = image_channels
        .iter()
        .map(|channel| channel.path_text.clone())
        .collect::<Vec<_>>();
    let image_channel_indices = image_channels
        .iter()
        .map(|channel| channel.channel_index)
        .collect::<Vec<_>>();
    let image_content_hashes = image_channels
        .iter()
        .map(|channel| channel.source_content_hash.clone())
        .collect::<Vec<_>>();
    let image_volumes = image_channels
        .into_iter()
        .map(|channel| channel.volume)
        .collect::<Vec<_>>();
    let prepared =
        plan.apply_channels_with_geometry(image_volumes, label.volume, source_geometry)?;
    let image_channel_count = prepared.images.len();
    let foreground_indices = prepared
        .label
        .data
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value != 0).then_some(index))
        .collect::<Vec<_>>();
    let foreground_voxels = foreground_indices.len();
    let source_metadata_hash = source_hash(
        case,
        &image_path_texts,
        &image_channel_indices,
        &source_geometry,
        &label_geometry,
        &image_content_hashes,
        &label.source_content_hash,
    )?;
    let effective_chunk_shape =
        chunk_shape.map(|shape| valid_chunk_shape(shape, prepared.label.shape));
    let storage_layout = if effective_chunk_shape.is_some() {
        "chunked"
    } else {
        "resident"
    };
    let cache_key = cache_key(
        &case.case_id,
        &source_metadata_hash,
        plan_hash,
        CACHE_SCHEMA_VERSION,
        storage_layout,
        effective_chunk_shape,
        CACHE_WRITER_VERSION,
    );
    let staging_case_dir = staging_cache_dir.join(&cache_key);
    let final_case_dir = final_cache_dir.join(&cache_key);
    fs::create_dir_all(&staging_case_dir)
        .map_err(|source| CacheError::io(&staging_case_dir, source))?;
    let staging_image_cache_path = staging_case_dir.join("image.f32.raw");
    let staging_label_cache_path = staging_case_dir.join("label.u16.raw");
    let staging_foreground_indices_path = staging_case_dir.join("foreground_indices.u64.raw");
    let staging_foreground_prefix_path = staging_case_dir.join("foreground_prefix.u32.raw");
    let image_cache_path = final_case_dir.join("image.f32.raw");
    let label_cache_path = final_case_dir.join("label.u16.raw");
    let foreground_indices_path = final_case_dir.join("foreground_indices.u64.raw");
    let foreground_prefix_path = final_case_dir.join("foreground_prefix.u32.raw");
    write_f32_channels(&prepared.images, &staging_image_cache_path)?;
    write_u16_volume(&prepared.label, &staging_label_cache_path)?;
    write_u64_indices(&foreground_indices, &staging_foreground_indices_path)?;
    let image_cache_sha256 = sha256_file(&staging_image_cache_path)?;
    let label_cache_sha256 = sha256_file(&staging_label_cache_path)?;
    let foreground_indices_sha256 = sha256_file(&staging_foreground_indices_path)?;
    let foreground_prefix_shape = checked_prefix_shape(prepared.label.shape)?;
    let foreground_prefix = foreground_prefix_values(&prepared.label)?;
    write_u32_values(&foreground_prefix, &staging_foreground_prefix_path)?;
    let foreground_prefix_sha256 = sha256_file(&staging_foreground_prefix_path)?;
    let chunk_paths = if let Some(chunk_shape) = effective_chunk_shape {
        let staging_image_chunk_cache_path = staging_case_dir.join("image.chunks.f32.raw");
        let staging_label_chunk_cache_path = staging_case_dir.join("label.chunks.u16.raw");
        let image_chunk_cache_path = final_case_dir.join("image.chunks.f32.raw");
        let label_chunk_cache_path = final_case_dir.join("label.chunks.u16.raw");
        write_chunked_channels_f32(
            &prepared.images,
            chunk_shape,
            &staging_image_chunk_cache_path,
        )?;
        write_chunked_volume_u16(
            &prepared.label,
            chunk_shape,
            &staging_label_chunk_cache_path,
        )?;
        let image_chunk_cache_sha256 = sha256_file(&staging_image_chunk_cache_path)?;
        let label_chunk_cache_sha256 = sha256_file(&staging_label_chunk_cache_path)?;
        let chunk_grid = chunk_grid(prepared.label.shape, chunk_shape);
        Some((
            chunk_shape,
            chunk_grid,
            image_chunk_cache_path,
            label_chunk_cache_path,
            image_chunk_cache_sha256,
            label_chunk_cache_sha256,
        ))
    } else {
        None
    };
    let image_value_count = prepared.images.iter().try_fold(0_usize, |sum, image| {
        checked_add(sum, image.data.len(), "image value count")
    })?;
    let mut bytes_written = checked_add(
        checked_add(
            checked_mul(image_value_count, 4, "image cache byte count")?,
            checked_mul(prepared.label.data.len(), 2, "label cache byte count")?,
            "resident cache byte count",
        )?,
        checked_add(
            checked_mul(foreground_indices.len(), 8, "foreground index byte count")?,
            checked_mul(foreground_prefix.len(), 4, "foreground prefix byte count")?,
            "foreground byte count",
        )?,
        "case byte count",
    )?;
    if let Some((chunk_shape, chunk_grid, _, _, _, _)) = &chunk_paths {
        let chunk_voxels = checked_value_count(*chunk_shape, "chunk voxel count")?;
        let chunks = checked_value_count(*chunk_grid, "chunk grid count")?;
        let image_chunk_bytes_per_voxel =
            checked_mul(image_channel_count, 4, "image channel chunk byte count")?;
        let chunk_bytes_per_voxel =
            checked_add(image_chunk_bytes_per_voxel, 2, "chunk byte count per voxel")?;
        let chunk_bytes = checked_mul(
            checked_mul(chunks, chunk_voxels, "chunked value count")?,
            chunk_bytes_per_voxel,
            "chunked cache byte count",
        )?;
        bytes_written = checked_add(bytes_written, chunk_bytes, "case byte count")?;
    }
    let (
        chunk_shape_out,
        chunk_grid,
        image_chunk_cache_path,
        label_chunk_cache_path,
        image_chunk_cache_sha256,
        label_chunk_cache_sha256,
    ) = if let Some((chunk_shape, chunk_grid, image_path, label_path, image_hash, label_hash)) =
        chunk_paths
    {
        (
            chunk_shape,
            Some(chunk_grid),
            Some(image_path.to_string_lossy().into_owned()),
            Some(label_path.to_string_lossy().into_owned()),
            Some(image_hash),
            Some(label_hash),
        )
    } else {
        (prepared.label.shape, None, None, None, None, None)
    };
    let cached = CachedCase {
        case_id: case.case_id.clone(),
        cache_key,
        source_metadata_hash,
        transform_plan_hash: plan_hash.to_string(),
        image_path: image_path_texts[0].clone(),
        image_paths: image_path_texts,
        image_channel_count,
        label_path: label_path_text,
        source_geometry,
        output_geometry: prepared.geometry,
        image_cache_path: image_cache_path.to_string_lossy().into_owned(),
        image_cache_sha256,
        label_cache_path: label_cache_path.to_string_lossy().into_owned(),
        label_cache_sha256,
        foreground_indices_path: Some(foreground_indices_path.to_string_lossy().into_owned()),
        foreground_indices_sha256: Some(foreground_indices_sha256),
        foreground_prefix_path: Some(foreground_prefix_path.to_string_lossy().into_owned()),
        foreground_prefix_sha256: Some(foreground_prefix_sha256),
        foreground_prefix_shape: Some(foreground_prefix_shape),
        image_chunk_cache_path,
        image_chunk_cache_sha256,
        label_chunk_cache_path,
        label_chunk_cache_sha256,
        shape: prepared.label.shape,
        chunk_shape: chunk_shape_out,
        chunk_grid,
        crop_origin: prepared.crop_origin,
        applied_operations: prepared.applied_operations,
        foreground_voxels,
        bytes_written,
    };
    let case_json = staging_case_dir.join("case.json");
    let text = serde_json::to_string_pretty(&cached)
        .map_err(|source| CacheError::json(&case_json, source))?;
    fs::write(&case_json, text).map_err(|source| CacheError::io(case_json, source))?;
    Ok(cached)
}

fn image_sources_for_case(
    case: &medkit_dataset::CaseManifest,
    dataset_root: &Path,
) -> Result<Vec<SourceImage>> {
    let mut sources = if !case.images.is_empty() {
        case.images
            .iter()
            .map(|image| {
                let path = resolve_manifest_source_path(dataset_root, &image.path);
                SourceImage {
                    path_text: path.to_string_lossy().into_owned(),
                    path,
                    channel_index: image.channel_index,
                }
            })
            .collect::<Vec<_>>()
    } else {
        let image_path = case.image_path.as_ref().ok_or_else(|| {
            CacheError::invalid_input(format!("case {} has no image path", case.case_id))
        })?;
        let path = resolve_manifest_source_path(dataset_root, image_path);
        vec![SourceImage {
            path_text: path.to_string_lossy().into_owned(),
            path,
            channel_index: None,
        }]
    };
    if sources.is_empty() {
        return Err(CacheError::invalid_input(format!(
            "case {} has no image channels",
            case.case_id
        )));
    }
    let indexed_channels = sources
        .iter()
        .filter(|source| source.channel_index.is_some())
        .count();
    if indexed_channels != 0 && indexed_channels != sources.len() {
        return Err(CacheError::invalid_input(format!(
            "case {} mixes indexed and unindexed image channels",
            case.case_id
        )));
    }
    if indexed_channels == sources.len() {
        sources.sort_by(|left, right| {
            left.channel_index
                .cmp(&right.channel_index)
                .then_with(|| left.path_text.cmp(&right.path_text))
        });
        for (expected, source) in sources.iter().enumerate() {
            let expected = u16::try_from(expected).map_err(|_| {
                CacheError::invalid_input(format!(
                    "case {} has too many indexed image channels",
                    case.case_id
                ))
            })?;
            if source.channel_index != Some(expected) {
                return Err(CacheError::invalid_input(format!(
                    "case {} image channel indices must be contiguous from 0; expected channel {expected}, found {:?}",
                    case.case_id, source.channel_index
                )));
            }
        }
    } else {
        sources.sort_by(|left, right| left.path_text.cmp(&right.path_text));
    }
    Ok(sources)
}

fn resolve_manifest_source_path(dataset_root: &Path, path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        dataset_root.join(path)
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(|source| CacheError::io(path, source))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| CacheError::io(path, source))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn write_u16_volume(volume: &Volume3D<u16>, path: &Path) -> Result<()> {
    let mut writer =
        BufWriter::new(File::create(path).map_err(|source| CacheError::io(path, source))?);
    for value in &volume.data {
        writer
            .write_all(&value.to_le_bytes())
            .map_err(|source| CacheError::io(path, source))?;
    }
    writer
        .flush()
        .map_err(|source| CacheError::io(path, source))
}

fn write_f32_channels(channels: &[Volume3D<f32>], path: &Path) -> Result<()> {
    let mut writer =
        BufWriter::new(File::create(path).map_err(|source| CacheError::io(path, source))?);
    for channel in channels {
        for value in &channel.data {
            writer
                .write_all(&value.to_le_bytes())
                .map_err(|source| CacheError::io(path, source))?;
        }
    }
    writer
        .flush()
        .map_err(|source| CacheError::io(path, source))
}

fn write_u64_indices(indices: &[usize], path: &Path) -> Result<()> {
    let mut writer =
        BufWriter::new(File::create(path).map_err(|source| CacheError::io(path, source))?);
    for value in indices {
        writer
            .write_all(&(*value as u64).to_le_bytes())
            .map_err(|source| CacheError::io(path, source))?;
    }
    writer
        .flush()
        .map_err(|source| CacheError::io(path, source))
}

fn write_u32_values(values: &[u32], path: &Path) -> Result<()> {
    let mut writer =
        BufWriter::new(File::create(path).map_err(|source| CacheError::io(path, source))?);
    for value in values {
        writer
            .write_all(&value.to_le_bytes())
            .map_err(|source| CacheError::io(path, source))?;
    }
    writer
        .flush()
        .map_err(|source| CacheError::io(path, source))
}

fn foreground_prefix_values(label: &Volume3D<u16>) -> Result<Vec<u32>> {
    let shape = label.shape;
    let prefix_shape = checked_prefix_shape(shape)?;
    let mut values = vec![0_u32; checked_value_count(prefix_shape, "foreground prefix values")?];
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
                row_sum = row_sum
                    .checked_add(u32::from(label.data[label_row_base + x - 1] != 0))
                    .ok_or_else(|| {
                        CacheError::invalid_input("foreground prefix count overflowed u32")
                    })?;
                values[row_base + x] = row_sum
                    .checked_add(values[above_base + x])
                    .and_then(|value| value.checked_add(values[behind_base + x]))
                    .and_then(|value| value.checked_sub(values[above_behind_base + x]))
                    .ok_or_else(|| {
                        CacheError::invalid_input("foreground prefix count overflowed u32")
                    })?;
            }
        }
    }
    Ok(values)
}

fn checked_prefix_shape(shape: [usize; 3]) -> Result<[usize; 3]> {
    Ok([
        shape[0]
            .checked_add(1)
            .ok_or_else(|| CacheError::invalid_input("foreground prefix x dimension overflow"))?,
        shape[1]
            .checked_add(1)
            .ok_or_else(|| CacheError::invalid_input("foreground prefix y dimension overflow"))?,
        shape[2]
            .checked_add(1)
            .ok_or_else(|| CacheError::invalid_input("foreground prefix z dimension overflow"))?,
    ])
}

fn checked_value_count(shape: [usize; 3], what: &str) -> Result<usize> {
    shape[0]
        .checked_mul(shape[1])
        .and_then(|value| value.checked_mul(shape[2]))
        .ok_or_else(|| CacheError::invalid_input(format!("{what} overflow for shape {shape:?}")))
}

fn checked_mul(left: usize, right: usize, what: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| CacheError::invalid_input(format!("{what} overflow")))
}

fn checked_add(left: usize, right: usize, what: &str) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| CacheError::invalid_input(format!("{what} overflow")))
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

fn write_chunked_channels_f32(
    channels: &[Volume3D<f32>],
    chunk_shape: [usize; 3],
    path: &Path,
) -> Result<()> {
    let mut writer =
        BufWriter::new(File::create(path).map_err(|source| CacheError::io(path, source))?);
    for channel in channels {
        write_chunked_values(
            channel,
            chunk_shape,
            0.0_f32,
            |value, writer| {
                writer.write_all(&value.to_le_bytes())?;
                Ok(())
            },
            &mut writer,
        )
        .map_err(|source| CacheError::io(path, source))?;
    }
    writer
        .flush()
        .map_err(|source| CacheError::io(path, source))
}

fn write_chunked_volume_u16(
    volume: &Volume3D<u16>,
    chunk_shape: [usize; 3],
    path: &Path,
) -> Result<()> {
    let mut writer =
        BufWriter::new(File::create(path).map_err(|source| CacheError::io(path, source))?);
    write_chunked_values(
        volume,
        chunk_shape,
        0_u16,
        |value, writer| {
            writer.write_all(&value.to_le_bytes())?;
            Ok(())
        },
        &mut writer,
    )
    .map_err(|source| CacheError::io(path, source))?;
    writer
        .flush()
        .map_err(|source| CacheError::io(path, source))
}

fn write_chunked_values<T: Copy>(
    volume: &Volume3D<T>,
    chunk_shape: [usize; 3],
    fill: T,
    mut write: impl FnMut(T, &mut dyn Write) -> std::io::Result<()>,
    out: &mut dyn Write,
) -> std::io::Result<()> {
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
                            write(value, out)?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
fn chunked_value_count(shape: [usize; 3], chunk_shape: [usize; 3]) -> usize {
    let grid = chunk_grid(shape, chunk_shape);
    grid[0] * grid[1] * grid[2] * chunk_shape[0] * chunk_shape[1] * chunk_shape[2]
}

fn source_hash(
    case: &medkit_dataset::CaseManifest,
    image_paths: &[String],
    image_channel_indices: &[Option<u16>],
    image_geometry: &VolumeGeometry,
    label_geometry: &VolumeGeometry,
    image_content_hashes: &[String],
    label_content_hash: &str,
) -> Result<String> {
    let text = serde_json::to_string(&(
        case,
        image_paths,
        image_channel_indices,
        image_geometry,
        label_geometry,
        image_content_hashes,
        label_content_hash,
    ))
    .map_err(|source| CacheError::json(PathBuf::from("<case-source-hash>"), source))?;
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

fn cache_key(
    case_id: &str,
    source_hash: &str,
    plan_hash: &str,
    cache_schema_version: u32,
    storage_layout: &str,
    effective_chunk_shape: Option<[usize; 3]>,
    writer_version: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(case_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(source_hash.as_bytes());
    hasher.update(b"\0");
    hasher.update(plan_hash.as_bytes());
    hasher.update(b"\0");
    hasher.update(cache_schema_version.to_le_bytes());
    hasher.update(b"\0");
    hasher.update(storage_layout.as_bytes());
    hasher.update(b"\0");
    match effective_chunk_shape {
        Some(shape) => {
            hasher.update(b"chunked");
            for value in shape {
                hasher.update(value.to_le_bytes());
            }
        }
        None => hasher.update(b"resident"),
    }
    hasher.update(b"\0");
    hasher.update(writer_version.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    use medkit_dataset::{
        CaseImage, CaseManifest, CaseStatus, DatasetLayout, DatasetManifest, ValidationSummary,
    };
    use medkit_transform::Volume3D;

    use super::*;

    const VOX_OFFSET: usize = 352;

    #[derive(Debug, Clone)]
    struct NiftiFixture {
        bytes: Vec<u8>,
    }

    impl NiftiFixture {
        fn new(dims: &[i16], datatype: i16, pixdim: &[f32]) -> Self {
            let mut fixture = Self {
                bytes: vec![0; VOX_OFFSET],
            };
            fixture.put_i32(0, 348);
            fixture.put_i16(40, i16::try_from(dims.len()).unwrap());
            for (index, dim) in dims.iter().enumerate() {
                fixture.put_i16(42 + index * 2, *dim);
            }
            fixture.put_i16(70, datatype);
            fixture.put_i16(72, bitpix_for(datatype));
            fixture.put_f32(76, 1.0);
            for (index, spacing) in pixdim.iter().enumerate() {
                fixture.put_f32(80 + index * 4, *spacing);
            }
            fixture.put_f32(108, VOX_OFFSET as f32);
            fixture.bytes[344..348].copy_from_slice(b"n+1\0");
            fixture
        }

        fn append_f32_pixels(mut self, values: &[f32]) -> Vec<u8> {
            for value in values {
                self.bytes.extend_from_slice(&value.to_le_bytes());
            }
            self.bytes
        }

        fn append_u16_pixels(mut self, values: &[u16]) -> Vec<u8> {
            for value in values {
                self.bytes.extend_from_slice(&value.to_le_bytes());
            }
            self.bytes
        }

        fn put_i32(&mut self, offset: usize, value: i32) {
            self.bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }

        fn put_i16(&mut self, offset: usize, value: i16) {
            self.bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
        }

        fn put_f32(&mut self, offset: usize, value: f32) {
            self.bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
    }

    #[test]
    fn prepare_cache_writes_content_addressed_case_artifacts() {
        let root = temp_dir("prepare-success");
        let image_path = root.join("case_a_0000.nii");
        let label_path = root.join("case_a.nii");
        let manifest_path = root.join("manifest.json");
        let plan_path = root.join("plan.toml");
        let cache_dir = root.join("cache");

        fs::write(
            &image_path,
            NiftiFixture::new(&[3, 2, 1], 16, &[1.0, 1.0, 1.0])
                .append_f32_pixels(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        .unwrap();
        fs::write(
            &label_path,
            NiftiFixture::new(&[3, 2, 1], 512, &[1.0, 1.0, 1.0])
                .append_u16_pixels(&[0, 1, 0, 2, 0, 3]),
        )
        .unwrap();
        write_plan(&plan_path, identity_plan());
        write_manifest(
            &manifest_path,
            &root,
            vec![
                valid_case("case_a", Some(&image_path), Some(&label_path)),
                invalid_case("skipped_invalid"),
            ],
        );

        let manifest = prepare_cache(&PrepareConfig {
            dataset_root: root.clone(),
            manifest_path: manifest_path.clone(),
            plan_path: plan_path.clone(),
            cache_dir: cache_dir.clone(),
            chunk_shape: Some([2, 2, 2]),
        })
        .unwrap();

        assert_eq!(
            manifest.summary,
            CacheSummary {
                input_cases: 1,
                cached_cases: 1,
                failed_cases: 0,
                foreground_voxels: 3,
                bytes_written: 204,
            }
        );
        assert_eq!(manifest.cases.len(), 1);
        let cached = &manifest.cases[0];
        assert_eq!(cached.case_id, "case_a");
        assert_eq!(cached.cache_key.len(), 64);
        assert_eq!(cached.source_metadata_hash.len(), 64);
        assert_eq!(cached.transform_plan_hash, manifest.transform_plan_hash);
        assert_eq!(cached.transform_plan_hash.len(), 64);
        assert_eq!(cached.image_cache_sha256.len(), 64);
        assert_eq!(cached.label_cache_sha256.len(), 64);
        assert_eq!(cached.foreground_indices_sha256.as_ref().unwrap().len(), 64);
        assert_eq!(cached.foreground_prefix_sha256.as_ref().unwrap().len(), 64);
        assert_eq!(cached.image_chunk_cache_sha256.as_ref().unwrap().len(), 64);
        assert_eq!(cached.label_chunk_cache_sha256.as_ref().unwrap().len(), 64);
        assert_eq!(cached.image_path, path_text(&image_path));
        assert_eq!(cached.label_path, path_text(&label_path));
        assert_eq!(cached.image_paths, vec![path_text(&image_path)]);
        assert_eq!(cached.image_channel_count, 1);
        assert_eq!(cached.source_geometry.spacing, [1.0, 1.0, 1.0]);
        assert!(cached
            .source_geometry
            .approximately_eq(&cached.output_geometry, 1e-6));
        assert_eq!(cached.shape, [3, 2, 1]);
        assert_eq!(cached.chunk_shape, [2, 2, 1]);
        assert_eq!(cached.chunk_grid, Some([2, 1, 1]));
        assert_eq!(cached.crop_origin, [0, 0, 0]);
        assert!(cached.applied_operations.is_empty());
        assert_eq!(cached.foreground_voxels, 3);
        assert_eq!(cached.bytes_written, 204);

        assert_eq!(
            read_f32_values(Path::new(&cached.image_cache_path)),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(
            read_u16_values(Path::new(&cached.label_cache_path)),
            vec![0, 1, 0, 2, 0, 3]
        );
        assert_eq!(
            read_u64_values(Path::new(cached.foreground_indices_path.as_ref().unwrap())),
            vec![1, 3, 5]
        );
        let prefix = read_u32_values(Path::new(cached.foreground_prefix_path.as_ref().unwrap()));
        assert_eq!(cached.foreground_prefix_shape, Some([4, 3, 2]));
        assert_eq!(prefix.len(), 24);
        assert_eq!(prefix.last().copied(), Some(3));
        assert_eq!(
            read_f32_values(Path::new(cached.image_chunk_cache_path.as_ref().unwrap())),
            vec![1.0, 2.0, 4.0, 5.0, 3.0, 0.0, 6.0, 0.0]
        );
        assert_eq!(
            read_u16_values(Path::new(cached.label_chunk_cache_path.as_ref().unwrap())),
            vec![0, 1, 2, 0, 0, 0, 3, 0]
        );

        let loaded_manifest = read_cache_manifest(&cache_dir).unwrap();
        assert_eq!(loaded_manifest, manifest);
        let inspection = crate::inspect_cache(&cache_dir).unwrap();
        assert_eq!(inspection.status, "ok");
        assert_eq!(inspection.cases, 1);
        assert_eq!(inspection.chunked_cases, 1);
        assert_eq!(inspection.artifact_bytes, 204);
        assert!(crate::validate_cache(&cache_dir).unwrap().errors.is_empty());
        fs::write(Path::new(&cached.image_cache_path), vec![0_u8; 24]).unwrap();
        assert!(
            crate::inspect_cache(&cache_dir).unwrap().errors.is_empty(),
            "fast inspection should not read payload hashes"
        );
        let strict = crate::validate_cache(&cache_dir).unwrap();
        assert_eq!(strict.status, "failed");
        assert!(strict
            .errors
            .iter()
            .any(|error| error.contains("image cache") && error.contains("SHA-256")));
        fs::write(
            Path::new(&cached.image_cache_path),
            NiftiFixture::new(&[3, 2, 1], 16, &[1.0, 1.0, 1.0])
                .append_f32_pixels(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .split_off(VOX_OFFSET),
        )
        .unwrap();
        let case_json = Path::new(&cached.image_cache_path)
            .parent()
            .unwrap()
            .join("case.json");
        let loaded_case: CachedCase =
            serde_json::from_str(&fs::read_to_string(case_json).unwrap()).unwrap();
        assert_eq!(loaded_case, *cached);

        let original_source_hash = cached.source_metadata_hash.clone();
        let original_cache_key = cached.cache_key.clone();
        fs::write(
            &image_path,
            NiftiFixture::new(&[3, 2, 1], 16, &[1.0, 1.0, 1.0])
                .append_f32_pixels(&[9.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        .unwrap();
        let changed_manifest = prepare_cache(&PrepareConfig {
            dataset_root: root.clone(),
            manifest_path: manifest_path.clone(),
            plan_path: plan_path.clone(),
            cache_dir: root.join("cache-changed"),
            chunk_shape: Some([2, 2, 2]),
        })
        .unwrap();
        assert_ne!(
            changed_manifest.cases[0].source_metadata_hash,
            original_source_hash
        );
        assert_ne!(changed_manifest.cases[0].cache_key, original_cache_key);
    }

    #[test]
    fn prepare_cache_stacks_structured_image_channels() {
        let root = temp_dir("prepare-multi-channel");
        let image0_path = root.join("case_multi_0000.nii");
        let image1_path = root.join("case_multi_0001.nii");
        let label_path = root.join("case_multi.nii");
        let manifest_path = root.join("manifest.json");
        let plan_path = root.join("plan.toml");
        let cache_dir = root.join("cache");

        fs::write(
            &image0_path,
            NiftiFixture::new(&[3, 2, 1], 16, &[1.0, 1.0, 1.0])
                .append_f32_pixels(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        .unwrap();
        fs::write(
            &image1_path,
            NiftiFixture::new(&[3, 2, 1], 16, &[1.0, 1.0, 1.0])
                .append_f32_pixels(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        )
        .unwrap();
        fs::write(
            &label_path,
            NiftiFixture::new(&[3, 2, 1], 512, &[1.0, 1.0, 1.0])
                .append_u16_pixels(&[0, 1, 0, 2, 0, 3]),
        )
        .unwrap();
        write_plan(&plan_path, identity_plan());
        write_manifest(
            &manifest_path,
            &root,
            vec![multi_channel_case(
                "case_multi",
                &[&image0_path, &image1_path],
                &label_path,
            )],
        );

        let manifest = prepare_cache(&PrepareConfig {
            dataset_root: root.clone(),
            manifest_path: manifest_path.clone(),
            plan_path: plan_path.clone(),
            cache_dir: cache_dir.clone(),
            chunk_shape: Some([2, 2, 2]),
        })
        .unwrap();
        let cached = &manifest.cases[0];

        assert_eq!(cached.image_channel_count, 2);
        assert_eq!(
            cached.image_paths,
            vec![path_text(&image0_path), path_text(&image1_path)]
        );
        assert_eq!(cached.image_path, path_text(&image0_path));
        assert_eq!(cached.shape, [3, 2, 1]);
        assert_eq!(cached.chunk_shape, [2, 2, 1]);
        assert_eq!(cached.chunk_grid, Some([2, 1, 1]));
        assert_eq!(cached.bytes_written, 260);
        assert_eq!(
            read_f32_values(Path::new(&cached.image_cache_path)),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        );
        assert_eq!(
            read_f32_values(Path::new(cached.image_chunk_cache_path.as_ref().unwrap())),
            vec![
                1.0, 2.0, 4.0, 5.0, 3.0, 0.0, 6.0, 0.0, 10.0, 20.0, 40.0, 50.0, 30.0, 0.0, 60.0,
                0.0
            ]
        );
        assert_eq!(crate::validate_cache(&cache_dir).unwrap().status, "ok");

        let original_source_hash = cached.source_metadata_hash.clone();
        let original_cache_key = cached.cache_key.clone();
        fs::write(
            &image1_path,
            NiftiFixture::new(&[3, 2, 1], 16, &[1.0, 1.0, 1.0])
                .append_f32_pixels(&[10.0, 99.0, 30.0, 40.0, 50.0, 60.0]),
        )
        .unwrap();
        let changed_manifest = prepare_cache(&PrepareConfig {
            dataset_root: root,
            manifest_path,
            plan_path,
            cache_dir: cache_dir.join("changed"),
            chunk_shape: Some([2, 2, 2]),
        })
        .unwrap();
        assert_ne!(
            changed_manifest.cases[0].source_metadata_hash,
            original_source_hash
        );
        assert_ne!(changed_manifest.cases[0].cache_key, original_cache_key);
    }

    #[test]
    fn prepare_cache_replaces_existing_case_with_corrupt_artifacts() {
        let root = temp_dir("prepare-replace-corrupt");
        let image_path = root.join("case_reuse_0000.nii");
        let label_path = root.join("case_reuse.nii");
        let manifest_path = root.join("manifest.json");
        let plan_path = root.join("plan.toml");
        let cache_dir = root.join("cache");

        fs::write(
            &image_path,
            NiftiFixture::new(&[3, 2, 1], 16, &[1.0, 1.0, 1.0])
                .append_f32_pixels(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        .unwrap();
        fs::write(
            &label_path,
            NiftiFixture::new(&[3, 2, 1], 512, &[1.0, 1.0, 1.0])
                .append_u16_pixels(&[0, 1, 0, 2, 0, 3]),
        )
        .unwrap();
        write_plan(&plan_path, identity_plan());
        write_manifest(
            &manifest_path,
            &root,
            vec![valid_case(
                "case_reuse",
                Some(&image_path),
                Some(&label_path),
            )],
        );

        let original = prepare_cache(&PrepareConfig {
            dataset_root: root.clone(),
            manifest_path: manifest_path.clone(),
            plan_path: plan_path.clone(),
            cache_dir: cache_dir.clone(),
            chunk_shape: Some([2, 2, 2]),
        })
        .unwrap();
        let cached = &original.cases[0];
        fs::write(Path::new(&cached.image_cache_path), vec![0_u8; 24]).unwrap();
        assert_eq!(crate::validate_cache(&cache_dir).unwrap().status, "failed");

        let repaired = prepare_cache(&PrepareConfig {
            dataset_root: root,
            manifest_path,
            plan_path,
            cache_dir: cache_dir.clone(),
            chunk_shape: Some([2, 2, 2]),
        })
        .unwrap();

        assert_eq!(repaired.cases[0], *cached);
        assert_eq!(crate::validate_cache(&cache_dir).unwrap().status, "ok");
        assert_eq!(
            read_f32_values(Path::new(&cached.image_cache_path)),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn prepare_cache_rejects_case_cache_path_that_is_a_file() {
        let root = temp_dir("prepare-final-case-file");
        let image_path = root.join("case_file_0000.nii");
        let label_path = root.join("case_file.nii");
        let manifest_path = root.join("manifest.json");
        let plan_path = root.join("plan.toml");
        let cache_dir = root.join("cache");

        fs::write(
            &image_path,
            NiftiFixture::new(&[3, 2, 1], 16, &[1.0, 1.0, 1.0])
                .append_f32_pixels(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        .unwrap();
        fs::write(
            &label_path,
            NiftiFixture::new(&[3, 2, 1], 512, &[1.0, 1.0, 1.0])
                .append_u16_pixels(&[0, 1, 0, 2, 0, 3]),
        )
        .unwrap();
        write_plan(&plan_path, identity_plan());
        write_manifest(
            &manifest_path,
            &root,
            vec![valid_case(
                "case_file",
                Some(&image_path),
                Some(&label_path),
            )],
        );

        let original = prepare_cache(&PrepareConfig {
            dataset_root: root.clone(),
            manifest_path: manifest_path.clone(),
            plan_path: plan_path.clone(),
            cache_dir: cache_dir.clone(),
            chunk_shape: None,
        })
        .unwrap();
        let case_path = cache_dir.join(&original.cases[0].cache_key);
        fs::remove_dir_all(&case_path).unwrap();
        fs::write(&case_path, b"not a case directory").unwrap();

        let error = prepare_cache(&PrepareConfig {
            dataset_root: root,
            manifest_path,
            plan_path,
            cache_dir: cache_dir.clone(),
            chunk_shape: None,
        })
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("cache case path exists but is not a directory"),
            "{error}"
        );
        assert!(
            !cache_dir.join(".staging").exists(),
            "failed promotion should remove staging artifacts"
        );
    }

    #[test]
    fn image_sources_reject_mixed_or_noncontiguous_channel_indices() {
        let root = temp_dir("prepare-channel-index-policy");
        let image0_path = root.join("case_multi_0000.nii");
        let image1_path = root.join("case_multi_0001.nii");
        let label_path = root.join("case_multi.nii");

        let mut mixed =
            multi_channel_case("case_multi", &[&image0_path, &image1_path], &label_path);
        mixed.images[1].channel_index = None;
        let error = image_sources_for_case(&mixed, &root).unwrap_err();
        assert!(error.to_string().contains("mixes indexed and unindexed"));

        let mut gapped =
            multi_channel_case("case_multi", &[&image0_path, &image1_path], &label_path);
        gapped.images[0].channel_index = Some(1);
        gapped.images[1].channel_index = Some(2);
        let error = image_sources_for_case(&gapped, &root).unwrap_err();
        assert!(error.to_string().contains("contiguous from 0"));

        let mut unindexed =
            multi_channel_case("case_multi", &[&image1_path, &image0_path], &label_path);
        for image in &mut unindexed.images {
            image.channel_index = None;
        }
        let sources = image_sources_for_case(&unindexed, &root).unwrap();
        assert_eq!(sources[0].path_text, path_text(&image0_path));
        assert_eq!(sources[1].path_text, path_text(&image1_path));
    }

    #[test]
    fn prepare_cache_fails_on_case_errors() {
        let root = temp_dir("prepare-strict-errors");
        let image_path = root.join("case_a_0000.nii");
        let label_path = root.join("case_a.nii");
        let mismatch_label_path = root.join("case_mismatch.nii");
        let manifest_path = root.join("manifest.json");
        let plan_path = root.join("plan.toml");
        let cache_dir = root.join("cache");

        fs::write(
            &image_path,
            NiftiFixture::new(&[3, 2, 1], 16, &[1.0, 1.0, 1.0])
                .append_f32_pixels(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        .unwrap();
        fs::write(
            &label_path,
            NiftiFixture::new(&[3, 2, 1], 512, &[1.0, 1.0, 1.0])
                .append_u16_pixels(&[0, 1, 0, 2, 0, 3]),
        )
        .unwrap();
        fs::write(
            &mismatch_label_path,
            NiftiFixture::new(&[3, 2, 1], 512, &[2.0, 1.0, 1.0])
                .append_u16_pixels(&[0, 1, 0, 2, 0, 3]),
        )
        .unwrap();
        write_plan(&plan_path, identity_plan());
        write_manifest(
            &manifest_path,
            &root,
            vec![
                valid_case("case_a", Some(&image_path), Some(&label_path)),
                valid_case(
                    "geometry_mismatch",
                    Some(&image_path),
                    Some(&mismatch_label_path),
                ),
            ],
        );

        let error = prepare_cache(&PrepareConfig {
            dataset_root: root.clone(),
            manifest_path,
            plan_path,
            cache_dir: cache_dir.clone(),
            chunk_shape: None,
        })
        .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("failed to prepare case geometry_mismatch"));
        assert!(message.contains("image and label source geometry differ"));
        assert!(!cache_dir.join(CACHE_MANIFEST).exists());
        let entries = fs::read_dir(&cache_dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert!(
            entries.is_empty(),
            "failed prepare should leave no promoted or staging artifacts: {entries:?}"
        );
    }

    #[test]
    fn prepare_cache_reports_manifest_plan_cache_dir_and_read_errors() {
        let root = temp_dir("prepare-errors");
        let manifest_path = root.join("manifest.json");
        let plan_path = root.join("plan.toml");
        let cache_dir = root.join("cache");

        let missing_manifest = prepare_cache(&PrepareConfig {
            dataset_root: root.clone(),
            manifest_path: root.join("missing-manifest.json"),
            plan_path: plan_path.clone(),
            cache_dir: cache_dir.clone(),
            chunk_shape: None,
        })
        .unwrap_err();
        assert!(matches!(missing_manifest, CacheError::Io { .. }));

        fs::write(&manifest_path, "{").unwrap();
        let invalid_manifest = prepare_cache(&PrepareConfig {
            dataset_root: root.clone(),
            manifest_path: manifest_path.clone(),
            plan_path: plan_path.clone(),
            cache_dir: cache_dir.clone(),
            chunk_shape: None,
        })
        .unwrap_err();
        assert!(matches!(invalid_manifest, CacheError::Json { .. }));

        write_manifest(&manifest_path, &root, Vec::new());
        fs::write(&plan_path, "not valid toml").unwrap();
        let invalid_plan = prepare_cache(&PrepareConfig {
            dataset_root: root.clone(),
            manifest_path: manifest_path.clone(),
            plan_path: plan_path.clone(),
            cache_dir: cache_dir.clone(),
            chunk_shape: None,
        })
        .unwrap_err();
        assert!(matches!(invalid_plan, CacheError::Transform(_)));

        write_plan(&plan_path, identity_plan());
        let file_cache_dir = root.join("cache-file");
        fs::write(&file_cache_dir, b"not a directory").unwrap();
        let cache_dir_error = prepare_cache(&PrepareConfig {
            dataset_root: root.clone(),
            manifest_path: manifest_path.clone(),
            plan_path: plan_path.clone(),
            cache_dir: file_cache_dir,
            chunk_shape: None,
        })
        .unwrap_err();
        assert!(matches!(cache_dir_error, CacheError::Io { .. }));

        fs::create_dir_all(&cache_dir).unwrap();
        fs::write(cache_dir.join(CACHE_MANIFEST), "{").unwrap();
        let read_error = read_cache_manifest(&cache_dir).unwrap_err();
        assert!(matches!(read_error, CacheError::Json { .. }));
    }

    #[test]
    fn prepare_case_without_chunks_uses_volume_shape_and_omits_chunk_paths() {
        let root = temp_dir("prepare-no-chunks");
        let image_path = root.join("case_b_0000.nii");
        let label_path = root.join("case_b.nii");
        let cache_dir = root.join("cache");
        fs::write(
            &image_path,
            NiftiFixture::new(&[1, 1, 1], 16, &[1.0, 1.0, 1.0]).append_f32_pixels(&[2.0]),
        )
        .unwrap();
        fs::write(
            &label_path,
            NiftiFixture::new(&[1, 1, 1], 512, &[1.0, 1.0, 1.0]).append_u16_pixels(&[1]),
        )
        .unwrap();
        let plan = TransformPlan::from_toml_str(identity_plan()).unwrap();
        let plan_hash = plan.plan_hash().unwrap();

        let cached = prepare_case(
            &valid_case("case_b", Some(&image_path), Some(&label_path)),
            &root,
            &plan,
            &plan_hash,
            &cache_dir,
            &cache_dir,
            None,
        )
        .unwrap();

        assert_eq!(cached.shape, [1, 1, 1]);
        assert_eq!(cached.chunk_shape, [1, 1, 1]);
        assert_eq!(cached.chunk_grid, None);
        assert_eq!(cached.image_chunk_cache_path, None);
        assert_eq!(cached.label_chunk_cache_path, None);
        assert_eq!(cached.foreground_voxels, 1);
        assert_eq!(cached.bytes_written, 46);
    }

    #[test]
    fn chunk_shape_changes_cache_key_and_promotes_chunk_artifacts() {
        let root = temp_dir("prepare-chunk-key");
        let image_path = root.join("case_chunk_0000.nii");
        let label_path = root.join("case_chunk.nii");
        let manifest_path = root.join("manifest.json");
        let plan_path = root.join("plan.toml");
        let cache_dir = root.join("cache");
        fs::write(
            &image_path,
            NiftiFixture::new(&[3, 2, 1], 16, &[1.0, 1.0, 1.0])
                .append_f32_pixels(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        .unwrap();
        fs::write(
            &label_path,
            NiftiFixture::new(&[3, 2, 1], 512, &[1.0, 1.0, 1.0])
                .append_u16_pixels(&[0, 1, 0, 2, 0, 3]),
        )
        .unwrap();
        write_plan(&plan_path, identity_plan());
        write_manifest(
            &manifest_path,
            &root,
            vec![valid_case(
                "case_chunk",
                Some(&image_path),
                Some(&label_path),
            )],
        );

        let resident = prepare_cache(&PrepareConfig {
            dataset_root: root.clone(),
            manifest_path: manifest_path.clone(),
            plan_path: plan_path.clone(),
            cache_dir: cache_dir.clone(),
            chunk_shape: None,
        })
        .unwrap();
        let chunked = prepare_cache(&PrepareConfig {
            dataset_root: root,
            manifest_path,
            plan_path,
            cache_dir: cache_dir.clone(),
            chunk_shape: Some([2, 2, 2]),
        })
        .unwrap();

        assert_ne!(resident.cases[0].cache_key, chunked.cases[0].cache_key);
        assert!(chunked.cases[0].image_chunk_cache_path.is_some());
        assert!(Path::new(chunked.cases[0].image_chunk_cache_path.as_ref().unwrap()).is_file());
        assert_eq!(crate::validate_cache(&cache_dir).unwrap().status, "ok");
    }

    #[test]
    fn prepare_cache_resolves_manifest_paths_under_dataset_root() {
        let root = temp_dir("prepare-root-relative");
        let image_dir = root.join("images");
        let label_dir = root.join("labels");
        fs::create_dir_all(&image_dir).unwrap();
        fs::create_dir_all(&label_dir).unwrap();
        let image_path = image_dir.join("case_c.nii");
        let label_path = label_dir.join("case_c.nii");
        let manifest_path = root.join("manifest.json");
        let plan_path = root.join("plan.toml");
        let cache_dir = root.join("cache");
        fs::write(
            &image_path,
            NiftiFixture::new(&[1, 1, 1], 16, &[1.0, 1.0, 1.0]).append_f32_pixels(&[2.0]),
        )
        .unwrap();
        fs::write(
            &label_path,
            NiftiFixture::new(&[1, 1, 1], 512, &[1.0, 1.0, 1.0]).append_u16_pixels(&[1]),
        )
        .unwrap();
        write_plan(&plan_path, identity_plan());
        write_manifest(
            &manifest_path,
            &root,
            vec![valid_case(
                "case_c",
                Some(Path::new("images/case_c.nii")),
                Some(Path::new("labels/case_c.nii")),
            )],
        );

        let manifest = prepare_cache(&PrepareConfig {
            dataset_root: root.clone(),
            manifest_path,
            plan_path,
            cache_dir,
            chunk_shape: None,
        })
        .unwrap();

        assert_eq!(manifest.summary.cached_cases, 1);
        assert_eq!(manifest.cases[0].image_path, path_text(&image_path));
        assert_eq!(manifest.cases[0].label_path, path_text(&label_path));
    }

    #[test]
    fn foreground_prefix_values_count_3d_regions() {
        let label = Volume3D::new([2, 2, 2], vec![1, 0, 0, 1, 0, 1, 1, 0]).unwrap();
        let prefix = foreground_prefix_values(&label).unwrap();
        let index = |x: usize, y: usize, z: usize| x + 3 * (y + 3 * z);

        assert_eq!(prefix[index(0, 0, 0)], 0);
        assert_eq!(prefix[index(1, 1, 1)], 1);
        assert_eq!(prefix[index(2, 1, 2)], 2);
        assert_eq!(prefix[index(1, 2, 2)], 2);
        assert_eq!(prefix[index(2, 2, 2)], 4);
    }

    #[test]
    fn chunk_helpers_clamp_shape_compute_grid_and_pad_partial_chunks() {
        assert_eq!(valid_chunk_shape([0, 99, 1], [3, 2, 1]), [1, 2, 1]);
        assert_eq!(chunk_grid([5, 4, 3], [2, 3, 2]), [3, 2, 2]);
        assert_eq!(chunked_value_count([5, 4, 3], [2, 3, 2]), 144);
        assert_eq!(bitpix_for(2), 8);
        assert_eq!(bitpix_for(64), 64);
        assert_eq!(bitpix_for(999), 0);

        let volume = Volume3D::new([3, 2, 1], vec![1_u16, 2, 3, 4, 5, 6]).unwrap();
        let mut bytes = Vec::new();
        write_chunked_values(
            &volume,
            [2, 2, 1],
            0,
            |value, out| {
                out.write_all(&value.to_le_bytes())?;
                Ok(())
            },
            &mut bytes,
        )
        .unwrap();
        assert_eq!(decode_u16(&bytes), vec![1, 2, 4, 5, 3, 0, 6, 0]);
    }

    #[test]
    fn cache_key_is_stable_hash_without_raw_case_id_prefix() {
        let first = cache_key("case_a", "source", "plan", 1, "resident", None, "writer");
        let second = cache_key("case_a", "source", "plan", 1, "resident", None, "writer");
        let different_plan = cache_key(
            "case_a",
            "source",
            "other-plan",
            1,
            "resident",
            None,
            "writer",
        );
        let different_chunk = cache_key(
            "case_a",
            "source",
            "plan",
            1,
            "chunked",
            Some([2, 2, 1]),
            "writer",
        );
        let different_schema = cache_key("case_a", "source", "plan", 2, "resident", None, "writer");
        let different_writer = cache_key(
            "case_a",
            "source",
            "plan",
            1,
            "resident",
            None,
            "other-writer",
        );
        let path_like = cache_key("../case_a", "source", "plan", 1, "resident", None, "writer");

        assert_eq!(first, second);
        assert_ne!(first, different_plan);
        assert_ne!(first, different_chunk);
        assert_ne!(first, different_schema);
        assert_ne!(first, different_writer);
        assert_ne!(first, path_like);
        assert_eq!(first.len(), 64);
        assert!(!path_like.contains('/'));
        assert!(!path_like.contains(".."));
    }

    fn valid_case(
        case_id: &str,
        image_path: Option<&Path>,
        label_path: Option<&Path>,
    ) -> CaseManifest {
        CaseManifest {
            case_id: case_id.to_string(),
            status: CaseStatus::Valid,
            image_path: image_path.map(path_text),
            label_path: label_path.map(path_text),
            image: None,
            images: Vec::new(),
            label: None,
            problems: Vec::new(),
        }
    }

    fn multi_channel_case(case_id: &str, image_paths: &[&Path], label_path: &Path) -> CaseManifest {
        let mut case = valid_case(case_id, image_paths.first().copied(), Some(label_path));
        case.images = image_paths
            .iter()
            .enumerate()
            .map(|(index, path)| CaseImage {
                path: path_text(path),
                channel_index: Some(index as u16),
                modality: None,
                image: None,
            })
            .collect();
        case
    }

    fn invalid_case(case_id: &str) -> CaseManifest {
        CaseManifest {
            case_id: case_id.to_string(),
            status: CaseStatus::Invalid,
            image_path: None,
            label_path: None,
            image: None,
            images: Vec::new(),
            label: None,
            problems: Vec::new(),
        }
    }

    fn write_manifest(path: &Path, root: &Path, cases: Vec<CaseManifest>) {
        let valid_cases = cases
            .iter()
            .filter(|case| case.status == CaseStatus::Valid)
            .count();
        let manifest = DatasetManifest {
            dataset_root: path_text(root),
            images_dir: path_text(root),
            labels_dir: path_text(root),
            layout: DatasetLayout::Flat,
            summary: ValidationSummary {
                total_cases: cases.len(),
                valid_cases,
                invalid_cases: cases.len() - valid_cases,
                ..ValidationSummary::default()
            },
            cases,
        };
        fs::write(path, serde_json::to_string_pretty(&manifest).unwrap()).unwrap();
    }

    fn identity_plan() -> &'static str {
        r#"
name = "identity"
operations = []
image_interpolation = "linear"
label_interpolation = "nearest"
"#
    }

    fn write_plan(path: &Path, text: &str) {
        fs::write(path, text).unwrap();
    }

    fn path_text(path: &Path) -> String {
        path.to_string_lossy().into_owned()
    }

    fn read_f32_values(path: &Path) -> Vec<f32> {
        fs::read(path)
            .unwrap()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    fn read_u16_values(path: &Path) -> Vec<u16> {
        decode_u16(&fs::read(path).unwrap())
    }

    fn decode_u16(bytes: &[u8]) -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    fn read_u32_values(path: &Path) -> Vec<u32> {
        fs::read(path)
            .unwrap()
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    fn read_u64_values(path: &Path) -> Vec<u64> {
        fs::read(path)
            .unwrap()
            .chunks_exact(8)
            .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    fn bitpix_for(datatype: i16) -> i16 {
        match datatype {
            2 | 256 => 8,
            4 | 512 => 16,
            8 | 16 | 768 => 32,
            64 => 64,
            _ => 0,
        }
    }

    fn temp_dir(case: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "medkit-cache-prepare-{case}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }
}
