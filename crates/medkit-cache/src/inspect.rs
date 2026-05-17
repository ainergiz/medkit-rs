use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{read_cache_manifest, CachedCase, Result};

/// Human-readable storage kind for a cached case.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheStorageKind {
    /// Resident row-major image and label raw files.
    Resident,
    /// Resident raw files plus fixed-size chunked files.
    Chunked,
}

/// Inspection result for one cached case.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheCaseInspection {
    /// Case id.
    pub case_id: String,
    /// Cache key.
    pub cache_key: String,
    /// Cached shape in x, y, z order.
    pub shape: [usize; 3],
    /// Storage kind.
    pub storage: CacheStorageKind,
    /// Chunk shape in x, y, z order when chunked storage exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_shape: Option<[usize; 3]>,
    /// Chunk grid in x, y, z order when chunked storage exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_grid: Option<[usize; 3]>,
    /// Bytes occupied by checked case artifacts.
    pub bytes: u64,
    /// Errors found for this case.
    pub errors: Vec<String>,
}

/// Cache inspection and validation summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheInspection {
    /// Cache directory inspected.
    pub cache_dir: String,
    /// Cache manifest schema version.
    pub version: u32,
    /// Validation status, either `ok` or `failed`.
    pub status: String,
    /// Number of cases in the manifest.
    pub cases: usize,
    /// Number of cases with chunked storage.
    pub chunked_cases: usize,
    /// Total bytes occupied by checked case artifacts.
    pub artifact_bytes: u64,
    /// Transform plan hash from the manifest.
    pub transform_plan_hash: String,
    /// Top-level validation errors.
    pub errors: Vec<String>,
    /// Per-case inspection results.
    pub case_reports: Vec<CacheCaseInspection>,
}

/// Inspects cache metadata and artifact sizes without reading tensor payloads.
pub fn inspect_cache(cache_dir: impl AsRef<Path>) -> Result<CacheInspection> {
    let cache_dir = cache_dir.as_ref();
    let manifest = read_cache_manifest(cache_dir)?;
    let mut case_reports = Vec::with_capacity(manifest.cases.len());
    let mut errors = Vec::new();
    let mut artifact_bytes = 0_u64;
    let mut chunked_cases = 0_usize;
    for case in &manifest.cases {
        let report = inspect_case(case);
        if report.storage == CacheStorageKind::Chunked {
            chunked_cases += 1;
        }
        artifact_bytes += report.bytes;
        errors.extend(
            report
                .errors
                .iter()
                .map(|error| format!("{}: {error}", case.case_id)),
        );
        case_reports.push(report);
    }
    let status = if errors.is_empty() { "ok" } else { "failed" }.to_string();
    Ok(CacheInspection {
        cache_dir: manifest.cache_dir,
        version: manifest.version,
        status,
        cases: manifest.cases.len(),
        chunked_cases,
        artifact_bytes,
        transform_plan_hash: manifest.transform_plan_hash,
        errors,
        case_reports,
    })
}

/// Validates cache metadata and artifact sizes.
pub fn validate_cache(cache_dir: impl AsRef<Path>) -> Result<CacheInspection> {
    inspect_cache(cache_dir)
}

fn inspect_case(case: &CachedCase) -> CacheCaseInspection {
    let mut errors = Vec::new();
    let mut bytes = 0_u64;
    let voxels = value_count(case.shape);
    bytes += check_file_bytes(
        &case.image_cache_path,
        voxels.clone().map(|count| count * 4),
        "image cache",
        &mut errors,
    );
    bytes += check_file_bytes(
        &case.label_cache_path,
        voxels.map(|count| count * 2),
        "label cache",
        &mut errors,
    );
    if let Some(path) = &case.foreground_indices_path {
        bytes += check_file_multiple(path, 8, "foreground indices", &mut errors);
    }
    if let Some(path) = &case.foreground_prefix_path {
        let prefix_shape = case.foreground_prefix_shape.unwrap_or([
            case.shape[0] + 1,
            case.shape[1] + 1,
            case.shape[2] + 1,
        ]);
        bytes += check_file_bytes(
            path,
            value_count(prefix_shape).map(|count| count * 4),
            "foreground prefix",
            &mut errors,
        );
    }

    let storage = if case.image_chunk_cache_path.is_some()
        || case.label_chunk_cache_path.is_some()
        || case.chunk_grid.is_some()
    {
        inspect_chunked_case(case, &mut bytes, &mut errors);
        CacheStorageKind::Chunked
    } else {
        CacheStorageKind::Resident
    };

    CacheCaseInspection {
        case_id: case.case_id.clone(),
        cache_key: case.cache_key.clone(),
        shape: case.shape,
        storage,
        chunk_shape: case.chunk_grid.map(|_| case.chunk_shape),
        chunk_grid: case.chunk_grid,
        bytes,
        errors,
    }
}

fn inspect_chunked_case(case: &CachedCase, bytes: &mut u64, errors: &mut Vec<String>) {
    let Some(chunk_grid) = case.chunk_grid else {
        errors.push("chunked storage is missing chunk_grid".to_string());
        return;
    };
    let chunk_values = value_count(chunk_grid).and_then(|chunks| {
        value_count(case.chunk_shape).and_then(|values_per_chunk| {
            chunks
                .checked_mul(values_per_chunk)
                .ok_or_else(|| "chunked value count overflow".to_string())
        })
    });
    match &case.image_chunk_cache_path {
        Some(path) => {
            *bytes += check_file_bytes(
                path,
                chunk_values.clone().map(|count| count * 4),
                "image chunk cache",
                errors,
            );
        }
        None => errors.push("chunked storage is missing image_chunk_cache_path".to_string()),
    }
    match &case.label_chunk_cache_path {
        Some(path) => {
            *bytes += check_file_bytes(
                path,
                chunk_values.map(|count| count * 2),
                "label chunk cache",
                errors,
            );
        }
        None => errors.push("chunked storage is missing label_chunk_cache_path".to_string()),
    }
}

fn check_file_bytes(
    path: &str,
    expected: std::result::Result<usize, String>,
    kind: &str,
    errors: &mut Vec<String>,
) -> u64 {
    let expected = match expected {
        Ok(expected) => expected,
        Err(error) => {
            errors.push(format!("{kind} size overflow: {error}"));
            return 0;
        }
    };
    match fs::metadata(path) {
        Ok(metadata) if metadata.len() == expected as u64 => metadata.len(),
        Ok(metadata) => {
            errors.push(format!(
                "{kind} {} has {} bytes, expected {expected}",
                PathBuf::from(path).display(),
                metadata.len()
            ));
            metadata.len()
        }
        Err(error) => {
            errors.push(format!(
                "missing or unreadable {kind} {}: {error}",
                PathBuf::from(path).display()
            ));
            0
        }
    }
}

fn check_file_multiple(path: &str, multiple: u64, kind: &str, errors: &mut Vec<String>) -> u64 {
    match fs::metadata(path) {
        Ok(metadata) if metadata.len() % multiple == 0 => metadata.len(),
        Ok(metadata) => {
            errors.push(format!(
                "{kind} {} has {} bytes, expected a multiple of {multiple}",
                PathBuf::from(path).display(),
                metadata.len()
            ));
            metadata.len()
        }
        Err(error) => {
            errors.push(format!(
                "missing or unreadable {kind} {}: {error}",
                PathBuf::from(path).display()
            ));
            0
        }
    }
}

fn value_count(shape: [usize; 3]) -> std::result::Result<usize, String> {
    shape[0]
        .checked_mul(shape[1])
        .and_then(|value| value.checked_mul(shape[2]))
        .ok_or_else(|| format!("shape {shape:?} overflows usize"))
}
