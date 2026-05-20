#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::{self, File},
    io::{self, BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
    sync::Arc,
    thread,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use image::{imageops::FilterType, DynamicImage, GrayImage, Luma};
use memmap2::Mmap;
use sha2::{Digest, Sha256};

use crate::{
    error::CxrError,
    manifest::{is_dicom_record, read_manifest},
    types::{
        CacheConfig, CacheSplitSummary, CacheSummary, CacheValidationSummary, CxrCacheBatch,
        CxrCacheDType, CxrCacheReadMode, CxrCacheReader, CxrIndexedReadMetrics, CxrRecord,
        DicomPresentationPolicy, ImageSizePolicy, LabelPolicy, Normalization, SplitFile,
        SplitPolicyMetadata, TransferSyntaxPolicy, ValidateCacheConfig, CXR_CACHE_SCHEMA_VERSION,
        CXR_REPORT_SCHEMA_VERSION,
    },
    util::{collect_targets, directory_size, hash_file, resolve_cache_path, write_json},
};

struct CxrIndexedRun {
    start_sample: usize,
    out_indices: Vec<usize>,
}

#[derive(Debug)]
struct CxrIndexedRunRead {
    out_indices: Vec<usize>,
    image_start: usize,
    label_start: usize,
}

#[derive(Debug)]
struct CxrIndexedChunkRead {
    runs: Vec<CxrIndexedRunRead>,
    images: Vec<f32>,
    labels: Vec<f32>,
    masks: Vec<f32>,
}

#[derive(Clone, Copy)]
struct CacheSplitFiles<'a> {
    images: &'a File,
    labels: &'a File,
    masks: &'a File,
}

struct CacheSplitBuffers<'a> {
    images: &'a mut [f32],
    labels: &'a mut [f32],
    masks: &'a mut [f32],
}

#[derive(Debug, Clone, Default)]
pub struct CxrCacheOptions {
    pub targets: Vec<String>,
    pub recipe_hash: String,
    pub recipe_path: String,
    pub label_policy: LabelPolicy,
    pub cache_dtype: CxrCacheDType,
    pub image_size_policy: ImageSizePolicy,
    pub dicom_presentation_policy: DicomPresentationPolicy,
    pub transfer_syntax_policy: TransferSyntaxPolicy,
    pub split_policy: SplitPolicyMetadata,
}

pub fn cache_cxr(config: &CacheConfig) -> Result<CacheSummary, CxrError> {
    cache_cxr_with_options(config, &CxrCacheOptions::default())
}

pub fn cache_cxr_with_options(
    config: &CacheConfig,
    options: &CxrCacheOptions,
) -> Result<CacheSummary, CxrError> {
    let records = read_manifest(&config.manifest_path)?;
    let split_file = read_split_file(&config.splits_path)?;
    validate_split_membership(&records, &split_file)?;
    let image_size = image_size_from_plan(&config.plan_path)?;
    fs::create_dir_all(&config.cache_dir)?;
    let staging_dir = staging_dir(&config.cache_dir);
    fs::create_dir_all(&staging_dir)?;
    let result = cache_cxr_in_staging(
        config,
        options,
        &records,
        &split_file,
        image_size,
        &staging_dir,
    );
    match result {
        Ok(summary) => {
            promote_staged_cxr_cache(&staging_dir, &config.cache_dir, &summary)?;
            Ok(summary)
        }
        Err(error) => {
            cleanup_staging(&staging_dir)?;
            Err(error)
        }
    }
}

fn cache_cxr_in_staging(
    config: &CacheConfig,
    options: &CxrCacheOptions,
    records: &[CxrRecord],
    split_file: &SplitFile,
    image_size: usize,
    staging_dir: &Path,
) -> Result<CacheSummary, CxrError> {
    let targets = if options.targets.is_empty() {
        collect_targets(records)
    } else {
        options.targets.clone()
    };
    let transform_plan_hash = hash_file(&config.plan_path)?;
    let transform_fingerprint = cache_transform_fingerprint(&transform_plan_hash, options)?;
    let transform_description = if records.iter().any(is_dicom_record) {
        "medkit-dicom presentation to MONOCHROME2 u8, resize longest side, pad square, normalize dataset mean/std"
    } else {
        "decode grayscale, resize longest side, pad square, normalize dataset mean/std"
    };
    let cache_dtype = options.cache_dtype;
    let train_records = records_for_split(records, &split_file.train)?;
    let normalization = estimate_normalization(&train_records, image_size, options)?;
    let mut splits = BTreeMap::new();
    for (name, ids) in [
        ("train", &split_file.train),
        ("val", &split_file.val),
        ("test", &split_file.test),
    ] {
        let split_records = records_for_split(records, ids)?;
        let split_summary = write_cache_split(
            staging_dir,
            name,
            &split_records,
            &targets,
            image_size,
            &normalization,
            options,
        )?;
        splits.insert(name.to_string(), split_summary);
    }

    let split_names = splits.keys().cloned().collect::<Vec<_>>();
    let summary = CacheSummary {
        cache_schema_version: CXR_CACHE_SCHEMA_VERSION,
        report_schema_version: CXR_REPORT_SCHEMA_VERSION,
        cache_dir: config.cache_dir.display().to_string(),
        image_size,
        channels: 1,
        dtype: cache_dtype.as_str().to_string(),
        targets,
        label_policy: options.label_policy.clone(),
        normalization,
        transform_fingerprint,
        transform_plan_hash,
        recipe_hash: options.recipe_hash.clone(),
        recipe_path: options.recipe_path.clone(),
        source_manifest_checksum: hash_file(&config.manifest_path)?,
        split_names,
        image_size_policy: if options.image_size_policy.height > 0 {
            let mut policy = options.image_size_policy.clone();
            policy.dtype = cache_dtype.as_str().to_string();
            policy
        } else {
            ImageSizePolicy {
                channels: 1,
                height: image_size,
                width: image_size,
                dtype: cache_dtype.as_str().to_string(),
                transform: transform_description.to_string(),
            }
        },
        dicom_presentation_policy: options.dicom_presentation_policy.clone(),
        transfer_syntax_policy: options.transfer_syntax_policy.clone(),
        split_policy: effective_split_policy(options, split_file),
        splits,
        failed_samples: Vec::new(),
        cache_size_bytes: directory_size(staging_dir)?,
    };
    write_json(&staging_dir.join("cache-metadata.json"), &summary)?;
    Ok(summary)
}

pub fn read_cache_summary(cache_dir: &Path) -> Result<CacheSummary, CxrError> {
    let text = fs::read_to_string(cache_dir.join("cache-metadata.json"))?;
    Ok(serde_json::from_str(&text)?)
}

fn staging_dir(cache_dir: &Path) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    cache_dir
        .join(".staging")
        .join(format!("cxr-cache-{}-{nanos}", std::process::id()))
}

fn cleanup_staging(staging_dir: &Path) -> Result<(), CxrError> {
    match fs::remove_dir_all(staging_dir) {
        Ok(()) => {
            if let Some(parent) = staging_dir.parent() {
                let _ = fs::remove_dir(parent);
            }
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(CxrError::Io(error)),
    }
}

fn promote_staged_cxr_cache(
    staging_dir: &Path,
    cache_dir: &Path,
    summary: &CacheSummary,
) -> Result<(), CxrError> {
    for split in summary.splits.values() {
        for relative in [
            &split.images_path,
            &split.labels_path,
            &split.masks_path,
            &split.metadata_path,
        ] {
            promote_staged_file(staging_dir, cache_dir, relative)?;
        }
    }
    promote_staged_file(staging_dir, cache_dir, "cache-metadata.json")?;
    cleanup_staging(staging_dir)
}

fn promote_staged_file(
    staging_dir: &Path,
    cache_dir: &Path,
    relative: &str,
) -> Result<(), CxrError> {
    let staged = staging_dir.join(relative);
    let final_path = cache_dir.join(relative);
    replace_staged_file(&staged, &final_path)?;
    sync_parent_dir(&final_path);
    Ok(())
}

#[cfg(unix)]
fn replace_staged_file(staged: &Path, final_path: &Path) -> Result<(), CxrError> {
    fs::rename(staged, final_path)?;
    Ok(())
}

#[cfg(not(unix))]
fn replace_staged_file(staged: &Path, final_path: &Path) -> Result<(), CxrError> {
    if !final_path.exists() {
        fs::rename(staged, final_path)?;
        return Ok(());
    }
    let backup = unique_sibling_path(final_path, "replace-cxr");
    if backup.exists() {
        fs::remove_file(&backup)?;
    }
    fs::rename(final_path, &backup)?;
    match fs::rename(staged, final_path) {
        Ok(()) => {
            fs::remove_file(&backup)?;
            Ok(())
        }
        Err(error) => {
            let _ = fs::rename(&backup, final_path);
            Err(CxrError::Io(error))
        }
    }
}

#[cfg(not(unix))]
fn unique_sibling_path(path: &Path, prefix: &str) -> std::path::PathBuf {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact");
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    parent.join(format!(".{prefix}-{name}-{}-{nanos}", std::process::id()))
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

pub fn validate_cache_cxr(
    config: &ValidateCacheConfig,
) -> Result<CacheValidationSummary, CxrError> {
    let summary = read_cache_summary(&config.cache_dir)?;
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    if summary.cache_schema_version != CXR_CACHE_SCHEMA_VERSION {
        errors.push(format!(
            "unsupported CXR cache schema version {}; expected {}; rebuild with `medkit cxr cache`",
            summary.cache_schema_version, CXR_CACHE_SCHEMA_VERSION
        ));
    }
    if summary.report_schema_version != CXR_REPORT_SCHEMA_VERSION {
        warnings.push(format!(
            "CXR report schema version is {}; current writer is {}",
            summary.report_schema_version, CXR_REPORT_SCHEMA_VERSION
        ));
    }
    if let Some(expected_targets) = &config.expected_targets {
        if expected_targets != &summary.targets {
            errors.push(format!(
                "target-list mismatch: cache has [{}], expected [{}]",
                summary.targets.join(", "),
                expected_targets.join(", ")
            ));
        }
    }
    if let Some(plan_path) = &config.plan_path {
        let expected = hash_file(plan_path)?;
        if summary.transform_fingerprint != expected && summary.transform_plan_hash != expected {
            errors.push(format!(
                "stale transform fingerprint: cache has {}, expected {} from {}",
                if summary.transform_fingerprint.is_empty() {
                    &summary.transform_plan_hash
                } else {
                    &summary.transform_fingerprint
                },
                expected,
                plan_path.display()
            ));
        }
    }

    let checked_splits = match &config.split {
        Some(split) => {
            if summary.splits.contains_key(split) {
                vec![split.clone()]
            } else {
                errors.push(format!("cache split {split:?} not found"));
                Vec::new()
            }
        }
        None => summary.splits.keys().cloned().collect::<Vec<_>>(),
    };
    let image_dtype = if summary.dtype.is_empty() {
        CxrCacheDType::Float32
    } else {
        summary.dtype.parse().map_err(CxrError::Message)?
    };

    for split in &checked_splits {
        let split_summary = summary
            .splits
            .get(split)
            .expect("checked split names are drawn from cache metadata");
        let result = validate_cache_split_files(
            &config.cache_dir,
            split,
            split_summary,
            image_dtype,
            summary.targets.len(),
            config.expected_image_shape,
            &mut errors,
        );
        result?;
    }

    if !summary.failed_samples.is_empty() {
        errors.push(format!(
            "cache contains {} failed preprocessed samples; first failure: {}",
            summary.failed_samples.len(),
            summary.failed_samples[0]
        ));
    }

    let status = if errors.is_empty() { "ok" } else { "failed" }.to_string();
    let report = CacheValidationSummary {
        cache_dir: config.cache_dir.display().to_string(),
        cache_schema_version: summary.cache_schema_version,
        expected_cache_schema_version: CXR_CACHE_SCHEMA_VERSION,
        report_schema_version: summary.report_schema_version,
        status,
        errors,
        warnings,
        targets: summary.targets,
        label_policy: summary.label_policy,
        split_names: if summary.split_names.is_empty() {
            summary.splits.keys().cloned().collect()
        } else {
            summary.split_names
        },
        checked_splits,
        image_size_policy: summary.image_size_policy,
        transform_fingerprint: if summary.transform_fingerprint.is_empty() {
            summary.transform_plan_hash
        } else {
            summary.transform_fingerprint
        },
        recipe_hash: summary.recipe_hash,
        source_manifest_checksum: summary.source_manifest_checksum,
        cache_size_bytes: directory_size(&config.cache_dir)?,
    };
    if let Some(path) = &config.json_path {
        write_json(path, &report)?;
    }
    if let Some(path) = &config.report_path {
        write_cache_validation_report(path, &report)?;
    }
    Ok(report)
}

impl CxrCacheReader {
    pub fn open(cache_dir: impl AsRef<Path>, split: impl Into<String>) -> Result<Self, CxrError> {
        Self::open_with_read_mode(cache_dir, split, CxrCacheReadMode::Mmap)
    }

    pub fn open_with_read_mode(
        cache_dir: impl AsRef<Path>,
        split: impl Into<String>,
        read_mode: CxrCacheReadMode,
    ) -> Result<Self, CxrError> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        let split = split.into();
        let summary = read_cache_summary(&cache_dir)?;
        if summary.cache_schema_version != CXR_CACHE_SCHEMA_VERSION {
            return Err(CxrError::Message(format!(
                "unsupported CXR cache schema version {}; expected {}; rebuild with `medkit cxr cache`",
                summary.cache_schema_version, CXR_CACHE_SCHEMA_VERSION
            )));
        }
        let split_summary = summary
            .splits
            .get(&split)
            .cloned()
            .ok_or_else(|| CxrError::Message(format!("cache split {split:?} not found")))?;
        if split_summary.shape[1] != 1 {
            return Err(CxrError::Message(format!(
                "expected single-channel CXR cache, got shape {:?}",
                split_summary.shape
            )));
        }
        let image_dtype = if summary.dtype.is_empty() {
            CxrCacheDType::Float32
        } else {
            summary.dtype.parse().map_err(CxrError::Message)?
        };
        let mut file_errors = Vec::new();
        validate_cache_split_files(
            &cache_dir,
            &split,
            &split_summary,
            image_dtype,
            summary.targets.len(),
            None,
            &mut file_errors,
        )?;
        if !file_errors.is_empty() {
            return Err(CxrError::Message(file_errors.join("; ")));
        }
        let target_count = summary.targets.len();
        let image_values_per_sample = split_summary.shape[1]
            .checked_mul(split_summary.shape[2])
            .and_then(|value| value.checked_mul(split_summary.shape[3]))
            .ok_or_else(|| CxrError::Message("image shape overflow".to_string()))?;
        let metadata_path = resolve_cache_path(&cache_dir, &split_summary.metadata_path);
        let records = read_cache_metadata(&metadata_path)?;
        let images_path = resolve_cache_path(&cache_dir, &split_summary.images_path);
        let labels_path = resolve_cache_path(&cache_dir, &split_summary.labels_path);
        let masks_path = resolve_cache_path(&cache_dir, &split_summary.masks_path);
        let (images_mmap, labels_mmap, masks_mmap, images_file, labels_file, masks_file) =
            match read_mode {
                CxrCacheReadMode::Mmap => (
                    Some(Arc::new(mmap_file(&images_path)?)),
                    Some(Arc::new(mmap_file(&labels_path)?)),
                    Some(Arc::new(mmap_file(&masks_path)?)),
                    None,
                    None,
                    None,
                ),
                CxrCacheReadMode::Stream => (
                    None,
                    None,
                    None,
                    Some(Arc::new(File::open(&images_path)?)),
                    Some(Arc::new(File::open(&labels_path)?)),
                    Some(Arc::new(File::open(&masks_path)?)),
                ),
            };
        Ok(Self {
            split,
            read_mode,
            image_dtype,
            summary,
            split_summary,
            records,
            image_values_per_sample,
            target_count,
            images_path,
            labels_path,
            masks_path,
            images_mmap,
            labels_mmap,
            masks_mmap,
            images_file,
            labels_file,
            masks_file,
        })
    }

    pub fn split(&self) -> &str {
        &self.split
    }

    pub fn read_mode(&self) -> CxrCacheReadMode {
        self.read_mode
    }

    pub fn samples(&self) -> usize {
        self.split_summary.samples
    }

    pub fn targets(&self) -> &[String] {
        &self.summary.targets
    }

    pub fn image_shape(&self) -> [usize; 4] {
        self.split_summary.shape
    }

    pub fn image_size(&self) -> usize {
        self.summary.image_size
    }

    pub fn cache_schema_version(&self) -> u32 {
        self.summary.cache_schema_version
    }

    pub fn label_policy(&self) -> &LabelPolicy {
        &self.summary.label_policy
    }

    pub fn records_for_range(
        &self,
        start_index: usize,
        samples: usize,
    ) -> Result<Vec<CxrRecord>, CxrError> {
        let end = start_index
            .checked_add(samples)
            .ok_or_else(|| CxrError::Message("metadata range overflow".to_string()))?;
        if end > self.records.len() {
            return Err(CxrError::Message(format!(
                "metadata range {start_index}..{end} out of bounds for {} samples",
                self.records.len()
            )));
        }
        Ok(self.records[start_index..end].to_vec())
    }

    pub fn records_for_indices(&self, indices: &[usize]) -> Result<Vec<CxrRecord>, CxrError> {
        indices
            .iter()
            .map(|index| {
                self.records.get(*index).cloned().ok_or_else(|| {
                    CxrError::Message(format!(
                        "metadata sample index {index} out of bounds for {} samples",
                        self.records.len()
                    ))
                })
            })
            .collect()
    }

    pub fn read_batch(
        &self,
        start_index: usize,
        batch_size: usize,
    ) -> Result<CxrCacheBatch, CxrError> {
        let samples = self.actual_batch_size(start_index, batch_size);
        let mut images = vec![0.0f32; samples * self.image_values_per_sample];
        let mut labels = vec![0.0f32; samples * self.target_count];
        let mut masks = vec![0.0f32; samples * self.target_count];
        let written = self.fill_batch(
            start_index,
            batch_size,
            &mut images,
            &mut labels,
            &mut masks,
        )?;
        images.truncate(written * self.image_values_per_sample);
        labels.truncate(written * self.target_count);
        masks.truncate(written * self.target_count);
        let records = if written == 0 {
            Vec::new()
        } else {
            self.records[start_index..start_index + written].to_vec()
        };
        Ok(CxrCacheBatch {
            samples: written,
            image_shape: [
                written,
                self.split_summary.shape[1],
                self.split_summary.shape[2],
                self.split_summary.shape[3],
            ],
            labels_shape: [written, self.target_count],
            images,
            labels,
            masks,
            records,
        })
    }

    pub fn fill_batch(
        &self,
        start_index: usize,
        batch_size: usize,
        image_out: &mut [f32],
        labels_out: &mut [f32],
        masks_out: &mut [f32],
    ) -> Result<usize, CxrError> {
        let samples = self.actual_batch_size(start_index, batch_size);
        let image_values = samples * self.image_values_per_sample;
        let label_values = samples * self.target_count;
        if image_out.len() < image_values {
            return Err(CxrError::Message(format!(
                "image output buffer too small: need {image_values}, got {}",
                image_out.len()
            )));
        }
        if labels_out.len() < label_values {
            return Err(CxrError::Message(format!(
                "label output buffer too small: need {label_values}, got {}",
                labels_out.len()
            )));
        }
        if masks_out.len() < label_values {
            return Err(CxrError::Message(format!(
                "mask output buffer too small: need {label_values}, got {}",
                masks_out.len()
            )));
        }
        if samples == 0 {
            return Ok(0);
        }

        let image_offset = start_index
            .checked_mul(self.image_values_per_sample)
            .ok_or_else(|| CxrError::Message("image offset overflow".to_string()))?;
        let label_offset = start_index
            .checked_mul(self.target_count)
            .ok_or_else(|| CxrError::Message("label offset overflow".to_string()))?;
        self.read_image_range(
            self.images_mmap.as_deref(),
            self.images_file.as_deref(),
            &self.images_path,
            image_offset,
            &mut image_out[..image_values],
            "images",
        )?;
        self.read_f32_range(
            self.labels_mmap.as_deref(),
            self.labels_file.as_deref(),
            &self.labels_path,
            label_offset,
            &mut labels_out[..label_values],
            "labels",
        )?;
        self.read_f32_range(
            self.masks_mmap.as_deref(),
            self.masks_file.as_deref(),
            &self.masks_path,
            label_offset,
            &mut masks_out[..label_values],
            "masks",
        )?;
        Ok(samples)
    }

    pub fn fill_indices(
        &self,
        indices: &[usize],
        image_out: &mut [f32],
        labels_out: &mut [f32],
        masks_out: &mut [f32],
    ) -> Result<usize, CxrError> {
        let samples = indices.len();
        let image_values = samples * self.image_values_per_sample;
        let label_values = samples * self.target_count;
        if image_out.len() < image_values {
            return Err(CxrError::Message(format!(
                "image output buffer too small: need {image_values}, got {}",
                image_out.len()
            )));
        }
        if labels_out.len() < label_values {
            return Err(CxrError::Message(format!(
                "label output buffer too small: need {label_values}, got {}",
                labels_out.len()
            )));
        }
        if masks_out.len() < label_values {
            return Err(CxrError::Message(format!(
                "mask output buffer too small: need {label_values}, got {}",
                masks_out.len()
            )));
        }
        if samples == 0 {
            return Ok(0);
        }

        let runs = self.indexed_runs(indices)?;

        if self.read_mode == CxrCacheReadMode::Stream {
            return self
                .fill_indexed_runs_streaming(
                    &runs,
                    self.stream_files()?,
                    CacheSplitBuffers {
                        images: image_out,
                        labels: labels_out,
                        masks: masks_out,
                    },
                )
                .map(|metrics| metrics.samples);
        }

        let max_run_samples = max_indexed_run_samples(&runs);
        let max_image_values = checked_value_count(
            max_run_samples,
            self.image_values_per_sample,
            "image scratch",
        )?;
        let max_label_values =
            checked_value_count(max_run_samples, self.target_count, "label scratch")?;
        let mut image_scratch = vec![0.0f32; max_image_values];
        let mut label_scratch = vec![0.0f32; max_label_values];
        let mut mask_scratch = vec![0.0f32; max_label_values];
        let mut outputs = CacheSplitBuffers {
            images: image_out,
            labels: labels_out,
            masks: masks_out,
        };

        for run in &runs {
            let run_samples = run.out_indices.len();
            let image_values =
                checked_value_count(run_samples, self.image_values_per_sample, "image scratch")?;
            let label_values =
                checked_value_count(run_samples, self.target_count, "label scratch")?;
            let image_offset = run
                .start_sample
                .checked_mul(self.image_values_per_sample)
                .ok_or_else(|| CxrError::Message("image offset overflow".to_string()))?;
            let label_offset = run
                .start_sample
                .checked_mul(self.target_count)
                .ok_or_else(|| CxrError::Message("label offset overflow".to_string()))?;
            self.read_image_range(
                self.images_mmap.as_deref(),
                self.images_file.as_deref(),
                &self.images_path,
                image_offset,
                &mut image_scratch[..image_values],
                "images",
            )?;
            self.read_f32_range(
                self.labels_mmap.as_deref(),
                self.labels_file.as_deref(),
                &self.labels_path,
                label_offset,
                &mut label_scratch[..label_values],
                "labels",
            )?;
            self.read_f32_range(
                self.masks_mmap.as_deref(),
                self.masks_file.as_deref(),
                &self.masks_path,
                label_offset,
                &mut mask_scratch[..label_values],
                "masks",
            )?;

            scatter_indexed_run(
                &run.out_indices,
                &image_scratch[..image_values],
                &label_scratch[..label_values],
                &mask_scratch[..label_values],
                self.image_values_per_sample,
                self.target_count,
                &mut outputs,
            );
        }

        Ok(samples)
    }

    pub fn fill_indices_parallel(
        &self,
        indices: &[usize],
        image_out: &mut [f32],
        labels_out: &mut [f32],
        masks_out: &mut [f32],
        workers: usize,
    ) -> Result<CxrIndexedReadMetrics, CxrError> {
        let samples = indices.len();
        let image_values = samples * self.image_values_per_sample;
        let label_values = samples * self.target_count;
        if image_out.len() < image_values {
            return Err(CxrError::Message(format!(
                "image output buffer too small: need {image_values}, got {}",
                image_out.len()
            )));
        }
        if labels_out.len() < label_values {
            return Err(CxrError::Message(format!(
                "label output buffer too small: need {label_values}, got {}",
                labels_out.len()
            )));
        }
        if masks_out.len() < label_values {
            return Err(CxrError::Message(format!(
                "mask output buffer too small: need {label_values}, got {}",
                masks_out.len()
            )));
        }
        if samples == 0 {
            return Ok(CxrIndexedReadMetrics::default());
        }

        let runs = self.indexed_runs(indices)?;
        let bytes = (image_values + label_values + label_values) * std::mem::size_of::<f32>();
        if self.read_mode == CxrCacheReadMode::Mmap {
            let read_start = Instant::now();
            let samples = self.fill_indices(indices, image_out, labels_out, masks_out)?;
            return Ok(CxrIndexedReadMetrics {
                samples,
                runs: runs.len(),
                workers: 1,
                read_bytes: bytes,
                scatter_bytes: bytes,
                read_micros: read_start.elapsed().as_micros(),
                scatter_micros: 0,
            });
        }
        let worker_count = workers.max(1).min(runs.len().max(1));
        let files = self.stream_files()?;
        if worker_count == 1 {
            return self.fill_indexed_runs_streaming(
                &runs,
                files,
                CacheSplitBuffers {
                    images: image_out,
                    labels: labels_out,
                    masks: masks_out,
                },
            );
        }

        let read_start = Instant::now();
        let chunk_size = runs.len().div_ceil(worker_count);
        let chunk_reads = thread::scope(|scope| {
            let mut handles = Vec::new();
            for chunk in runs.chunks(chunk_size) {
                handles.push(scope.spawn(move || {
                    read_indexed_run_chunk(
                        chunk,
                        files,
                        self.image_dtype,
                        &self.summary.normalization,
                        self.image_values_per_sample,
                        self.target_count,
                    )
                }));
            }

            let mut chunk_reads = Vec::with_capacity(handles.len());
            for handle in handles {
                chunk_reads.push(handle.join().expect("indexed CXR read worker panicked")?);
            }
            Ok::<_, CxrError>(chunk_reads)
        })?;
        let read_micros = read_start.elapsed().as_micros();

        let scatter_start = Instant::now();
        let mut outputs = CacheSplitBuffers {
            images: image_out,
            labels: labels_out,
            masks: masks_out,
        };
        for chunk_read in &chunk_reads {
            for run_read in &chunk_read.runs {
                let run_samples = run_read.out_indices.len();
                let image_values = checked_value_count(
                    run_samples,
                    self.image_values_per_sample,
                    "image scratch",
                )?;
                let label_values =
                    checked_value_count(run_samples, self.target_count, "label scratch")?;
                scatter_indexed_run(
                    &run_read.out_indices,
                    &chunk_read.images[run_read.image_start..run_read.image_start + image_values],
                    &chunk_read.labels[run_read.label_start..run_read.label_start + label_values],
                    &chunk_read.masks[run_read.label_start..run_read.label_start + label_values],
                    self.image_values_per_sample,
                    self.target_count,
                    &mut outputs,
                );
            }
        }
        let scatter_micros = scatter_start.elapsed().as_micros();
        Ok(CxrIndexedReadMetrics {
            samples,
            runs: runs.len(),
            workers: worker_count,
            read_bytes: bytes,
            scatter_bytes: bytes,
            read_micros,
            scatter_micros,
        })
    }

    fn fill_indexed_runs_streaming(
        &self,
        runs: &[CxrIndexedRun],
        files: CacheSplitFiles<'_>,
        mut outputs: CacheSplitBuffers<'_>,
    ) -> Result<CxrIndexedReadMetrics, CxrError> {
        let mut samples = 0usize;
        let mut read_micros = 0u128;
        let mut scatter_micros = 0u128;
        let max_run_samples = max_indexed_run_samples(runs);
        let max_image_values = checked_value_count(
            max_run_samples,
            self.image_values_per_sample,
            "image scratch",
        )?;
        let max_label_values =
            checked_value_count(max_run_samples, self.target_count, "label scratch")?;
        let mut image_scratch = vec![0.0f32; max_image_values];
        let mut label_scratch = vec![0.0f32; max_label_values];
        let mut mask_scratch = vec![0.0f32; max_label_values];
        let mut image_byte_scratch = image_file_byte_scratch(self.image_dtype, max_image_values)?;
        for run in runs {
            samples = samples.checked_add(run.out_indices.len()).ok_or_else(|| {
                CxrError::Message("indexed run sample count overflow".to_string())
            })?;
            let run_samples = run.out_indices.len();
            let image_values =
                checked_value_count(run_samples, self.image_values_per_sample, "image scratch")?;
            let label_values =
                checked_value_count(run_samples, self.target_count, "label scratch")?;
            let image_offset = run
                .start_sample
                .checked_mul(self.image_values_per_sample)
                .ok_or_else(|| CxrError::Message("image offset overflow".to_string()))?;
            let label_offset = run
                .start_sample
                .checked_mul(self.target_count)
                .ok_or_else(|| CxrError::Message("label offset overflow".to_string()))?;
            let read_start = Instant::now();
            read_image_range_from_file_with_scratch(
                files.images,
                self.image_dtype,
                &self.summary.normalization,
                image_offset,
                &mut image_scratch[..image_values],
                &mut image_byte_scratch,
            )?;
            read_f32_range_from_file(
                files.labels,
                label_offset,
                &mut label_scratch[..label_values],
            )?;
            read_f32_range_from_file(files.masks, label_offset, &mut mask_scratch[..label_values])?;
            read_micros += read_start.elapsed().as_micros();

            let scatter_start = Instant::now();
            scatter_indexed_run(
                &run.out_indices,
                &image_scratch[..image_values],
                &label_scratch[..label_values],
                &mask_scratch[..label_values],
                self.image_values_per_sample,
                self.target_count,
                &mut outputs,
            );
            scatter_micros += scatter_start.elapsed().as_micros();
        }
        let image_values = samples * self.image_values_per_sample;
        let label_values = samples * self.target_count;
        let bytes = (image_values + label_values + label_values) * std::mem::size_of::<f32>();
        Ok(CxrIndexedReadMetrics {
            samples,
            runs: runs.len(),
            workers: 1,
            read_bytes: bytes,
            scatter_bytes: bytes,
            read_micros,
            scatter_micros,
        })
    }

    fn read_f32_range(
        &self,
        mmap: Option<&Mmap>,
        file: Option<&File>,
        path: &Path,
        start_value: usize,
        out: &mut [f32],
        kind: &str,
    ) -> Result<(), CxrError> {
        match self.read_mode {
            CxrCacheReadMode::Mmap => {
                let mmap = mmap.ok_or_else(|| {
                    CxrError::Message(format!(
                        "CXR cache {kind} mmap is unavailable for mmap read mode"
                    ))
                })?;
                read_f32_range_from_mmap(mmap, start_value, out)
            }
            CxrCacheReadMode::Stream => {
                let file = file.ok_or_else(|| {
                    CxrError::Message(format!(
                        "CXR cache {kind} file is unavailable for stream read mode: {}",
                        path.display()
                    ))
                })?;
                read_f32_range_from_file(file, start_value, out)
            }
        }
    }

    fn read_image_range(
        &self,
        mmap: Option<&Mmap>,
        file: Option<&File>,
        path: &Path,
        start_value: usize,
        out: &mut [f32],
        kind: &str,
    ) -> Result<(), CxrError> {
        match self.read_mode {
            CxrCacheReadMode::Mmap => {
                let mmap = mmap.ok_or_else(|| {
                    CxrError::Message(format!(
                        "CXR cache {kind} mmap is unavailable for mmap read mode"
                    ))
                })?;
                read_image_range_from_mmap(
                    mmap,
                    self.image_dtype,
                    &self.summary.normalization,
                    start_value,
                    out,
                )
            }
            CxrCacheReadMode::Stream => {
                let file = file.ok_or_else(|| {
                    CxrError::Message(format!(
                        "CXR cache {kind} file is unavailable for stream read mode: {}",
                        path.display()
                    ))
                })?;
                read_image_range_from_file(
                    file,
                    self.image_dtype,
                    &self.summary.normalization,
                    start_value,
                    out,
                )
            }
        }
    }

    fn stream_files(&self) -> Result<CacheSplitFiles<'_>, CxrError> {
        let images = self
            .images_file
            .as_deref()
            .ok_or_else(|| CxrError::Message("CXR image stream file is unavailable".to_string()))?;
        let labels = self
            .labels_file
            .as_deref()
            .ok_or_else(|| CxrError::Message("CXR label stream file is unavailable".to_string()))?;
        let masks = self
            .masks_file
            .as_deref()
            .ok_or_else(|| CxrError::Message("CXR mask stream file is unavailable".to_string()))?;
        Ok(CacheSplitFiles {
            images,
            labels,
            masks,
        })
    }

    fn actual_batch_size(&self, start_index: usize, batch_size: usize) -> usize {
        if start_index >= self.samples() {
            return 0;
        }
        batch_size.min(self.samples() - start_index)
    }

    fn indexed_runs(&self, indices: &[usize]) -> Result<Vec<CxrIndexedRun>, CxrError> {
        let mut order: Vec<(usize, usize)> = indices
            .iter()
            .copied()
            .enumerate()
            .map(|(out_index, sample_index)| (sample_index, out_index))
            .collect();
        for (sample_index, _) in &order {
            if *sample_index >= self.samples() {
                return Err(CxrError::Message(format!(
                    "sample index {sample_index} out of bounds for {} samples",
                    self.samples()
                )));
            }
        }
        order.sort_unstable_by_key(|(sample_index, _)| *sample_index);

        let mut runs = Vec::new();
        let mut cursor = 0usize;
        while cursor < order.len() {
            let start_sample = order[cursor].0;
            let mut end = cursor + 1;
            while end < order.len() && order[end].0 == order[end - 1].0 + 1 {
                end += 1;
            }
            runs.push(CxrIndexedRun {
                start_sample,
                out_indices: order[cursor..end]
                    .iter()
                    .map(|(_sample_index, out_index)| *out_index)
                    .collect(),
            });
            cursor = end;
        }
        Ok(runs)
    }
}

fn max_indexed_run_samples(runs: &[CxrIndexedRun]) -> usize {
    runs.iter()
        .map(|run| run.out_indices.len())
        .max()
        .unwrap_or(0)
}

fn checked_value_count(
    samples: usize,
    values_per_sample: usize,
    kind: &str,
) -> Result<usize, CxrError> {
    samples
        .checked_mul(values_per_sample)
        .ok_or_else(|| CxrError::Message(format!("{kind} value count overflow")))
}

fn scatter_indexed_run(
    out_indices: &[usize],
    image_scratch: &[f32],
    label_scratch: &[f32],
    mask_scratch: &[f32],
    image_values_per_sample: usize,
    target_count: usize,
    outputs: &mut CacheSplitBuffers<'_>,
) {
    for (run_index, out_index) in out_indices.iter().copied().enumerate() {
        let src_image_start = run_index * image_values_per_sample;
        let dst_image_start = out_index * image_values_per_sample;
        outputs.images[dst_image_start..dst_image_start + image_values_per_sample].copy_from_slice(
            &image_scratch[src_image_start..src_image_start + image_values_per_sample],
        );

        let src_label_start = run_index * target_count;
        let dst_label_start = out_index * target_count;
        outputs.labels[dst_label_start..dst_label_start + target_count]
            .copy_from_slice(&label_scratch[src_label_start..src_label_start + target_count]);
        outputs.masks[dst_label_start..dst_label_start + target_count]
            .copy_from_slice(&mask_scratch[src_label_start..src_label_start + target_count]);
    }
}

fn image_file_byte_scratch(dtype: CxrCacheDType, image_values: usize) -> Result<Vec<u8>, CxrError> {
    match dtype {
        CxrCacheDType::Float32 => Ok(Vec::new()),
        CxrCacheDType::Float16 | CxrCacheDType::Uint8 => {
            let byte_len = image_values
                .checked_mul(dtype.bytes_per_value())
                .ok_or_else(|| CxrError::Message("image byte scratch overflow".to_string()))?;
            Ok(vec![0u8; byte_len])
        }
    }
}

fn write_cache_split(
    cache_dir: &Path,
    split: &str,
    records: &[CxrRecord],
    targets: &[String],
    image_size: usize,
    normalization: &Normalization,
    options: &CxrCacheOptions,
) -> Result<CacheSplitSummary, CxrError> {
    let image_dtype = options.cache_dtype;
    let images_name = format!("{split}-images.{}.dat", image_dtype.as_str());
    let labels_name = format!("{split}-labels.float32.dat");
    let masks_name = format!("{split}-masks.float32.dat");
    let metadata_name = format!("{split}-metadata.jsonl");
    let images_path = cache_dir.join(&images_name);
    let labels_path = cache_dir.join(&labels_name);
    let masks_path = cache_dir.join(&masks_name);
    let metadata_path = cache_dir.join(&metadata_name);
    let mut images = BufWriter::new(File::create(&images_path)?);
    let mut labels = BufWriter::new(File::create(&labels_path)?);
    let mut masks = BufWriter::new(File::create(&masks_path)?);
    let mut metadata = BufWriter::new(File::create(&metadata_path)?);

    for record in records {
        write_image_cache_record(&mut images, record, image_size, normalization, options).map_err(
            |error| {
                CxrError::Message(format!(
                    "failed to preprocess sample {}: {error}",
                    record.sample_id
                ))
            },
        )?;
        for target in targets {
            let value = record.labels.get(target).copied().flatten();
            let (label, mask) =
                encode_label_value(value, &options.label_policy, &record.sample_id, target)?;
            labels.write_all(&label.to_le_bytes())?;
            masks.write_all(&mask.to_le_bytes())?;
        }
        serde_json::to_writer(&mut metadata, record)?;
        metadata.write_all(b"\n")?;
    }

    images.flush()?;
    labels.flush()?;
    masks.flush()?;
    metadata.flush()?;
    images.get_ref().sync_all()?;
    labels.get_ref().sync_all()?;
    masks.get_ref().sync_all()?;
    metadata.get_ref().sync_all()?;
    let images_sha256 = hash_file(&images_path)?;
    let labels_sha256 = hash_file(&labels_path)?;
    let masks_sha256 = hash_file(&masks_path)?;
    let metadata_sha256 = hash_file(&metadata_path)?;

    Ok(CacheSplitSummary {
        samples: records.len(),
        shape: [records.len(), 1, image_size, image_size],
        images_path: images_name,
        images_sha256,
        labels_path: labels_name,
        labels_sha256,
        masks_path: masks_name,
        masks_sha256,
        metadata_path: metadata_name,
        metadata_sha256,
    })
}

fn encode_label_value(
    value: Option<i8>,
    policy: &LabelPolicy,
    sample_id: &str,
    target: &str,
) -> Result<(f32, f32), CxrError> {
    match value {
        Some(1) => Ok((1.0, 1.0)),
        Some(0) => Ok((0.0, 1.0)),
        Some(-1) => encode_special_label(policy.uncertain, "uncertain", sample_id, target),
        None => encode_special_label(policy.missing, "missing", sample_id, target),
        Some(other) => Ok((f32::from(other), 1.0)),
    }
}

fn encode_special_label(
    action: crate::types::LabelAction,
    kind: &str,
    sample_id: &str,
    target: &str,
) -> Result<(f32, f32), CxrError> {
    match action {
        crate::types::LabelAction::Ignore => Ok((0.0, 0.0)),
        crate::types::LabelAction::Zero | crate::types::LabelAction::Negative => Ok((0.0, 1.0)),
        crate::types::LabelAction::One | crate::types::LabelAction::Positive => Ok((1.0, 1.0)),
        crate::types::LabelAction::Fail => Err(CxrError::Message(format!(
            "{kind} label for sample {sample_id} target {target} is disallowed by label policy"
        ))),
    }
}

fn estimate_normalization(
    records: &[CxrRecord],
    image_size: usize,
    options: &CxrCacheOptions,
) -> Result<Normalization, CxrError> {
    let mut sum = 0.0f64;
    let mut sq_sum = 0.0f64;
    let mut count = 0usize;
    let stride = (records.len() / 512).max(1);
    for record in records.iter().step_by(stride) {
        let gray = load_resized_luma(record, image_size, options)?;
        for value in gray {
            let scaled = value as f64 / 255.0;
            sum += scaled;
            sq_sum += scaled * scaled;
            count += 1;
        }
    }
    if count == 0 {
        return Ok(Normalization {
            mean: 0.5,
            std: 0.25,
        });
    }
    let mean = sum / count as f64;
    let variance = (sq_sum / count as f64 - mean * mean).max(1.0e-6);
    Ok(Normalization {
        mean: mean as f32,
        std: variance.sqrt().max(1.0e-3) as f32,
    })
}

fn write_image_cache_record(
    writer: &mut impl Write,
    record: &CxrRecord,
    image_size: usize,
    normalization: &Normalization,
    options: &CxrCacheOptions,
) -> Result<(), CxrError> {
    let gray = load_resized_luma(record, image_size, options)?;
    match options.cache_dtype {
        CxrCacheDType::Float32 => {
            for value in normalized_luma_values(gray, normalization) {
                writer.write_all(&value.to_le_bytes())?;
            }
        }
        CxrCacheDType::Float16 => {
            for value in normalized_luma_values(gray, normalization) {
                let half = half::f16::from_f32(value);
                writer.write_all(&half.to_le_bytes())?;
            }
        }
        CxrCacheDType::Uint8 => {
            writer.write_all(&gray)?;
        }
    }
    Ok(())
}

fn normalized_luma_values(
    gray: Vec<u8>,
    normalization: &Normalization,
) -> impl Iterator<Item = f32> + '_ {
    gray.into_iter().map(|value| {
        let scaled = value as f32 / 255.0;
        (scaled - normalization.mean) / normalization.std
    })
}

fn load_resized_luma(
    record: &CxrRecord,
    image_size: usize,
    options: &CxrCacheOptions,
) -> Result<Vec<u8>, CxrError> {
    if is_dicom_record(record) {
        validate_dicom_transfer_syntax(record, &options.transfer_syntax_policy)?;
        load_resized_dicom_luma(
            &record.image_path,
            image_size,
            &options.dicom_presentation_policy,
        )
    } else {
        load_resized_raster_luma(&record.image_path, image_size)
    }
}

fn load_resized_raster_luma(path: &str, image_size: usize) -> Result<Vec<u8>, CxrError> {
    let image = image::open(path)?;
    let gray = match image {
        DynamicImage::ImageLuma8(value) => value,
        other => other.to_luma8(),
    };
    Ok(resize_luma_fit_pad(&gray, image_size, 0)?.into_raw())
}

fn load_resized_dicom_luma(
    path: &str,
    image_size: usize,
    policy: &DicomPresentationPolicy,
) -> Result<Vec<u8>, CxrError> {
    let image =
        medkit_dicom::present_dicom_pixels_with_options(path, dicom_presentation_options(policy)?)?;
    let gray = GrayImage::from_raw(image.width as u32, image.height as u32, image.pixels)
        .ok_or_else(|| CxrError::Message(format!("invalid DICOM raster shape for {path}")))?;
    Ok(resize_luma_fit_pad(&gray, image_size, 0)?.into_raw())
}

pub(crate) fn resize_luma_fit_pad(
    gray: &GrayImage,
    image_size: usize,
    pad_value: u8,
) -> Result<GrayImage, CxrError> {
    if image_size == 0 {
        return Err(CxrError::Message(
            "image size must be greater than zero".to_string(),
        ));
    }
    let (width, height) = gray.dimensions();
    if width == 0 || height == 0 {
        return Err(CxrError::Message(
            "cannot resize an image with zero width or height".to_string(),
        ));
    }
    let image_size_u32 = image_size as u32;
    let scale = (image_size as f32 / width as f32).min(image_size as f32 / height as f32);
    let resized_width = ((width as f32 * scale).round() as u32).clamp(1, image_size_u32);
    let resized_height = ((height as f32 * scale).round() as u32).clamp(1, image_size_u32);
    let resized =
        image::imageops::resize(gray, resized_width, resized_height, FilterType::Triangle);
    let mut canvas = GrayImage::from_pixel(image_size_u32, image_size_u32, Luma([pad_value]));
    let x = ((image_size_u32 - resized_width) / 2) as i64;
    let y = ((image_size_u32 - resized_height) / 2) as i64;
    image::imageops::overlay(&mut canvas, &resized, x, y);
    Ok(canvas)
}

fn validate_dicom_transfer_syntax(
    record: &CxrRecord,
    policy: &TransferSyntaxPolicy,
) -> Result<(), CxrError> {
    if policy.allow_transfer_syntaxes.is_empty() {
        return Ok(());
    }
    let Some(uid) = &record.transfer_syntax_uid else {
        return Ok(());
    };
    if policy
        .allow_transfer_syntaxes
        .iter()
        .any(|allowed| allowed == uid)
    {
        return Ok(());
    }
    match policy.unsupported_transfer_syntax.as_str() {
        "warn" => Ok(()),
        "skip" => Err(CxrError::Message(format!(
            "sample {} has unsupported transfer syntax {uid}; direct cache cannot skip split members",
            record.sample_id
        ))),
        _ => Err(CxrError::Message(format!(
            "sample {} has unsupported transfer syntax {uid}",
            record.sample_id
        ))),
    }
}

fn dicom_presentation_options(
    policy: &DicomPresentationPolicy,
) -> Result<medkit_dicom::DicomPresentationOptions, CxrError> {
    let voi = match policy.voi.as_str() {
        "auto" => medkit_dicom::DicomVoiStrategy::Auto,
        "window" => medkit_dicom::DicomVoiStrategy::Window,
        "minmax" | "min_max" => medkit_dicom::DicomVoiStrategy::MinMax,
        other => {
            return Err(CxrError::Message(format!(
                "unsupported DICOM VOI policy {other:?}; expected auto, window, or minmax"
            )))
        }
    };
    if policy.output != "mono8" {
        return Err(CxrError::Message(format!(
            "unsupported DICOM presentation output {:?}; expected mono8",
            policy.output
        )));
    }
    let decoder = policy
        .decoder_backend
        .parse::<medkit_dicom::DicomDecoderSelection>()
        .map_err(CxrError::Message)?;
    Ok(medkit_dicom::DicomPresentationOptions {
        apply_rescale: policy.apply_rescale,
        voi,
        invert_monochrome1: policy.invert_monochrome1,
        decoder,
    })
}

fn cache_transform_fingerprint(
    transform_plan_hash: &str,
    options: &CxrCacheOptions,
) -> Result<String, CxrError> {
    let text = serde_json::to_string(&(
        transform_plan_hash,
        &options.targets,
        &options.label_policy,
        options.cache_dtype.as_str(),
        &options.image_size_policy,
        &options.dicom_presentation_policy,
        &options.transfer_syntax_policy,
    ))?;
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

fn effective_split_policy(
    options: &CxrCacheOptions,
    split_file: &SplitFile,
) -> SplitPolicyMetadata {
    let default_policy = SplitPolicyMetadata::default();
    if options.split_policy != default_policy {
        return options.split_policy.clone();
    }
    let audit = &split_file.split_audit;
    let ratio = |name: &str, fallback: f64| audit.ratios.get(name).copied().unwrap_or(fallback);
    SplitPolicyMetadata {
        by: audit.by.clone(),
        train: ratio("train", default_policy.train),
        val: ratio("val", default_policy.val),
        test: ratio("test", default_policy.test),
        stratify: audit.stratify.clone(),
        seed: audit.seed,
    }
}

pub(crate) fn image_size_from_plan(plan_path: &Path) -> Result<usize, CxrError> {
    let text = fs::read_to_string(plan_path)?;
    let value = toml::from_str::<toml::Value>(&text)?;
    if let Some(size) = value
        .get("image")
        .and_then(|image| image.get("size"))
        .and_then(|size| size.as_array())
        .and_then(|items| items.first())
        .and_then(|item| item.as_integer())
    {
        return Ok(size as usize);
    }
    if let Some(ops) = value.get("operations").and_then(|ops| ops.as_array()) {
        for op in ops {
            if let Some(size) = op.get("size").and_then(|item| item.as_integer()) {
                return Ok(size as usize);
            }
        }
    }
    Err(CxrError::Message(format!(
        "could not determine image size from plan {}",
        plan_path.display()
    )))
}

pub(crate) fn records_for_split(
    records: &[CxrRecord],
    ids: &[String],
) -> Result<Vec<CxrRecord>, CxrError> {
    let by_id = records
        .iter()
        .map(|record| (record.sample_id.as_str(), record))
        .collect::<HashMap<_, _>>();
    ids.iter()
        .map(|id| {
            by_id
                .get(id.as_str())
                .map(|record| (*record).clone())
                .ok_or_else(|| {
                    CxrError::Message(format!(
                        "split artifact references unknown sample_id {id:?}"
                    ))
                })
        })
        .collect()
}

fn validate_split_membership(
    records: &[CxrRecord],
    split_file: &SplitFile,
) -> Result<(), CxrError> {
    let manifest_ids = records
        .iter()
        .map(|record| record.sample_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();
    let mut duplicates = BTreeSet::new();
    let mut unknown = BTreeSet::new();
    for id in split_file
        .train
        .iter()
        .chain(split_file.val.iter())
        .chain(split_file.test.iter())
    {
        if !manifest_ids.contains(id.as_str()) {
            unknown.insert(id.clone());
        }
        if !seen.insert(id.clone()) {
            duplicates.insert(id.clone());
        }
    }
    if !unknown.is_empty() {
        return Err(CxrError::Message(format!(
            "split artifact references unknown sample IDs: {}",
            unknown.into_iter().collect::<Vec<_>>().join(", ")
        )));
    }
    if !duplicates.is_empty() {
        return Err(CxrError::Message(format!(
            "split artifact assigns samples to multiple splits: {}",
            duplicates.into_iter().collect::<Vec<_>>().join(", ")
        )));
    }
    let missing = manifest_ids
        .into_iter()
        .filter(|id| !seen.contains(*id))
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(CxrError::Message(format!(
            "split artifact omits {} manifest samples; first omitted sample_id {:?}",
            missing.len(),
            missing[0]
        )));
    }
    Ok(())
}
pub(crate) fn read_split_file(path: &Path) -> Result<SplitFile, CxrError> {
    let file = File::open(path)?;
    Ok(serde_json::from_reader(file)?)
}
fn write_cache_validation_report(
    path: &Path,
    summary: &CacheValidationSummary,
) -> Result<(), CxrError> {
    let mut report = String::new();
    report.push_str("# CXR Cache Validation Report\n\n");
    report.push_str(&format!("- status: {}\n", summary.status));
    report.push_str(&format!(
        "- cache schema version: {} (expected {})\n",
        summary.cache_schema_version, summary.expected_cache_schema_version
    ));
    report.push_str(&format!(
        "- report schema version: {}\n",
        summary.report_schema_version
    ));
    report.push_str(&format!(
        "- transform fingerprint: {}\n",
        summary.transform_fingerprint
    ));
    report.push_str(&format!("- recipe hash: {}\n", summary.recipe_hash));
    report.push_str(&format!(
        "- source manifest checksum: {}\n",
        summary.source_manifest_checksum
    ));
    report.push_str(&format!(
        "- checked splits: {}\n",
        summary.checked_splits.join(", ")
    ));
    report.push_str(&format!("- targets: {}\n", summary.targets.join(", ")));
    report.push_str(&format!(
        "- label policy: uncertain={}, missing={}, loss_mask={}\n",
        summary.label_policy.uncertain,
        summary.label_policy.missing,
        summary.label_policy.loss_mask
    ));
    report.push_str("\n## Errors\n\n");
    if summary.errors.is_empty() {
        report.push_str("- none\n");
    } else {
        for error in &summary.errors {
            report.push_str(&format!("- {error}\n"));
        }
    }
    report.push_str("\n## Warnings\n\n");
    if summary.warnings.is_empty() {
        report.push_str("- none\n");
    } else {
        for warning in &summary.warnings {
            report.push_str(&format!("- {warning}\n"));
        }
    }
    fs::write(path, report)?;
    Ok(())
}

fn validate_cache_split_files(
    cache_dir: &Path,
    split: &str,
    split_summary: &CacheSplitSummary,
    image_dtype: CxrCacheDType,
    target_count: usize,
    expected_image_shape: Option<[usize; 4]>,
    errors: &mut Vec<String>,
) -> Result<(), CxrError> {
    if let Some(expected) = expected_image_shape {
        if split_summary.shape != expected {
            errors.push(format!(
                "wrong image shape for split {split}: cache has {:?}, expected {:?}",
                split_summary.shape, expected
            ));
        }
    }
    if split_summary.shape[0] != split_summary.samples {
        errors.push(format!(
            "shape/sample mismatch for split {split}: shape[0]={}, samples={}",
            split_summary.shape[0], split_summary.samples
        ));
    }
    let image_values = split_summary
        .shape
        .iter()
        .try_fold(1usize, |acc, value| acc.checked_mul(*value))
        .ok_or_else(|| CxrError::Message(format!("image shape overflow for split {split}")))?;
    let label_values = split_summary
        .samples
        .checked_mul(target_count)
        .ok_or_else(|| CxrError::Message(format!("label shape overflow for split {split}")))?;
    let images_path = resolve_cache_path(cache_dir, &split_summary.images_path);
    check_file_size(
        &images_path,
        image_values * image_dtype.bytes_per_value(),
        split,
        "images",
        errors,
    )?;
    check_file_hash(
        &images_path,
        &split_summary.images_sha256,
        split,
        "images",
        errors,
    );
    let label_bytes = label_values * std::mem::size_of::<f32>();
    let labels_path = resolve_cache_path(cache_dir, &split_summary.labels_path);
    check_file_size(&labels_path, label_bytes, split, "labels", errors)?;
    check_file_hash(
        &labels_path,
        &split_summary.labels_sha256,
        split,
        "labels",
        errors,
    );
    let masks_path = resolve_cache_path(cache_dir, &split_summary.masks_path);
    check_file_size(&masks_path, label_bytes, split, "masks", errors)?;
    check_file_hash(
        &masks_path,
        &split_summary.masks_sha256,
        split,
        "masks",
        errors,
    );
    let metadata_path = resolve_cache_path(cache_dir, &split_summary.metadata_path);
    match count_jsonl_records(&metadata_path) {
        Ok(count) if count == split_summary.samples => {}
        Ok(count) => errors.push(format!(
            "metadata sample count mismatch for split {split}: metadata has {count}, summary has {}",
            split_summary.samples
        )),
        Err(error) => errors.push(format!(
            "missing or unreadable metadata for split {split}: {error}"
        )),
    }
    check_file_hash(
        &metadata_path,
        &split_summary.metadata_sha256,
        split,
        "metadata",
        errors,
    );
    Ok(())
}

fn check_file_size(
    path: &Path,
    expected_bytes: usize,
    split: &str,
    kind: &str,
    errors: &mut Vec<String>,
) -> Result<(), CxrError> {
    match fs::metadata(path) {
        Ok(metadata) if metadata.len() == expected_bytes as u64 => {}
        Ok(metadata) => errors.push(format!(
            "wrong {kind} file size for split {split}: {} has {} bytes, expected {}",
            path.display(),
            metadata.len(),
            expected_bytes
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => errors.push(format!(
            "missing {kind} file for split {split}: {}",
            path.display()
        )),
        Err(error) => return Err(CxrError::Io(error)),
    }
    Ok(())
}

fn check_file_hash(path: &Path, expected: &str, split: &str, kind: &str, errors: &mut Vec<String>) {
    if expected.is_empty() {
        errors.push(format!(
            "missing {kind} SHA-256 metadata for split {split}: {}",
            path.display()
        ));
        return;
    }
    match hash_file(path) {
        Ok(actual) if actual == expected => {}
        Ok(actual) => errors.push(format!(
            "wrong {kind} SHA-256 for split {split}: {} has {actual}, expected {expected}",
            path.display()
        )),
        Err(error) => errors.push(format!(
            "missing or unreadable {kind} for split {split}: {}: {error}",
            path.display()
        )),
    }
}

fn count_jsonl_records(path: &Path) -> Result<usize, CxrError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut count = 0usize;
    for line in reader.lines() {
        if !line?.trim().is_empty() {
            count += 1;
        }
    }
    Ok(count)
}
fn read_cache_metadata(path: &Path) -> Result<Vec<CxrRecord>, CxrError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        records.push(serde_json::from_str(&line)?);
    }
    Ok(records)
}

fn mmap_file(path: &Path) -> Result<Mmap, CxrError> {
    let file = File::open(path)?;
    // SAFETY: the mapping is read-only and all reads are bounds-checked
    // against the mapped byte slice before decoding values.
    unsafe { Mmap::map(&file).map_err(CxrError::Io) }
}

fn read_f32_range_from_mmap(
    mmap: &Mmap,
    start_value: usize,
    out: &mut [f32],
) -> Result<(), CxrError> {
    let byte_offset = start_value
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| CxrError::Message("cache byte offset overflow".to_string()))?;
    let byte_len = out
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| CxrError::Message("cache byte length overflow".to_string()))?;
    let byte_end = byte_offset
        .checked_add(byte_len)
        .ok_or_else(|| CxrError::Message("cache byte range overflow".to_string()))?;
    let bytes = mmap
        .get(byte_offset..byte_end)
        .ok_or_else(|| CxrError::Message("cache mmap range is out of bounds".to_string()))?;
    for (slot, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(())
}

fn read_image_range_from_mmap(
    mmap: &Mmap,
    dtype: CxrCacheDType,
    normalization: &Normalization,
    start_value: usize,
    out: &mut [f32],
) -> Result<(), CxrError> {
    match dtype {
        CxrCacheDType::Float32 => read_f32_range_from_mmap(mmap, start_value, out),
        CxrCacheDType::Float16 => {
            let bytes = mmap_value_bytes(mmap, start_value, out.len(), dtype.bytes_per_value())?;
            for (slot, chunk) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                *slot = half::f16::from_bits(bits).to_f32();
            }
            Ok(())
        }
        CxrCacheDType::Uint8 => {
            let bytes = mmap_value_bytes(mmap, start_value, out.len(), dtype.bytes_per_value())?;
            decode_uint8_image_values(bytes, normalization, out);
            Ok(())
        }
    }
}

fn mmap_value_bytes(
    mmap: &Mmap,
    start_value: usize,
    values: usize,
    bytes_per_value: usize,
) -> Result<&[u8], CxrError> {
    let byte_offset = byte_offset_for(start_value, bytes_per_value)?;
    let byte_offset = usize::try_from(byte_offset)
        .map_err(|_| CxrError::Message("cache byte offset overflow".to_string()))?;
    let byte_len = values
        .checked_mul(bytes_per_value)
        .ok_or_else(|| CxrError::Message("cache byte length overflow".to_string()))?;
    let byte_end = byte_offset
        .checked_add(byte_len)
        .ok_or_else(|| CxrError::Message("cache byte range overflow".to_string()))?;
    mmap.get(byte_offset..byte_end)
        .ok_or_else(|| CxrError::Message("cache mmap range is out of bounds".to_string()))
}

fn read_indexed_run_chunk(
    runs: &[CxrIndexedRun],
    files: CacheSplitFiles<'_>,
    image_dtype: CxrCacheDType,
    normalization: &Normalization,
    image_values_per_sample: usize,
    target_count: usize,
) -> Result<CxrIndexedChunkRead, CxrError> {
    let total_samples = runs.iter().try_fold(0usize, |total, run| {
        total
            .checked_add(run.out_indices.len())
            .ok_or_else(|| CxrError::Message("indexed run sample count overflow".to_string()))
    })?;
    let total_image_values =
        checked_value_count(total_samples, image_values_per_sample, "image scratch")?;
    let total_label_values = checked_value_count(total_samples, target_count, "label scratch")?;
    let max_image_values = checked_value_count(
        max_indexed_run_samples(runs),
        image_values_per_sample,
        "image scratch",
    )?;
    let mut images = vec![0.0f32; total_image_values];
    let mut labels = vec![0.0f32; total_label_values];
    let mut masks = vec![0.0f32; total_label_values];
    let mut image_byte_scratch = image_file_byte_scratch(image_dtype, max_image_values)?;
    let mut run_reads = Vec::with_capacity(runs.len());
    let mut image_cursor = 0usize;
    let mut label_cursor = 0usize;
    for run in runs {
        let run_len = run.out_indices.len();
        let image_values = checked_value_count(run_len, image_values_per_sample, "image scratch")?;
        let label_values = checked_value_count(run_len, target_count, "label scratch")?;
        let image_offset = run
            .start_sample
            .checked_mul(image_values_per_sample)
            .ok_or_else(|| CxrError::Message("image offset overflow".to_string()))?;
        let label_offset = run
            .start_sample
            .checked_mul(target_count)
            .ok_or_else(|| CxrError::Message("label offset overflow".to_string()))?;
        read_image_range_from_file_with_scratch(
            files.images,
            image_dtype,
            normalization,
            image_offset,
            &mut images[image_cursor..image_cursor + image_values],
            &mut image_byte_scratch,
        )?;
        read_f32_range_from_file(
            files.labels,
            label_offset,
            &mut labels[label_cursor..label_cursor + label_values],
        )?;
        read_f32_range_from_file(
            files.masks,
            label_offset,
            &mut masks[label_cursor..label_cursor + label_values],
        )?;
        run_reads.push(CxrIndexedRunRead {
            out_indices: run.out_indices.clone(),
            image_start: image_cursor,
            label_start: label_cursor,
        });
        image_cursor += image_values;
        label_cursor += label_values;
    }
    Ok(CxrIndexedChunkRead {
        runs: run_reads,
        images,
        labels,
        masks,
    })
}

fn read_f32_range_from_file(
    file: &File,
    start_value: usize,
    out: &mut [f32],
) -> Result<(), CxrError> {
    if cfg!(target_endian = "little") {
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                out.as_mut_ptr().cast::<u8>(),
                std::mem::size_of_val(out),
            )
        };
        read_exact_at(file, bytes, byte_offset(start_value)?)?;
        return Ok(());
    }

    let mut file = file.try_clone()?;
    file.seek(SeekFrom::Start(byte_offset(start_value)?))?;
    let mut bytes = vec![0u8; std::mem::size_of_val(out)];
    file.read_exact(&mut bytes)?;
    for (slot, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(())
}

fn read_image_range_from_file(
    file: &File,
    dtype: CxrCacheDType,
    normalization: &Normalization,
    start_value: usize,
    out: &mut [f32],
) -> Result<(), CxrError> {
    let mut byte_scratch = Vec::new();
    read_image_range_from_file_with_scratch(
        file,
        dtype,
        normalization,
        start_value,
        out,
        &mut byte_scratch,
    )
}

fn read_image_range_from_file_with_scratch(
    file: &File,
    dtype: CxrCacheDType,
    normalization: &Normalization,
    start_value: usize,
    out: &mut [f32],
    byte_scratch: &mut Vec<u8>,
) -> Result<(), CxrError> {
    match dtype {
        CxrCacheDType::Float32 => read_f32_range_from_file(file, start_value, out),
        CxrCacheDType::Float16 => {
            let byte_len = out
                .len()
                .checked_mul(dtype.bytes_per_value())
                .ok_or_else(|| CxrError::Message("cache byte length overflow".to_string()))?;
            if byte_scratch.len() < byte_len {
                byte_scratch.resize(byte_len, 0);
            }
            let bytes = &mut byte_scratch[..byte_len];
            read_exact_at(
                file,
                bytes,
                byte_offset_for(start_value, dtype.bytes_per_value())?,
            )?;
            for (slot, chunk) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                *slot = half::f16::from_bits(bits).to_f32();
            }
            Ok(())
        }
        CxrCacheDType::Uint8 => {
            let byte_len = out.len();
            if byte_scratch.len() < byte_len {
                byte_scratch.resize(byte_len, 0);
            }
            let bytes = &mut byte_scratch[..byte_len];
            read_exact_at(
                file,
                bytes,
                byte_offset_for(start_value, dtype.bytes_per_value())?,
            )?;
            decode_uint8_image_values(bytes, normalization, out);
            Ok(())
        }
    }
}

fn decode_uint8_image_values(values: &[u8], normalization: &Normalization, out: &mut [f32]) {
    debug_assert_eq!(values.len(), out.len());
    if normalization.std == 0.0 {
        for (slot, value) in out.iter_mut().zip(values.iter().copied()) {
            let scaled = value as f32 / 255.0;
            *slot = (scaled - normalization.mean) / normalization.std;
        }
        return;
    }

    let scale = 1.0f32 / (255.0 * normalization.std);
    let bias = -normalization.mean / normalization.std;
    for (slot, value) in out.iter_mut().zip(values.iter().copied()) {
        *slot = value as f32 * scale + bias;
    }
}

#[cfg(unix)]
fn read_exact_at(file: &File, mut buffer: &mut [u8], mut offset: u64) -> Result<(), CxrError> {
    while !buffer.is_empty() {
        let read = file.read_at(buffer, offset)?;
        if read == 0 {
            return Err(CxrError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short positioned read from CXR cache",
            )));
        }
        offset = offset
            .checked_add(read as u64)
            .ok_or_else(|| CxrError::Message("positioned read offset overflow".to_string()))?;
        let (_, rest) = buffer.split_at_mut(read);
        buffer = rest;
    }
    Ok(())
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut buffer: &mut [u8], mut offset: u64) -> Result<(), CxrError> {
    while !buffer.is_empty() {
        let read = file.seek_read(buffer, offset)?;
        if read == 0 {
            return Err(CxrError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short positioned read from CXR cache",
            )));
        }
        offset = offset
            .checked_add(read as u64)
            .ok_or_else(|| CxrError::Message("positioned read offset overflow".to_string()))?;
        let (_, rest) = buffer.split_at_mut(read);
        buffer = rest;
    }
    Ok(())
}

fn byte_offset(value_index: usize) -> Result<u64, CxrError> {
    byte_offset_for(value_index, std::mem::size_of::<f32>())
}

fn byte_offset_for(value_index: usize, bytes_per_value: usize) -> Result<u64, CxrError> {
    value_index
        .checked_mul(bytes_per_value)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| CxrError::Message("cache byte offset overflow".to_string()))
}
