#![forbid(unsafe_code)]

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    thread,
    time::Instant,
};

use flate2::read::MultiGzDecoder;
use image::{imageops::FilterType, DynamicImage};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const CXR_CACHE_SCHEMA_VERSION: u32 = 1;
pub const CXR_REPORT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct IndexConfig {
    pub images_root: PathBuf,
    pub metadata_path: Option<PathBuf>,
    pub labels_path: Option<PathBuf>,
    pub reports_root: Option<PathBuf>,
    pub out_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ValidateConfig {
    pub manifest_path: PathBuf,
    pub require_frontal: bool,
    pub check_patient_leakage: bool,
    pub check_duplicates: bool,
    pub report_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct SplitConfig {
    pub manifest_path: PathBuf,
    pub by: String,
    pub train: f64,
    pub val: f64,
    pub test: f64,
    pub stratify: Vec<String>,
    pub out_path: PathBuf,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub manifest_path: PathBuf,
    pub splits_path: PathBuf,
    pub plan_path: PathBuf,
    pub cache_dir: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ValidateCacheConfig {
    pub cache_dir: PathBuf,
    pub split: Option<String>,
    pub expected_targets: Option<Vec<String>>,
    pub expected_image_shape: Option<[usize; 4]>,
    pub plan_path: Option<PathBuf>,
    pub report_path: Option<PathBuf>,
    pub json_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CxrRecord {
    pub sample_id: String,
    pub patient_id: String,
    pub study_id: String,
    pub image_id: String,
    pub image_path: String,
    pub source_format: String,
    pub modality: Option<String>,
    pub view_position: Option<String>,
    pub laterality: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub photometric_interpretation: Option<String>,
    pub labels: BTreeMap<String, Option<i8>>,
    pub label_source: Option<String>,
    pub report_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexSummary {
    pub images_root: String,
    pub metadata_path: Option<String>,
    pub labels_path: Option<String>,
    pub records: usize,
    pub patients: usize,
    pub studies: usize,
    pub labels: BTreeMap<String, LabelCount>,
    pub out_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LabelCount {
    pub positive: usize,
    pub negative: usize,
    pub uncertain: usize,
    pub missing: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    pub records: usize,
    pub readable_images: usize,
    pub unreadable_images: usize,
    pub filtered_non_frontal: usize,
    pub patient_overlap_count: usize,
    pub duplicate_hash_overlap_count: usize,
    pub split_counts: BTreeMap<String, usize>,
    pub target_counts: BTreeMap<String, LabelCount>,
    pub report_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitSummary {
    pub counts: BTreeMap<String, usize>,
    pub patient_counts: BTreeMap<String, usize>,
    pub by: String,
    pub ratios: BTreeMap<String, f64>,
    pub stratify: Vec<String>,
    pub patient_overlap_count: usize,
    pub out_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitFile {
    pub train: Vec<String>,
    pub val: Vec<String>,
    pub test: Vec<String>,
    pub split_audit: SplitSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheSummary {
    #[serde(default)]
    pub cache_schema_version: u32,
    #[serde(default)]
    pub report_schema_version: u32,
    pub cache_dir: String,
    pub image_size: usize,
    pub channels: usize,
    pub dtype: String,
    pub targets: Vec<String>,
    #[serde(default = "default_label_policy")]
    pub label_policy: LabelPolicy,
    pub normalization: Normalization,
    pub transform_plan_hash: String,
    #[serde(default)]
    pub transform_fingerprint: String,
    #[serde(default)]
    pub source_manifest_checksum: String,
    #[serde(default)]
    pub split_names: Vec<String>,
    #[serde(default)]
    pub image_size_policy: ImageSizePolicy,
    pub splits: BTreeMap<String, CacheSplitSummary>,
    pub failed_samples: Vec<String>,
    pub cache_size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LabelPolicy {
    pub positive: String,
    pub negative: String,
    pub uncertain: String,
    pub missing: String,
    pub loss_mask: String,
}

impl Default for LabelPolicy {
    fn default() -> Self {
        Self {
            positive: "label=1 mask=1".to_string(),
            negative: "label=0 mask=1".to_string(),
            uncertain: "ignore".to_string(),
            missing: "ignore".to_string(),
            loss_mask: "uncertain and missing labels are masked from loss".to_string(),
        }
    }
}

fn default_label_policy() -> LabelPolicy {
    LabelPolicy::default()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageSizePolicy {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub dtype: String,
    pub transform: String,
}

impl Default for ImageSizePolicy {
    fn default() -> Self {
        Self {
            channels: 0,
            height: 0,
            width: 0,
            dtype: String::new(),
            transform: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Normalization {
    pub mean: f32,
    pub std: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheSplitSummary {
    pub samples: usize,
    pub shape: [usize; 4],
    pub images_path: String,
    pub labels_path: String,
    pub masks_path: String,
    pub metadata_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheValidationSummary {
    pub cache_dir: String,
    pub cache_schema_version: u32,
    pub expected_cache_schema_version: u32,
    pub report_schema_version: u32,
    pub status: String,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub targets: Vec<String>,
    pub label_policy: LabelPolicy,
    pub split_names: Vec<String>,
    pub checked_splits: Vec<String>,
    pub image_size_policy: ImageSizePolicy,
    pub transform_fingerprint: String,
    pub source_manifest_checksum: String,
    pub cache_size_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct CxrCacheReader {
    cache_dir: PathBuf,
    split: String,
    summary: CacheSummary,
    split_summary: CacheSplitSummary,
    records: Vec<CxrRecord>,
    image_values_per_sample: usize,
    target_count: usize,
}

#[derive(Debug, Clone)]
pub struct CxrCacheBatch {
    pub samples: usize,
    pub image_shape: [usize; 4],
    pub labels_shape: [usize; 2],
    pub images: Vec<f32>,
    pub labels: Vec<f32>,
    pub masks: Vec<f32>,
    pub records: Vec<CxrRecord>,
}

#[derive(Debug, Clone, Default)]
pub struct CxrIndexedReadMetrics {
    pub samples: usize,
    pub runs: usize,
    pub workers: usize,
    pub read_bytes: usize,
    pub scatter_bytes: usize,
    pub read_micros: u128,
    pub scatter_micros: u128,
}

#[derive(Debug, Clone)]
struct CxrIndexedRun {
    start_sample: usize,
    out_indices: Vec<usize>,
}

#[derive(Debug)]
struct CxrIndexedRunRead {
    out_indices: Vec<usize>,
    images: Vec<f32>,
    labels: Vec<f32>,
    masks: Vec<f32>,
}

pub fn index_cxr(config: &IndexConfig) -> Result<IndexSummary, CxrError> {
    let image_map = scan_images(&config.images_root)?;
    let label_map = match &config.labels_path {
        Some(path) => read_label_csv(path)?,
        None => HashMap::new(),
    };
    let mut records = match &config.metadata_path {
        Some(path) => records_from_metadata(path, &image_map, &label_map, config)?,
        None => records_from_images(&image_map, config)?,
    };
    records.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));
    write_manifest(&config.out_path, &records)?;
    let summary = index_summary(config, &records);
    Ok(summary)
}

pub fn validate_cxr(config: &ValidateConfig) -> Result<ValidationSummary, CxrError> {
    let records = read_manifest(&config.manifest_path)?;
    let mut readable_images = 0usize;
    let mut unreadable_images = 0usize;
    let mut filtered_non_frontal = 0usize;
    let mut split_counts = BTreeMap::new();
    let mut split_patients: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut split_hashes: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut target_counts = BTreeMap::new();

    for record in &records {
        if config.require_frontal && !is_frontal(record.view_position.as_deref()) {
            filtered_non_frontal += 1;
            continue;
        }
        if image::image_dimensions(&record.image_path).is_ok() {
            readable_images += 1;
        } else {
            unreadable_images += 1;
        }
        for (target, value) in &record.labels {
            add_label_count(target_counts.entry(target.clone()).or_default(), *value);
        }
        if let Some(split) = &record.split {
            *split_counts.entry(split.clone()).or_insert(0) += 1;
            split_patients
                .entry(split.clone())
                .or_default()
                .insert(record.patient_id.clone());
            if config.check_duplicates {
                let hash = match &record.sha256 {
                    Some(value) => value.clone(),
                    None => hash_file(Path::new(&record.image_path))?,
                };
                split_hashes.entry(split.clone()).or_default().insert(hash);
            }
        }
    }

    let patient_overlap_count = if config.check_patient_leakage {
        overlap_count(&split_patients)
    } else {
        0
    };
    let duplicate_hash_overlap_count = if config.check_duplicates {
        overlap_count(&split_hashes)
    } else {
        0
    };

    let summary = ValidationSummary {
        records: records.len(),
        readable_images,
        unreadable_images,
        filtered_non_frontal,
        patient_overlap_count,
        duplicate_hash_overlap_count,
        split_counts,
        target_counts,
        report_path: config.report_path.display().to_string(),
    };
    write_validation_report(&config.report_path, &summary)?;
    Ok(summary)
}

pub fn split_cxr(config: &SplitConfig) -> Result<SplitSummary, CxrError> {
    if config.by != "patient_id" && config.by != "patient" {
        return Err(CxrError::Message(format!(
            "only patient-level CXR splits are supported, got --by {}",
            config.by
        )));
    }
    let mut records = read_manifest(&config.manifest_path)?;
    let mut by_patient: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, record) in records.iter().enumerate() {
        by_patient
            .entry(record.patient_id.clone())
            .or_default()
            .push(index);
    }
    let mut groups = by_patient.into_iter().collect::<Vec<_>>();
    groups.sort_by_key(|(patient, _)| stable_bucket(patient, config.seed));

    let total = records.len().max(1);
    let train_target = (config.train * total as f64).round() as usize;
    let val_target = (config.val * total as f64).round() as usize;
    let mut train = Vec::new();
    let mut val = Vec::new();
    let mut test = Vec::new();
    let mut patient_counts = BTreeMap::from([
        ("train".to_string(), 0usize),
        ("val".to_string(), 0usize),
        ("test".to_string(), 0usize),
    ]);

    for (_patient, indices) in groups {
        let target_split = if train.len() < train_target {
            "train"
        } else if val.len() < val_target {
            "val"
        } else {
            "test"
        };
        *patient_counts.entry(target_split.to_string()).or_insert(0) += 1;
        for index in indices {
            records[index].split = Some(target_split.to_string());
            match target_split {
                "train" => train.push(records[index].sample_id.clone()),
                "val" => val.push(records[index].sample_id.clone()),
                _ => test.push(records[index].sample_id.clone()),
            }
        }
    }

    let counts = BTreeMap::from([
        ("train".to_string(), train.len()),
        ("val".to_string(), val.len()),
        ("test".to_string(), test.len()),
    ]);
    let ratios = BTreeMap::from([
        ("train".to_string(), config.train),
        ("val".to_string(), config.val),
        ("test".to_string(), config.test),
    ]);
    let summary = SplitSummary {
        counts,
        patient_counts,
        by: "patient_id".to_string(),
        ratios,
        stratify: config.stratify.clone(),
        patient_overlap_count: 0,
        out_path: config.out_path.display().to_string(),
    };
    let split_file = SplitFile {
        train,
        val,
        test,
        split_audit: summary.clone(),
    };
    write_json(&config.out_path, &split_file)?;
    write_manifest(&config.manifest_path, &records)?;
    Ok(summary)
}

pub fn cache_cxr(config: &CacheConfig) -> Result<CacheSummary, CxrError> {
    let records = read_manifest(&config.manifest_path)?;
    let split_file = read_split_file(&config.splits_path)?;
    validate_split_membership(&records, &split_file)?;
    let image_size = image_size_from_plan(&config.plan_path)?;
    fs::create_dir_all(&config.cache_dir)?;
    let targets = collect_targets(&records);
    let transform_plan_hash = hash_file(&config.plan_path)?;
    let train_records = records_for_split(&records, &split_file.train)?;
    let normalization = estimate_normalization(&train_records, image_size)?;
    let mut splits = BTreeMap::new();
    let mut failed_samples = Vec::new();

    for (name, ids) in [
        ("train", &split_file.train),
        ("val", &split_file.val),
        ("test", &split_file.test),
    ] {
        let split_records = records_for_split(&records, ids)?;
        let split_summary = write_cache_split(
            &config.cache_dir,
            name,
            &split_records,
            &targets,
            image_size,
            &normalization,
            &mut failed_samples,
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
        dtype: "float32".to_string(),
        targets,
        label_policy: LabelPolicy::default(),
        normalization,
        transform_fingerprint: transform_plan_hash.clone(),
        transform_plan_hash,
        source_manifest_checksum: hash_file(&config.manifest_path)?,
        split_names,
        image_size_policy: ImageSizePolicy {
            channels: 1,
            height: image_size,
            width: image_size,
            dtype: "float32".to_string(),
            transform: "decode grayscale, resize square, normalize dataset mean/std".to_string(),
        },
        splits,
        failed_samples,
        cache_size_bytes: directory_size(&config.cache_dir)?,
    };
    write_json(&config.cache_dir.join("cache-metadata.json"), &summary)?;
    Ok(summary)
}

pub fn read_cache_summary(cache_dir: &Path) -> Result<CacheSummary, CxrError> {
    let text = fs::read_to_string(cache_dir.join("cache-metadata.json"))?;
    Ok(serde_json::from_str(&text)?)
}

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

    for split in &checked_splits {
        let split_summary = summary
            .splits
            .get(split)
            .expect("checked split names are drawn from cache metadata");
        let result = validate_cache_split_files(
            &config.cache_dir,
            split,
            split_summary,
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
        let mut file_errors = Vec::new();
        validate_cache_split_files(
            &cache_dir,
            &split,
            &split_summary,
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
        Ok(Self {
            cache_dir,
            split,
            summary,
            split_summary,
            records,
            image_values_per_sample,
            target_count,
        })
    }

    pub fn split(&self) -> &str {
        &self.split
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
        read_f32_range(
            &resolve_cache_path(&self.cache_dir, &self.split_summary.images_path),
            image_offset,
            &mut image_out[..image_values],
        )?;
        read_f32_range(
            &resolve_cache_path(&self.cache_dir, &self.split_summary.labels_path),
            label_offset,
            &mut labels_out[..label_values],
        )?;
        read_f32_range(
            &resolve_cache_path(&self.cache_dir, &self.split_summary.masks_path),
            label_offset,
            &mut masks_out[..label_values],
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

        let image_path = resolve_cache_path(&self.cache_dir, &self.split_summary.images_path);
        let labels_path = resolve_cache_path(&self.cache_dir, &self.split_summary.labels_path);
        let masks_path = resolve_cache_path(&self.cache_dir, &self.split_summary.masks_path);
        let mut images = File::open(image_path)?;
        let mut labels = File::open(labels_path)?;
        let mut masks = File::open(masks_path)?;

        let mut cursor = 0usize;
        while cursor < order.len() {
            let start_sample = order[cursor].0;
            let mut end = cursor + 1;
            while end < order.len() && order[end].0 == order[end - 1].0 + 1 {
                end += 1;
            }
            let run_samples = end - cursor;
            let image_offset = start_sample
                .checked_mul(self.image_values_per_sample)
                .ok_or_else(|| CxrError::Message("image offset overflow".to_string()))?;
            let label_offset = start_sample
                .checked_mul(self.target_count)
                .ok_or_else(|| CxrError::Message("label offset overflow".to_string()))?;
            let mut image_scratch = vec![0.0f32; run_samples * self.image_values_per_sample];
            let mut label_scratch = vec![0.0f32; run_samples * self.target_count];
            let mut mask_scratch = vec![0.0f32; run_samples * self.target_count];
            read_f32_range_from(&mut images, image_offset, &mut image_scratch)?;
            read_f32_range_from(&mut labels, label_offset, &mut label_scratch)?;
            read_f32_range_from(&mut masks, label_offset, &mut mask_scratch)?;

            for (run_index, (_sample_index, out_index)) in order[cursor..end].iter().enumerate() {
                let src_image_start = run_index * self.image_values_per_sample;
                let dst_image_start = *out_index * self.image_values_per_sample;
                image_out[dst_image_start..dst_image_start + self.image_values_per_sample]
                    .copy_from_slice(
                        &image_scratch
                            [src_image_start..src_image_start + self.image_values_per_sample],
                    );
                let src_label_start = run_index * self.target_count;
                let dst_label_start = *out_index * self.target_count;
                labels_out[dst_label_start..dst_label_start + self.target_count].copy_from_slice(
                    &label_scratch[src_label_start..src_label_start + self.target_count],
                );
                masks_out[dst_label_start..dst_label_start + self.target_count].copy_from_slice(
                    &mask_scratch[src_label_start..src_label_start + self.target_count],
                );
            }
            cursor = end;
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
        let worker_count = workers.max(1).min(runs.len().max(1));
        let image_path = resolve_cache_path(&self.cache_dir, &self.split_summary.images_path);
        let labels_path = resolve_cache_path(&self.cache_dir, &self.split_summary.labels_path);
        let masks_path = resolve_cache_path(&self.cache_dir, &self.split_summary.masks_path);
        if worker_count == 1 {
            return self.fill_indexed_runs_streaming(
                &runs,
                &image_path,
                &labels_path,
                &masks_path,
                image_out,
                labels_out,
                masks_out,
            );
        }

        let read_start = Instant::now();
        let chunk_size = runs.len().div_ceil(worker_count);
        let run_reads = thread::scope(|scope| {
            let mut handles = Vec::new();
            for chunk in runs.chunks(chunk_size) {
                let image_path = &image_path;
                let labels_path = &labels_path;
                let masks_path = &masks_path;
                handles.push(scope.spawn(move || {
                    read_indexed_runs(
                        chunk,
                        image_path,
                        labels_path,
                        masks_path,
                        self.image_values_per_sample,
                        self.target_count,
                    )
                }));
            }

            let mut run_reads = Vec::with_capacity(runs.len());
            for handle in handles {
                let mut chunk_reads = handle.join().expect("indexed CXR read worker panicked")?;
                run_reads.append(&mut chunk_reads);
            }
            Ok::<_, CxrError>(run_reads)
        })?;
        let read_micros = read_start.elapsed().as_micros();

        let scatter_start = Instant::now();
        for run_read in &run_reads {
            for (run_index, out_index) in run_read.out_indices.iter().copied().enumerate() {
                let src_image_start = run_index * self.image_values_per_sample;
                let dst_image_start = out_index * self.image_values_per_sample;
                image_out[dst_image_start..dst_image_start + self.image_values_per_sample]
                    .copy_from_slice(
                        &run_read.images
                            [src_image_start..src_image_start + self.image_values_per_sample],
                    );
                let src_label_start = run_index * self.target_count;
                let dst_label_start = out_index * self.target_count;
                labels_out[dst_label_start..dst_label_start + self.target_count].copy_from_slice(
                    &run_read.labels[src_label_start..src_label_start + self.target_count],
                );
                masks_out[dst_label_start..dst_label_start + self.target_count].copy_from_slice(
                    &run_read.masks[src_label_start..src_label_start + self.target_count],
                );
            }
        }
        let scatter_micros = scatter_start.elapsed().as_micros();
        let bytes = (image_values + label_values + label_values) * std::mem::size_of::<f32>();
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
        image_path: &Path,
        labels_path: &Path,
        masks_path: &Path,
        image_out: &mut [f32],
        labels_out: &mut [f32],
        masks_out: &mut [f32],
    ) -> Result<CxrIndexedReadMetrics, CxrError> {
        let mut images = File::open(image_path)?;
        let mut labels = File::open(labels_path)?;
        let mut masks = File::open(masks_path)?;
        let mut samples = 0usize;
        let mut read_micros = 0u128;
        let mut scatter_micros = 0u128;
        for run in runs {
            samples += run.out_indices.len();
            let image_offset = run
                .start_sample
                .checked_mul(self.image_values_per_sample)
                .ok_or_else(|| CxrError::Message("image offset overflow".to_string()))?;
            let label_offset = run
                .start_sample
                .checked_mul(self.target_count)
                .ok_or_else(|| CxrError::Message("label offset overflow".to_string()))?;
            let mut image_scratch =
                vec![0.0f32; run.out_indices.len() * self.image_values_per_sample];
            let mut label_scratch = vec![0.0f32; run.out_indices.len() * self.target_count];
            let mut mask_scratch = vec![0.0f32; run.out_indices.len() * self.target_count];
            let read_start = Instant::now();
            read_f32_range_from(&mut images, image_offset, &mut image_scratch)?;
            read_f32_range_from(&mut labels, label_offset, &mut label_scratch)?;
            read_f32_range_from(&mut masks, label_offset, &mut mask_scratch)?;
            read_micros += read_start.elapsed().as_micros();

            let scatter_start = Instant::now();
            for (run_index, out_index) in run.out_indices.iter().copied().enumerate() {
                let src_image_start = run_index * self.image_values_per_sample;
                let dst_image_start = out_index * self.image_values_per_sample;
                image_out[dst_image_start..dst_image_start + self.image_values_per_sample]
                    .copy_from_slice(
                        &image_scratch
                            [src_image_start..src_image_start + self.image_values_per_sample],
                    );
                let src_label_start = run_index * self.target_count;
                let dst_label_start = out_index * self.target_count;
                labels_out[dst_label_start..dst_label_start + self.target_count].copy_from_slice(
                    &label_scratch[src_label_start..src_label_start + self.target_count],
                );
                masks_out[dst_label_start..dst_label_start + self.target_count].copy_from_slice(
                    &mask_scratch[src_label_start..src_label_start + self.target_count],
                );
            }
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

fn records_from_metadata(
    metadata_path: &Path,
    image_map: &HashMap<String, PathBuf>,
    label_map: &HashMap<(String, String), BTreeMap<String, Option<i8>>>,
    config: &IndexConfig,
) -> Result<Vec<CxrRecord>, CxrError> {
    let mut reader = csv_reader(metadata_path)?;
    let headers = HeaderIndex::new(reader.headers()?);
    let mut records = Vec::new();
    for row in reader.records() {
        let row = row?;
        let image_id = headers
            .get(&row, &["dicom_id", "image_id", "filename"])
            .ok_or_else(|| CxrError::Message("metadata row missing dicom_id".to_string()))?;
        let subject = headers
            .get(&row, &["subject_id", "patient_id"])
            .unwrap_or_else(|| patient_from_filename(&image_id));
        let study = headers
            .get(&row, &["study_id"])
            .unwrap_or_else(|| "unknown-study".to_string());
        let Some(path) = image_map.get(&image_id) else {
            continue;
        };
        let (width, height) = metadata_or_image_dimensions(&headers, &row, path)?;
        let labels = label_map
            .get(&(subject.clone(), study.clone()))
            .cloned()
            .unwrap_or_default();
        let report_path = config
            .reports_root
            .as_ref()
            .map(|root| root.join(format!("s{study}.txt")).display().to_string());
        records.push(CxrRecord {
            sample_id: format!("p{subject}/s{study}/{image_id}"),
            patient_id: format!("p{subject}"),
            study_id: format!("s{study}"),
            image_id: image_id.clone(),
            image_path: path.display().to_string(),
            source_format: path
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or("unknown")
                .to_ascii_lowercase(),
            modality: headers.get(&row, &["modality"]),
            view_position: headers.get(&row, &["ViewPosition", "view_position"]),
            laterality: headers.get(&row, &["laterality", "Laterality"]),
            width: Some(width),
            height: Some(height),
            photometric_interpretation: Some("MONOCHROME2".to_string()),
            labels,
            label_source: Some("chexpert_csv".to_string()),
            report_path,
            split: None,
            sha256: Some(hash_file(path)?),
        });
    }
    Ok(records)
}

fn records_from_images(
    image_map: &HashMap<String, PathBuf>,
    _config: &IndexConfig,
) -> Result<Vec<CxrRecord>, CxrError> {
    let mut records = Vec::new();
    for (image_id, path) in image_map {
        let (width, height) = image::image_dimensions(path)?;
        let patient = patient_from_filename(image_id);
        records.push(CxrRecord {
            sample_id: format!("{patient}/{image_id}"),
            patient_id: patient.clone(),
            study_id: patient,
            image_id: image_id.clone(),
            image_path: path.display().to_string(),
            source_format: path
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or("unknown")
                .to_ascii_lowercase(),
            modality: Some("CR".to_string()),
            view_position: None,
            laterality: None,
            width: Some(width),
            height: Some(height),
            photometric_interpretation: Some("MONOCHROME2".to_string()),
            labels: BTreeMap::new(),
            label_source: None,
            report_path: None,
            split: None,
            sha256: Some(hash_file(path)?),
        });
    }
    Ok(records)
}

fn read_label_csv(
    labels_path: &Path,
) -> Result<HashMap<(String, String), BTreeMap<String, Option<i8>>>, CxrError> {
    let mut reader = csv_reader(labels_path)?;
    let headers = reader.headers()?.clone();
    let index = HeaderIndex::new(&headers);
    let mut map = HashMap::new();
    for row in reader.records() {
        let row = row?;
        let subject = index
            .get(&row, &["subject_id", "patient_id"])
            .ok_or_else(|| CxrError::Message("labels row missing subject_id".to_string()))?;
        let study = index
            .get(&row, &["study_id"])
            .ok_or_else(|| CxrError::Message("labels row missing study_id".to_string()))?;
        let mut labels = BTreeMap::new();
        for (header_index, header) in headers.iter().enumerate() {
            if matches!(header, "subject_id" | "study_id" | "patient_id") {
                continue;
            }
            let value = row.get(header_index).unwrap_or("").trim();
            labels.insert(header.to_string(), parse_label_value(value));
        }
        map.insert((subject, study), labels);
    }
    Ok(map)
}

fn write_cache_split(
    cache_dir: &Path,
    split: &str,
    records: &[CxrRecord],
    targets: &[String],
    image_size: usize,
    normalization: &Normalization,
    failed_samples: &mut Vec<String>,
) -> Result<CacheSplitSummary, CxrError> {
    let images_path = cache_dir.join(format!("{split}-images.float32.dat"));
    let labels_path = cache_dir.join(format!("{split}-labels.float32.dat"));
    let masks_path = cache_dir.join(format!("{split}-masks.float32.dat"));
    let metadata_path = cache_dir.join(format!("{split}-metadata.jsonl"));
    let mut images = BufWriter::new(File::create(&images_path)?);
    let mut labels = BufWriter::new(File::create(&labels_path)?);
    let mut masks = BufWriter::new(File::create(&masks_path)?);
    let mut metadata = BufWriter::new(File::create(&metadata_path)?);

    for record in records {
        match preprocess_image(&record.image_path, image_size, normalization) {
            Ok(values) => {
                for value in values {
                    images.write_all(&value.to_le_bytes())?;
                }
            }
            Err(error) => {
                failed_samples.push(format!("{}: {error}", record.sample_id));
                for _ in 0..(image_size * image_size) {
                    images.write_all(&0.0f32.to_le_bytes())?;
                }
            }
        }
        for target in targets {
            let value = record.labels.get(target).copied().flatten();
            let (label, mask) = match value {
                Some(1) => (1.0f32, 1.0f32),
                Some(0) => (0.0f32, 1.0f32),
                Some(-1) | None => (0.0f32, 0.0f32),
                Some(other) => (other as f32, 1.0f32),
            };
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

    Ok(CacheSplitSummary {
        samples: records.len(),
        shape: [records.len(), 1, image_size, image_size],
        images_path: images_path.display().to_string(),
        labels_path: labels_path.display().to_string(),
        masks_path: masks_path.display().to_string(),
        metadata_path: metadata_path.display().to_string(),
    })
}

fn estimate_normalization(
    records: &[CxrRecord],
    image_size: usize,
) -> Result<Normalization, CxrError> {
    let mut sum = 0.0f64;
    let mut sq_sum = 0.0f64;
    let mut count = 0usize;
    let stride = (records.len() / 512).max(1);
    for record in records.iter().step_by(stride) {
        let gray = load_resized_luma(&record.image_path, image_size)?;
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

fn preprocess_image(
    path: &str,
    image_size: usize,
    normalization: &Normalization,
) -> Result<Vec<f32>, CxrError> {
    let gray = load_resized_luma(path, image_size)?;
    Ok(gray
        .into_iter()
        .map(|value| {
            let scaled = value as f32 / 255.0;
            (scaled - normalization.mean) / normalization.std
        })
        .collect())
}

fn load_resized_luma(path: &str, image_size: usize) -> Result<Vec<u8>, CxrError> {
    let image = image::open(path)?;
    let gray = match image {
        DynamicImage::ImageLuma8(value) => value,
        other => other.to_luma8(),
    };
    let resized = image::imageops::resize(
        &gray,
        image_size as u32,
        image_size as u32,
        FilterType::Triangle,
    );
    Ok(resized.into_raw())
}

fn image_size_from_plan(plan_path: &Path) -> Result<usize, CxrError> {
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

fn records_for_split(records: &[CxrRecord], ids: &[String]) -> Result<Vec<CxrRecord>, CxrError> {
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

fn collect_targets(records: &[CxrRecord]) -> Vec<String> {
    let mut targets = BTreeSet::new();
    for record in records {
        targets.extend(record.labels.keys().cloned());
    }
    targets.into_iter().collect()
}

fn read_split_file(path: &Path) -> Result<SplitFile, CxrError> {
    let file = File::open(path)?;
    Ok(serde_json::from_reader(file)?)
}

fn index_summary(config: &IndexConfig, records: &[CxrRecord]) -> IndexSummary {
    let mut labels = BTreeMap::new();
    for record in records {
        for (target, value) in &record.labels {
            add_label_count(labels.entry(target.clone()).or_default(), *value);
        }
    }
    IndexSummary {
        images_root: config.images_root.display().to_string(),
        metadata_path: config
            .metadata_path
            .as_ref()
            .map(|path| path.display().to_string()),
        labels_path: config
            .labels_path
            .as_ref()
            .map(|path| path.display().to_string()),
        records: records.len(),
        patients: records
            .iter()
            .map(|record| &record.patient_id)
            .collect::<BTreeSet<_>>()
            .len(),
        studies: records
            .iter()
            .map(|record| &record.study_id)
            .collect::<BTreeSet<_>>()
            .len(),
        labels,
        out_path: config.out_path.display().to_string(),
    }
}

fn write_validation_report(path: &Path, summary: &ValidationSummary) -> Result<(), CxrError> {
    let mut report = String::new();
    report.push_str("# CXR Validation Report\n\n");
    report.push_str(&format!("- records: {}\n", summary.records));
    report.push_str(&format!("- readable images: {}\n", summary.readable_images));
    report.push_str(&format!(
        "- unreadable images: {}\n",
        summary.unreadable_images
    ));
    report.push_str(&format!(
        "- filtered non-frontal: {}\n",
        summary.filtered_non_frontal
    ));
    report.push_str(&format!(
        "- patient overlap count: {}\n",
        summary.patient_overlap_count
    ));
    report.push_str(&format!(
        "- duplicate image hash overlap count: {}\n",
        summary.duplicate_hash_overlap_count
    ));
    report.push_str("\n## Splits\n\n");
    for (split, count) in &summary.split_counts {
        report.push_str(&format!("- {split}: {count}\n"));
    }
    report.push_str("\n## Labels\n\n");
    for (target, counts) in &summary.target_counts {
        report.push_str(&format!(
            "- {target}: positive {}, negative {}, uncertain {}, missing {}\n",
            counts.positive, counts.negative, counts.uncertain, counts.missing
        ));
    }
    fs::write(path, report)?;
    Ok(())
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
    check_file_size(
        &resolve_cache_path(cache_dir, &split_summary.images_path),
        image_values * std::mem::size_of::<f32>(),
        split,
        "images",
        errors,
    )?;
    let label_bytes = label_values * std::mem::size_of::<f32>();
    let labels_path = resolve_cache_path(cache_dir, &split_summary.labels_path);
    check_file_size(&labels_path, label_bytes, split, "labels", errors)?;
    let masks_path = resolve_cache_path(cache_dir, &split_summary.masks_path);
    check_file_size(&masks_path, label_bytes, split, "masks", errors)?;
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

fn add_label_count(count: &mut LabelCount, value: Option<i8>) {
    match value {
        Some(1) => count.positive += 1,
        Some(0) => count.negative += 1,
        Some(-1) => count.uncertain += 1,
        _ => count.missing += 1,
    }
}

fn scan_images(root: &Path) -> Result<HashMap<String, PathBuf>, CxrError> {
    let mut map = HashMap::new();
    scan_images_inner(root, &mut map)?;
    Ok(map)
}

fn scan_images_inner(root: &Path, map: &mut HashMap<String, PathBuf>) -> Result<(), CxrError> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            scan_images_inner(&path, map)?;
        } else if is_image_path(&path) {
            if let Some(stem) = path.file_stem().and_then(|value| value.to_str()) {
                map.insert(stem.to_string(), path);
            }
        }
    }
    Ok(())
}

fn is_image_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase())
            .as_deref(),
        Some("jpg" | "jpeg" | "png")
    )
}

fn csv_reader(path: &Path) -> Result<csv::Reader<Box<dyn Read>>, CxrError> {
    let file = File::open(path)?;
    let reader: Box<dyn Read> = if path.extension().and_then(|value| value.to_str()) == Some("gz") {
        Box::new(MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };
    Ok(csv::ReaderBuilder::new().flexible(true).from_reader(reader))
}

struct HeaderIndex {
    headers: HashMap<String, usize>,
}

impl HeaderIndex {
    fn new(headers: &csv::StringRecord) -> Self {
        let headers = headers
            .iter()
            .enumerate()
            .map(|(index, header)| (header.to_ascii_lowercase(), index))
            .collect();
        Self { headers }
    }

    fn get(&self, row: &csv::StringRecord, names: &[&str]) -> Option<String> {
        names.iter().find_map(|name| {
            self.headers
                .get(&name.to_ascii_lowercase())
                .and_then(|index| row.get(*index))
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
        })
    }
}

fn metadata_or_image_dimensions(
    headers: &HeaderIndex,
    row: &csv::StringRecord,
    path: &Path,
) -> Result<(u32, u32), CxrError> {
    let rows = headers
        .get(row, &["Rows", "height"])
        .and_then(|value| value.parse::<u32>().ok());
    let columns = headers
        .get(row, &["Columns", "width"])
        .and_then(|value| value.parse::<u32>().ok());
    match (columns, rows) {
        (Some(width), Some(height)) => Ok((width, height)),
        _ => Ok(image::image_dimensions(path)?),
    }
}

fn parse_label_value(value: &str) -> Option<i8> {
    match value {
        "1" | "1.0" => Some(1),
        "0" | "0.0" => Some(0),
        "-1" | "-1.0" => Some(-1),
        "" => None,
        _ => None,
    }
}

fn patient_from_filename(value: &str) -> String {
    value
        .split('_')
        .next()
        .filter(|part| !part.is_empty())
        .unwrap_or(value)
        .to_string()
}

fn is_frontal(view_position: Option<&str>) -> bool {
    matches!(
        view_position
            .map(|value| value.to_ascii_uppercase())
            .as_deref(),
        Some("PA" | "AP")
    )
}

fn overlap_count(values_by_split: &BTreeMap<String, BTreeSet<String>>) -> usize {
    let entries = values_by_split.iter().collect::<Vec<_>>();
    let mut overlap = BTreeSet::new();
    for left_index in 0..entries.len() {
        for right_index in (left_index + 1)..entries.len() {
            overlap.extend(
                entries[left_index]
                    .1
                    .intersection(entries[right_index].1)
                    .cloned(),
            );
        }
    }
    overlap.len()
}

fn stable_bucket(value: &str, seed: u64) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(seed.to_le_bytes());
    hasher.update(value.as_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes([
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
    ])
}

fn write_manifest(path: &Path, records: &[CxrRecord]) -> Result<(), CxrError> {
    let mut writer = BufWriter::new(File::create(path)?);
    for record in records {
        serde_json::to_writer(&mut writer, record)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

pub fn read_manifest(path: &Path) -> Result<Vec<CxrRecord>, CxrError> {
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

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), CxrError> {
    let writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(writer, value)?;
    Ok(())
}

fn resolve_cache_path(cache_dir: &Path, path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        cache_dir.join(path)
    }
}

fn read_f32_range(path: &Path, start_value: usize, out: &mut [f32]) -> Result<(), CxrError> {
    let mut file = File::open(path)?;
    read_f32_range_from(&mut file, start_value, out)
}

fn read_indexed_runs(
    runs: &[CxrIndexedRun],
    image_path: &Path,
    labels_path: &Path,
    masks_path: &Path,
    image_values_per_sample: usize,
    target_count: usize,
) -> Result<Vec<CxrIndexedRunRead>, CxrError> {
    let mut images = File::open(image_path)?;
    let mut labels = File::open(labels_path)?;
    let mut masks = File::open(masks_path)?;
    let mut reads = Vec::with_capacity(runs.len());
    for run in runs {
        let run_len = run.out_indices.len();
        let image_offset = run
            .start_sample
            .checked_mul(image_values_per_sample)
            .ok_or_else(|| CxrError::Message("image offset overflow".to_string()))?;
        let label_offset = run
            .start_sample
            .checked_mul(target_count)
            .ok_or_else(|| CxrError::Message("label offset overflow".to_string()))?;
        let mut image_scratch = vec![0.0f32; run_len * image_values_per_sample];
        let mut label_scratch = vec![0.0f32; run_len * target_count];
        let mut mask_scratch = vec![0.0f32; run_len * target_count];
        read_f32_range_from(&mut images, image_offset, &mut image_scratch)?;
        read_f32_range_from(&mut labels, label_offset, &mut label_scratch)?;
        read_f32_range_from(&mut masks, label_offset, &mut mask_scratch)?;
        reads.push(CxrIndexedRunRead {
            out_indices: run.out_indices.clone(),
            images: image_scratch,
            labels: label_scratch,
            masks: mask_scratch,
        });
    }
    Ok(reads)
}

fn read_f32_range_from(
    file: &mut File,
    start_value: usize,
    out: &mut [f32],
) -> Result<(), CxrError> {
    file.seek(SeekFrom::Start(byte_offset(start_value)?))?;
    let mut bytes = vec![0u8; std::mem::size_of_val(out)];
    file.read_exact(&mut bytes)?;
    for (slot, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(())
}

fn byte_offset(value_index: usize) -> Result<u64, CxrError> {
    value_index
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| CxrError::Message("cache byte offset overflow".to_string()))
}

fn hash_file(path: &Path) -> Result<String, CxrError> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 64];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn directory_size(path: &Path) -> Result<u64, CxrError> {
    let mut total = 0u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            total += directory_size(&path)?;
        } else {
            total += metadata.len();
        }
    }
    Ok(total)
}

#[derive(Debug)]
pub enum CxrError {
    Io(std::io::Error),
    Csv(csv::Error),
    Json(serde_json::Error),
    Image(image::ImageError),
    Toml(toml::de::Error),
    Message(String),
}

impl std::fmt::Display for CxrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Csv(error) => write!(f, "{error}"),
            Self::Json(error) => write!(f, "{error}"),
            Self::Image(error) => write!(f, "{error}"),
            Self::Toml(error) => write!(f, "{error}"),
            Self::Message(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for CxrError {}

impl From<std::io::Error> for CxrError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<csv::Error> for CxrError {
    fn from(value: csv::Error) -> Self {
        Self::Csv(value)
    }
}

impl From<serde_json::Error> for CxrError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<image::ImageError> for CxrError {
    fn from(value: image::ImageError) -> Self {
        Self::Image(value)
    }
}

impl From<toml::de::Error> for CxrError {
    fn from(value: toml::de::Error) -> Self {
        Self::Toml(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_are_patient_safe_and_cache_builds() {
        let root = unique_test_dir();
        let Fixture {
            manifest,
            splits,
            plan,
            cache_dir,
            ..
        } = build_fixture(&root);

        let validation = validate_cxr(&ValidateConfig {
            manifest_path: manifest.clone(),
            require_frontal: false,
            check_patient_leakage: true,
            check_duplicates: true,
            report_path: root.join("validation.md"),
        })
        .unwrap();
        assert_eq!(validation.patient_overlap_count, 0);

        let cache = cache_cxr(&CacheConfig {
            manifest_path: manifest,
            splits_path: splits,
            plan_path: plan,
            cache_dir: cache_dir.clone(),
        })
        .unwrap();
        assert_eq!(cache.cache_schema_version, CXR_CACHE_SCHEMA_VERSION);
        assert_eq!(cache.report_schema_version, CXR_REPORT_SCHEMA_VERSION);
        assert_eq!(cache.image_size, 8);
        assert_eq!(cache.label_policy, LabelPolicy::default());
        assert_eq!(cache.transform_fingerprint, cache.transform_plan_hash);
        assert_eq!(cache.split_names, vec!["test", "train", "val"]);
        assert_eq!(cache.image_size_policy.height, 8);
        assert!(!cache.source_manifest_checksum.is_empty());
        assert!(cache_dir.join("cache-metadata.json").exists());

        let reader = CxrCacheReader::open(&cache_dir, "train").unwrap();
        assert_eq!(reader.cache_schema_version(), CXR_CACHE_SCHEMA_VERSION);
        assert_eq!(reader.label_policy(), &LabelPolicy::default());
        assert_eq!(reader.image_size(), 8);
        assert_eq!(reader.image_shape()[1], 1);
        assert!(!reader.targets().is_empty());
        assert!(reader.samples() > 0);

        let batch = reader.read_batch(0, 2).unwrap();
        assert!(batch.samples > 0);
        assert_eq!(batch.image_shape[0], batch.samples);
        assert_eq!(batch.image_shape[1], 1);
        assert_eq!(batch.image_shape[2], 8);
        assert_eq!(batch.image_shape[3], 8);
        assert_eq!(batch.labels_shape, [batch.samples, reader.targets().len()]);
        assert_eq!(batch.images.len(), batch.samples * 8 * 8);
        assert_eq!(batch.labels.len(), batch.samples * reader.targets().len());
        assert_eq!(batch.masks.len(), batch.samples * reader.targets().len());
        assert_eq!(batch.records.len(), batch.samples);
        assert_eq!(
            reader.records_for_range(0, batch.samples).unwrap(),
            batch.records
        );

        let mut image_out = vec![0.0f32; 8 * 8];
        let mut labels_out = vec![0.0f32; reader.targets().len()];
        let mut masks_out = vec![0.0f32; reader.targets().len()];
        let written = reader
            .fill_batch(0, 1, &mut image_out, &mut labels_out, &mut masks_out)
            .unwrap();
        assert_eq!(written, 1);

        let empty = reader.read_batch(reader.samples(), 1).unwrap();
        assert_eq!(empty.samples, 0);
    }

    #[test]
    fn cache_validation_reports_actionable_mismatches() {
        let root = unique_test_dir();
        let Fixture {
            manifest,
            splits,
            plan,
            cache_dir,
            ..
        } = build_fixture(&root);
        cache_cxr(&CacheConfig {
            manifest_path: manifest,
            splits_path: splits,
            plan_path: plan.clone(),
            cache_dir: cache_dir.clone(),
        })
        .unwrap();

        let ok = validate_cache_cxr(&ValidateCacheConfig {
            cache_dir: cache_dir.clone(),
            split: Some("train".to_string()),
            expected_targets: Some(vec!["No Finding".to_string(), "Pneumonia".to_string()]),
            expected_image_shape: None,
            plan_path: Some(plan.clone()),
            report_path: Some(root.join("cache-validation.md")),
            json_path: Some(root.join("cache-validation.json")),
        })
        .unwrap();
        assert_eq!(ok.status, "ok");
        assert!(ok.errors.is_empty());
        assert!(root.join("cache-validation.md").exists());
        assert!(root.join("cache-validation.json").exists());

        fs::write(
            root.join("different-plan.toml"),
            "[image]\nsize = [16, 16]\n",
        )
        .unwrap();
        let mismatch = validate_cache_cxr(&ValidateCacheConfig {
            cache_dir: cache_dir.clone(),
            split: Some("train".to_string()),
            expected_targets: Some(vec!["Pneumonia".to_string(), "No Finding".to_string()]),
            expected_image_shape: Some([99, 1, 8, 8]),
            plan_path: Some(root.join("different-plan.toml")),
            report_path: None,
            json_path: None,
        })
        .unwrap();
        assert_eq!(mismatch.status, "failed");
        assert!(mismatch
            .errors
            .iter()
            .any(|error| error.contains("target-list mismatch")));
        assert!(mismatch
            .errors
            .iter()
            .any(|error| error.contains("wrong image shape")));
        assert!(mismatch
            .errors
            .iter()
            .any(|error| error.contains("stale transform fingerprint")));

        let mut summary = read_cache_summary(&cache_dir).unwrap();
        summary.failed_samples.push("p1: failed".to_string());
        let train = summary.splits.get_mut("train").unwrap();
        fs::remove_file(resolve_cache_path(&cache_dir, &train.labels_path)).unwrap();
        train.samples += 1;
        write_json(&cache_dir.join("cache-metadata.json"), &summary).unwrap();
        let broken = validate_cache_cxr(&ValidateCacheConfig {
            cache_dir,
            split: Some("train".to_string()),
            expected_targets: None,
            expected_image_shape: None,
            plan_path: None,
            report_path: None,
            json_path: None,
        })
        .unwrap();
        assert!(broken
            .errors
            .iter()
            .any(|error| error.contains("missing labels file")));
        assert!(broken
            .errors
            .iter()
            .any(|error| error.contains("metadata sample count mismatch")));
        assert!(broken
            .errors
            .iter()
            .any(|error| error.contains("failed preprocessed samples")));
    }

    #[test]
    fn cache_open_rejects_incompatible_schema_and_missing_split() {
        let root = unique_test_dir();
        let Fixture {
            manifest,
            splits,
            plan,
            cache_dir,
            ..
        } = build_fixture(&root);
        cache_cxr(&CacheConfig {
            manifest_path: manifest,
            splits_path: splits,
            plan_path: plan,
            cache_dir: cache_dir.clone(),
        })
        .unwrap();
        let missing = CxrCacheReader::open(&cache_dir, "missing").unwrap_err();
        assert!(missing.to_string().contains("cache split"));

        let mut summary = read_cache_summary(&cache_dir).unwrap();
        summary.cache_schema_version = 99;
        write_json(&cache_dir.join("cache-metadata.json"), &summary).unwrap();
        let schema = CxrCacheReader::open(&cache_dir, "train").unwrap_err();
        assert!(schema
            .to_string()
            .contains("unsupported CXR cache schema version"));
    }

    #[test]
    fn split_membership_errors_are_hard_failures() {
        let root = unique_test_dir();
        let Fixture {
            manifest,
            splits,
            plan,
            cache_dir,
            ..
        } = build_fixture(&root);
        let mut split_file = read_split_file(&splits).unwrap();
        split_file.train.push("unknown".to_string());
        write_json(&splits, &split_file).unwrap();
        let unknown = cache_cxr(&CacheConfig {
            manifest_path: manifest.clone(),
            splits_path: splits.clone(),
            plan_path: plan.clone(),
            cache_dir: cache_dir.clone(),
        })
        .unwrap_err();
        assert!(unknown.to_string().contains("unknown sample IDs"));

        let Fixture {
            manifest,
            splits,
            plan,
            cache_dir,
            ..
        } = build_fixture(&unique_test_dir());
        let mut split_file = read_split_file(&splits).unwrap();
        let duplicate = split_file.train[0].clone();
        split_file.val.push(duplicate);
        write_json(&splits, &split_file).unwrap();
        let duplicate_error = cache_cxr(&CacheConfig {
            manifest_path: manifest.clone(),
            splits_path: splits.clone(),
            plan_path: plan.clone(),
            cache_dir: cache_dir.clone(),
        })
        .unwrap_err();
        assert!(duplicate_error.to_string().contains("multiple splits"));

        let Fixture {
            manifest,
            splits,
            plan,
            cache_dir,
            ..
        } = build_fixture(&unique_test_dir());
        let mut split_file = read_split_file(&splits).unwrap();
        split_file.train.pop();
        write_json(&splits, &split_file).unwrap();
        let omitted = cache_cxr(&CacheConfig {
            manifest_path: manifest,
            splits_path: splits,
            plan_path: plan,
            cache_dir,
        })
        .unwrap_err();
        assert!(omitted.to_string().contains("omits"));
    }

    #[test]
    fn image_only_indexing_and_helper_policies_are_stable() {
        let root = unique_test_dir();
        let images = root.join("nested/images");
        fs::create_dir_all(&images).unwrap();
        write_png(&images.join("patient_a_scan.png"), 12);
        write_png(&images.join("patient_b_scan.jpg"), 13);
        fs::write(images.join("ignore.txt"), "not an image").unwrap();

        let manifest = root.join("manifest.jsonl");
        let summary = index_cxr(&IndexConfig {
            images_root: root.join("nested"),
            metadata_path: None,
            labels_path: None,
            reports_root: None,
            out_path: manifest.clone(),
        })
        .unwrap();
        assert_eq!(summary.records, 2);
        assert_eq!(summary.patients, 1);
        let records = read_manifest(&manifest).unwrap();
        assert!(records.iter().all(|record| record.labels.is_empty()));

        assert_eq!(parse_label_value("1.0"), Some(1));
        assert_eq!(parse_label_value("0.0"), Some(0));
        assert_eq!(parse_label_value("-1.0"), Some(-1));
        assert_eq!(parse_label_value("maybe"), None);
        assert!(is_frontal(Some("pa")));
        assert!(is_frontal(Some("AP")));
        assert!(!is_frontal(Some("LL")));
        assert_eq!(patient_from_filename("_leading"), "_leading");
        assert!(is_image_path(Path::new("x.JPEG")));
        assert!(!is_image_path(Path::new("x.txt")));

        let mut overlaps = BTreeMap::new();
        overlaps.insert("train".to_string(), BTreeSet::from(["p1".to_string()]));
        overlaps.insert(
            "val".to_string(),
            BTreeSet::from(["p1".to_string(), "p2".to_string()]),
        );
        overlaps.insert("test".to_string(), BTreeSet::from(["p2".to_string()]));
        assert_eq!(overlap_count(&overlaps), 2);
        assert_eq!(stable_bucket("patient", 1), stable_bucket("patient", 1));
        assert_ne!(stable_bucket("patient", 1), stable_bucket("patient", 2));
    }

    #[test]
    fn validation_and_split_error_paths_are_explicit() {
        let root = unique_test_dir();
        let Fixture { manifest, .. } = build_fixture(&root);
        let records = read_manifest(&manifest).unwrap();
        let mut edited = records.clone();
        edited[0].view_position = Some("LL".to_string());
        edited[1].image_path = root.join("missing.png").display().to_string();
        edited[2].sha256 = None;
        edited[0].split = Some("train".to_string());
        edited[1].split = Some("val".to_string());
        edited[2].split = Some("test".to_string());
        write_manifest(&manifest, &edited).unwrap();

        let filtered = validate_cxr(&ValidateConfig {
            manifest_path: manifest.clone(),
            require_frontal: true,
            check_patient_leakage: false,
            check_duplicates: false,
            report_path: root.join("filtered.md"),
        })
        .unwrap();
        assert_eq!(filtered.filtered_non_frontal, 2);
        assert_eq!(filtered.patient_overlap_count, 0);
        assert_eq!(filtered.duplicate_hash_overlap_count, 0);

        let unfiltered = validate_cxr(&ValidateConfig {
            manifest_path: manifest.clone(),
            require_frontal: false,
            check_patient_leakage: true,
            check_duplicates: true,
            report_path: root.join("unfiltered.md"),
        })
        .unwrap();
        assert_eq!(unfiltered.unreadable_images, 1);

        let invalid_by = split_cxr(&SplitConfig {
            manifest_path: manifest.clone(),
            by: "study_id".to_string(),
            train: 0.8,
            val: 0.1,
            test: 0.1,
            stratify: Vec::new(),
            out_path: root.join("bad-splits.json"),
            seed: 0,
        })
        .unwrap_err();
        assert!(invalid_by.to_string().contains("only patient-level"));

        let patient_alias = split_cxr(&SplitConfig {
            manifest_path: manifest,
            by: "patient".to_string(),
            train: 0.34,
            val: 0.33,
            test: 0.33,
            stratify: vec!["Pneumonia".to_string()],
            out_path: root.join("patient-splits.json"),
            seed: 2,
        })
        .unwrap();
        assert_eq!(patient_alias.by, "patient_id");
        assert_eq!(patient_alias.ratios["test"], 0.33);
    }

    #[test]
    fn cache_reader_reports_buffer_and_index_errors() {
        let root = unique_test_dir();
        let Fixture {
            manifest,
            splits,
            plan,
            cache_dir,
            ..
        } = build_fixture(&root);
        cache_cxr(&CacheConfig {
            manifest_path: manifest,
            splits_path: splits,
            plan_path: plan,
            cache_dir: cache_dir.clone(),
        })
        .unwrap();
        let reader = CxrCacheReader::open(&cache_dir, "train").unwrap();
        let image_len = reader.image_shape()[1] * reader.image_shape()[2] * reader.image_shape()[3];
        let labels_len = reader.targets().len();

        let mut images = vec![0.0; image_len - 1];
        let mut labels = vec![0.0; labels_len];
        let mut masks = vec![0.0; labels_len];
        assert!(reader
            .fill_batch(0, 1, &mut images, &mut labels, &mut masks)
            .unwrap_err()
            .to_string()
            .contains("image output buffer too small"));

        let mut images = vec![0.0; image_len];
        let mut labels = vec![0.0; labels_len - 1];
        assert!(reader
            .fill_batch(0, 1, &mut images, &mut labels, &mut masks)
            .unwrap_err()
            .to_string()
            .contains("label output buffer too small"));

        let mut labels = vec![0.0; labels_len];
        let mut masks = vec![0.0; labels_len - 1];
        assert!(reader
            .fill_batch(0, 1, &mut images, &mut labels, &mut masks)
            .unwrap_err()
            .to_string()
            .contains("mask output buffer too small"));

        assert_eq!(
            reader
                .fill_indices(&[], &mut images, &mut labels, &mut masks)
                .unwrap(),
            0
        );
        let mut images = vec![0.0; image_len];
        let mut labels = vec![0.0; labels_len];
        let mut masks = vec![0.0; labels_len];
        assert!(reader
            .fill_indices(&[reader.samples()], &mut images, &mut labels, &mut masks)
            .unwrap_err()
            .to_string()
            .contains("out of bounds"));
        assert!(reader
            .records_for_range(reader.samples(), 1)
            .unwrap_err()
            .to_string()
            .contains("out of bounds"));
        assert!(reader
            .records_for_indices(&[reader.samples()])
            .unwrap_err()
            .to_string()
            .contains("out of bounds"));

        assert_eq!(
            reader
                .fill_indices_parallel(&[], &mut images, &mut labels, &mut masks, 4)
                .unwrap()
                .samples,
            0
        );
        assert!(reader
            .fill_indices_parallel(
                &[reader.samples()],
                &mut vec![0.0; image_len],
                &mut vec![0.0; labels_len],
                &mut vec![0.0; labels_len],
                1,
            )
            .unwrap_err()
            .to_string()
            .contains("out of bounds"));
    }

    #[test]
    fn cache_metadata_legacy_defaults_and_reader_file_checks_work() {
        let root = unique_test_dir();
        let Fixture {
            manifest,
            splits,
            plan,
            cache_dir,
            ..
        } = build_fixture(&root);
        cache_cxr(&CacheConfig {
            manifest_path: manifest,
            splits_path: splits,
            plan_path: plan,
            cache_dir: cache_dir.clone(),
        })
        .unwrap();

        let mut summary_json: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(cache_dir.join("cache-metadata.json")).unwrap(),
        )
        .unwrap();
        summary_json.as_object_mut().unwrap().remove("label_policy");
        summary_json
            .as_object_mut()
            .unwrap()
            .remove("image_size_policy");
        fs::write(
            cache_dir.join("cache-metadata.json"),
            serde_json::to_string_pretty(&summary_json).unwrap(),
        )
        .unwrap();
        let legacy = read_cache_summary(&cache_dir).unwrap();
        assert_eq!(legacy.label_policy, LabelPolicy::default());
        assert_eq!(legacy.image_size_policy, ImageSizePolicy::default());

        let mut summary = legacy;
        summary.cache_schema_version = CXR_CACHE_SCHEMA_VERSION;
        summary.splits.get_mut("train").unwrap().shape[1] = 3;
        write_json(&cache_dir.join("cache-metadata.json"), &summary).unwrap();
        assert!(CxrCacheReader::open(&cache_dir, "train")
            .unwrap_err()
            .to_string()
            .contains("single-channel"));

        let train = summary.splits.get_mut("train").unwrap();
        train.shape[1] = 1;
        train.images_path = "missing-images.float32.dat".to_string();
        write_json(&cache_dir.join("cache-metadata.json"), &summary).unwrap();
        assert!(CxrCacheReader::open(&cache_dir, "train")
            .unwrap_err()
            .to_string()
            .contains("missing images file"));
    }

    #[test]
    fn plan_and_io_error_paths_are_reported() {
        let root = unique_test_dir();
        assert!(image_size_from_plan(&root.join("missing.toml"))
            .unwrap_err()
            .to_string()
            .contains("No such file"));
        fs::write(root.join("bad-plan.toml"), "name = [").unwrap();
        assert!(matches!(
            image_size_from_plan(&root.join("bad-plan.toml")).unwrap_err(),
            CxrError::Toml(_)
        ));
        fs::write(root.join("no-size.toml"), "name = \"no-size\"\n").unwrap();
        assert!(image_size_from_plan(&root.join("no-size.toml"))
            .unwrap_err()
            .to_string()
            .contains("could not determine image size"));
        fs::write(
            root.join("ops-size.toml"),
            "[[operations]]\nop = \"resize\"\nsize = 12\n",
        )
        .unwrap();
        assert_eq!(
            image_size_from_plan(&root.join("ops-size.toml")).unwrap(),
            12
        );
        assert_eq!(ImageSizePolicy::default().height, 0);

        let Fixture {
            manifest,
            splits,
            plan,
            cache_dir,
            ..
        } = build_fixture(&root.join("cache-write-error"));
        fs::create_dir_all(&cache_dir).unwrap();
        fs::create_dir(cache_dir.join("train-images.float32.dat")).unwrap();
        assert!(matches!(
            cache_cxr(&CacheConfig {
                manifest_path: manifest,
                splits_path: splits,
                plan_path: plan,
                cache_dir,
            })
            .unwrap_err(),
            CxrError::Io(_)
        ));
    }

    #[test]
    fn metadata_indexing_skips_missing_images_and_reads_gz_labels() {
        let root = unique_test_dir();
        fs::create_dir_all(root.join("images")).unwrap();
        write_rgb_png(&root.join("images/keep.png"), [20, 40, 60]);
        fs::write(
            root.join("metadata.csv"),
            "dicom_id,subject_id,study_id,ViewPosition,laterality,modality\nkeep,42,7,AP,L,DX\nmissing,43,8,PA,R,CR\n",
        )
        .unwrap();
        let gz_labels = root.join("labels.csv.gz");
        {
            let file = File::create(&gz_labels).unwrap();
            let mut encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
            encoder
                .write_all(b"subject_id,study_id,Pneumonia,Other\n42,7,1,2\n")
                .unwrap();
            encoder.finish().unwrap();
        }

        let manifest = root.join("manifest.jsonl");
        let summary = index_cxr(&IndexConfig {
            images_root: root.join("images"),
            metadata_path: Some(root.join("metadata.csv")),
            labels_path: Some(gz_labels),
            reports_root: Some(root.join("reports")),
            out_path: manifest.clone(),
        })
        .unwrap();
        assert_eq!(summary.records, 1);
        assert_eq!(summary.labels["Pneumonia"].positive, 1);
        assert_eq!(summary.labels["Other"].missing, 1);

        fs::write(
            &manifest,
            format!("\n{}\n\n", fs::read_to_string(&manifest).unwrap()),
        )
        .unwrap();
        let records = read_manifest(&manifest).unwrap();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.patient_id, "p42");
        assert_eq!(record.study_id, "s7");
        assert_eq!(
            record.report_path.as_deref(),
            Some(root.join("reports/s7.txt").to_str().unwrap())
        );
        assert_eq!(record.width, Some(4));
        assert_eq!(record.height, Some(4));
        assert_eq!(record.modality.as_deref(), Some("DX"));
        assert_eq!(record.laterality.as_deref(), Some("L"));
    }

    #[test]
    fn cache_validation_covers_schema_warnings_missing_splits_and_reports() {
        let root = unique_test_dir();
        let cache_dir = root.join("cache");
        let mut summary = write_synthetic_cache(&cache_dir, 3, 2, &["A", "B"]);
        summary.cache_schema_version = 99;
        summary.report_schema_version = 77;
        summary.transform_fingerprint.clear();
        summary.split_names.clear();
        write_json(&cache_dir.join("cache-metadata.json"), &summary).unwrap();

        let report_path = root.join("report.md");
        let json_path = root.join("report.json");
        let validation = validate_cache_cxr(&ValidateCacheConfig {
            cache_dir: cache_dir.clone(),
            split: Some("missing".to_string()),
            expected_targets: None,
            expected_image_shape: None,
            plan_path: None,
            report_path: Some(report_path.clone()),
            json_path: Some(json_path.clone()),
        })
        .unwrap();

        assert_eq!(validation.status, "failed");
        assert!(validation
            .errors
            .iter()
            .any(|error| error.contains("unsupported CXR cache schema version")));
        assert!(validation
            .errors
            .iter()
            .any(|error| error.contains("cache split \"missing\" not found")));
        assert!(validation
            .warnings
            .iter()
            .any(|warning| warning.contains("CXR report schema version")));
        assert_eq!(validation.split_names, vec!["train"]);
        assert_eq!(
            validation.transform_fingerprint,
            summary.transform_plan_hash
        );
        let report = fs::read_to_string(report_path).unwrap();
        assert!(report.contains("unsupported CXR cache schema version"));
        assert!(report.contains("CXR report schema version"));
        assert!(json_path.exists());

        fs::write(root.join("different-plan.toml"), "[image]\nsize = [3, 3]\n").unwrap();
        let stale_plan = validate_cache_cxr(&ValidateCacheConfig {
            cache_dir: cache_dir.clone(),
            split: Some("train".to_string()),
            expected_targets: None,
            expected_image_shape: None,
            plan_path: Some(root.join("different-plan.toml")),
            report_path: None,
            json_path: None,
        })
        .unwrap();
        assert!(stale_plan
            .errors
            .iter()
            .any(|error| error.contains("stale transform fingerprint: cache has hash")));

        let all_splits = validate_cache_cxr(&ValidateCacheConfig {
            cache_dir,
            split: None,
            expected_targets: Some(vec!["A".to_string(), "B".to_string()]),
            expected_image_shape: Some([3, 1, 2, 2]),
            plan_path: None,
            report_path: None,
            json_path: None,
        })
        .unwrap();
        assert_eq!(all_splits.checked_splits, vec!["train"]);
    }

    #[test]
    fn parallel_indexed_reads_scatter_non_contiguous_runs() {
        let root = unique_test_dir();
        let cache_dir = root.join("cache");
        write_synthetic_cache(&cache_dir, 5, 2, &["A", "B"]);
        let reader = CxrCacheReader::open(&cache_dir, "train").unwrap();
        assert_eq!(reader.split(), "train");

        let indices = [4, 1, 0];
        let image_len = indices.len() * 4;
        let label_len = indices.len() * 2;
        let mut streaming_images = vec![0.0; image_len];
        let mut streaming_labels = vec![0.0; label_len];
        let mut streaming_masks = vec![0.0; label_len];
        reader
            .fill_indices(
                &indices,
                &mut streaming_images,
                &mut streaming_labels,
                &mut streaming_masks,
            )
            .unwrap();

        let mut parallel_images = vec![0.0; image_len];
        let mut parallel_labels = vec![0.0; label_len];
        let mut parallel_masks = vec![0.0; label_len];
        let metrics = reader
            .fill_indices_parallel(
                &indices,
                &mut parallel_images,
                &mut parallel_labels,
                &mut parallel_masks,
                2,
            )
            .unwrap();

        assert_eq!(metrics.samples, 3);
        assert_eq!(metrics.runs, 2);
        assert_eq!(metrics.workers, 2);
        assert_eq!(metrics.read_bytes, (image_len + label_len + label_len) * 4);
        assert_eq!(metrics.scatter_bytes, metrics.read_bytes);
        assert_eq!(parallel_images, streaming_images);
        assert_eq!(parallel_labels, streaming_labels);
        assert_eq!(parallel_masks, streaming_masks);
        assert_eq!(parallel_images[0..4], [16.0, 17.0, 18.0, 19.0]);
        assert_eq!(parallel_labels[0..2], [8.0, 9.0]);
        assert_eq!(parallel_masks[0..2], [108.0, 109.0]);

        let mut single_worker_images = vec![0.0; image_len];
        let mut single_worker_labels = vec![0.0; label_len];
        let mut single_worker_masks = vec![0.0; label_len];
        let single_worker = reader
            .fill_indices_parallel(
                &indices,
                &mut single_worker_images,
                &mut single_worker_labels,
                &mut single_worker_masks,
                1,
            )
            .unwrap();
        assert_eq!(single_worker.samples, 3);
        assert_eq!(single_worker.workers, 1);
        assert_eq!(single_worker_images, streaming_images);
        assert_eq!(single_worker_labels, streaming_labels);
        assert_eq!(single_worker_masks, streaming_masks);
    }

    #[test]
    fn indexed_read_buffer_errors_cover_all_entry_points() {
        let root = unique_test_dir();
        let cache_dir = root.join("cache");
        write_synthetic_cache(&cache_dir, 2, 2, &["A", "B"]);
        let reader = CxrCacheReader::open(&cache_dir, "train").unwrap();
        let mut images = vec![0.0; 3];
        let mut labels = vec![0.0; 2];
        let mut masks = vec![0.0; 2];
        assert!(reader
            .fill_indices(&[0], &mut images, &mut labels, &mut masks)
            .unwrap_err()
            .to_string()
            .contains("image output buffer too small"));

        let mut images = vec![0.0; 4];
        let mut labels = vec![0.0; 1];
        assert!(reader
            .fill_indices(&[0], &mut images, &mut labels, &mut masks)
            .unwrap_err()
            .to_string()
            .contains("label output buffer too small"));

        let mut labels = vec![0.0; 2];
        let mut masks = vec![0.0; 1];
        assert!(reader
            .fill_indices(&[0], &mut images, &mut labels, &mut masks)
            .unwrap_err()
            .to_string()
            .contains("mask output buffer too small"));

        let mut images = vec![0.0; 3];
        let mut labels = vec![0.0; 2];
        let mut masks = vec![0.0; 2];
        assert!(reader
            .fill_indices_parallel(&[0], &mut images, &mut labels, &mut masks, 4)
            .unwrap_err()
            .to_string()
            .contains("image output buffer too small"));

        let mut images = vec![0.0; 4];
        let mut labels = vec![0.0; 1];
        assert!(reader
            .fill_indices_parallel(&[0], &mut images, &mut labels, &mut masks, 4)
            .unwrap_err()
            .to_string()
            .contains("label output buffer too small"));

        let mut labels = vec![0.0; 2];
        let mut masks = vec![0.0; 1];
        assert!(reader
            .fill_indices_parallel(&[0], &mut images, &mut labels, &mut masks, 4)
            .unwrap_err()
            .to_string()
            .contains("mask output buffer too small"));
    }

    #[test]
    fn cache_build_records_failed_samples_and_non_binary_labels() {
        let root = unique_test_dir();
        let Fixture {
            manifest,
            splits,
            plan,
            cache_dir,
            ..
        } = build_fixture(&root);
        let split_file = read_split_file(&splits).unwrap();
        let failing_id = split_file
            .val
            .iter()
            .chain(split_file.test.iter())
            .next()
            .expect("fixture has non-train sample")
            .clone();
        let mut records = read_manifest(&manifest).unwrap();
        for record in &mut records {
            if record.sample_id == failing_id {
                record.image_path = root.join("missing-cache-source.png").display().to_string();
                record.labels.insert("Custom".to_string(), Some(2));
            } else {
                record.labels.insert("Custom".to_string(), None);
            }
        }
        write_manifest(&manifest, &records).unwrap();

        let cache = cache_cxr(&CacheConfig {
            manifest_path: manifest.clone(),
            splits_path: splits,
            plan_path: plan,
            cache_dir: cache_dir.clone(),
        })
        .unwrap();
        assert_eq!(cache.failed_samples.len(), 1);
        assert!(cache.failed_samples[0].contains(&failing_id));
        assert!(cache.targets.contains(&"Custom".to_string()));

        let target_index = cache
            .targets
            .iter()
            .position(|target| target == "Custom")
            .unwrap();
        let split_name = [("val", &split_file.val), ("test", &split_file.test)]
            .into_iter()
            .find(|(_name, ids)| ids.contains(&failing_id))
            .map(|(name, _ids)| name)
            .unwrap();
        let reader = CxrCacheReader::open(&cache_dir, split_name).unwrap();
        let row = reader
            .records_for_range(0, reader.samples())
            .unwrap()
            .into_iter()
            .position(|record| record.sample_id == failing_id)
            .unwrap();
        let mut images = vec![1.0; reader.samples() * 8 * 8];
        let mut labels = vec![0.0; reader.samples() * reader.targets().len()];
        let mut masks = vec![0.0; reader.samples() * reader.targets().len()];
        reader
            .fill_batch(0, reader.samples(), &mut images, &mut labels, &mut masks)
            .unwrap();
        assert!(images[row * 64..row * 64 + 64]
            .iter()
            .all(|value| *value == 0.0));
        assert_eq!(labels[row * reader.targets().len() + target_index], 2.0);
        assert_eq!(masks[row * reader.targets().len() + target_index], 1.0);
    }

    #[test]
    fn malformed_metadata_and_cache_io_errors_surface() {
        let root = unique_test_dir();
        fs::create_dir_all(root.join("images")).unwrap();
        write_png(&root.join("images/img.png"), 10);
        fs::write(root.join("metadata.csv"), "subject_id,study_id\n1,1\n").unwrap();
        let missing_dicom = index_cxr(&IndexConfig {
            images_root: root.join("images"),
            metadata_path: Some(root.join("metadata.csv")),
            labels_path: None,
            reports_root: None,
            out_path: root.join("manifest.jsonl"),
        })
        .unwrap_err();
        assert!(missing_dicom
            .to_string()
            .contains("metadata row missing dicom_id"));

        fs::write(root.join("labels.csv"), "subject_id,Pneumonia\n1,1\n").unwrap();
        let labels_error = read_label_csv(&root.join("labels.csv")).unwrap_err();
        assert!(labels_error
            .to_string()
            .contains("labels row missing study_id"));

        assert!(records_for_split(&[], &["unknown".to_string()])
            .unwrap_err()
            .to_string()
            .contains("unknown sample_id"));

        let cache_dir = root.join("cache");
        let mut summary = write_synthetic_cache(&cache_dir, 2, 2, &["A", "B"]);
        fs::write(cache_dir.join("train-metadata.jsonl"), "\n").unwrap();
        write_json(&cache_dir.join("cache-metadata.json"), &summary).unwrap();
        assert!(CxrCacheReader::open(&cache_dir, "train")
            .unwrap_err()
            .to_string()
            .contains("metadata sample count mismatch"));

        summary.splits.get_mut("train").unwrap().images_path = root.display().to_string();
        write_json(&cache_dir.join("cache-metadata.json"), &summary).unwrap();
        let invalid_file = validate_cache_cxr(&ValidateCacheConfig {
            cache_dir,
            split: Some("train".to_string()),
            expected_targets: None,
            expected_image_shape: None,
            plan_path: None,
            report_path: None,
            json_path: None,
        })
        .unwrap();
        assert!(invalid_file
            .errors
            .iter()
            .any(|error| error.contains("wrong images file size")));
    }

    #[test]
    fn empty_train_rgb_cache_and_remaining_fallbacks_are_covered() {
        let root = unique_test_dir();
        fs::create_dir_all(root.join("images")).unwrap();
        write_rgb_png(&root.join("images/rgb.png"), [10, 20, 30]);
        let image_path = root.join("images/rgb.png").display().to_string();
        let manifest = root.join("manifest.jsonl");
        write_manifest(
            &manifest,
            &[CxrRecord {
                sample_id: "rgb".to_string(),
                patient_id: "patient-rgb".to_string(),
                study_id: "study-rgb".to_string(),
                image_id: "rgb".to_string(),
                image_path,
                source_format: "png".to_string(),
                modality: Some("DX".to_string()),
                view_position: Some("PA".to_string()),
                laterality: None,
                width: Some(4),
                height: Some(4),
                photometric_interpretation: Some("RGB".to_string()),
                labels: BTreeMap::from([("Finding".to_string(), Some(1))]),
                label_source: Some("test".to_string()),
                report_path: None,
                split: None,
                sha256: None,
            }],
        )
        .unwrap();
        let split_file = SplitFile {
            train: Vec::new(),
            val: vec!["rgb".to_string()],
            test: Vec::new(),
            split_audit: SplitSummary {
                counts: BTreeMap::new(),
                patient_counts: BTreeMap::new(),
                by: "patient_id".to_string(),
                ratios: BTreeMap::new(),
                stratify: Vec::new(),
                patient_overlap_count: 0,
                out_path: root.join("splits.json").display().to_string(),
            },
        };
        let splits = root.join("splits.json");
        write_json(&splits, &split_file).unwrap();
        let plan = root.join("plan.toml");
        fs::write(&plan, "name = \"empty-train\"\n[image]\nsize = [4, 4]\n").unwrap();

        let cache = cache_cxr(&CacheConfig {
            manifest_path: manifest,
            splits_path: splits,
            plan_path: plan,
            cache_dir: root.join("cache"),
        })
        .unwrap();
        assert_eq!(cache.normalization.mean, 0.5);
        assert_eq!(cache.normalization.std, 0.25);
        assert_eq!(cache.splits["train"].samples, 0);
        assert_eq!(cache.splits["val"].samples, 1);
        assert_eq!(directory_size(&root).unwrap() > 0, true);

        fs::write(
            root.join("second-op-size.toml"),
            "[[operations]]\nop = \"noop\"\n\n[[operations]]\nop = \"resize\"\nsize = 10\n",
        )
        .unwrap();
        assert_eq!(
            image_size_from_plan(&root.join("second-op-size.toml")).unwrap(),
            10
        );

        let no_split_validation = validate_cxr(&ValidateConfig {
            manifest_path: root.join("cache/val-metadata.jsonl"),
            require_frontal: false,
            check_patient_leakage: true,
            check_duplicates: true,
            report_path: root.join("no-split-validation.md"),
        })
        .unwrap();
        assert!(no_split_validation.split_counts.is_empty());
    }

    #[test]
    fn cache_reader_late_file_errors_and_error_display_arms_are_covered() {
        let root = unique_test_dir();
        let cache_dir = root.join("cache");
        write_synthetic_cache(&cache_dir, 2, 2, &["A", "B"]);
        let reader = CxrCacheReader::open(&cache_dir, "train").unwrap();

        fs::remove_file(cache_dir.join("train-images.float32.dat")).unwrap();
        let mut images = vec![0.0; 4];
        let mut labels = vec![0.0; 2];
        let mut masks = vec![0.0; 2];
        assert!(reader
            .fill_batch(0, 1, &mut images, &mut labels, &mut masks)
            .unwrap_err()
            .to_string()
            .contains("No such file"));

        write_synthetic_cache(&cache_dir, 2, 2, &["A", "B"]);
        let reader = CxrCacheReader::open(&cache_dir, "train").unwrap();
        fs::remove_file(cache_dir.join("train-labels.float32.dat")).unwrap();
        assert!(reader
            .read_batch(0, 1)
            .unwrap_err()
            .to_string()
            .contains("No such file"));

        write_synthetic_cache(&cache_dir, 2, 2, &["A", "B"]);
        let reader = CxrCacheReader::open(&cache_dir, "train").unwrap();
        fs::remove_file(cache_dir.join("train-masks.float32.dat")).unwrap();
        let mut images = vec![0.0; 4];
        let mut labels = vec![0.0; 2];
        let mut masks = vec![0.0; 2];
        assert!(reader
            .fill_batch(0, 1, &mut images, &mut labels, &mut masks)
            .unwrap_err()
            .to_string()
            .contains("No such file"));

        write_synthetic_cache(&cache_dir, 2, 2, &["A", "B"]);
        let mut summary = read_cache_summary(&cache_dir).unwrap();
        fs::write(cache_dir.join("blocked-path"), "not a directory").unwrap();
        summary.splits.get_mut("train").unwrap().images_path = "blocked-path/child.dat".to_string();
        write_json(&cache_dir.join("cache-metadata.json"), &summary).unwrap();
        assert!(matches!(
            CxrCacheReader::open(&cache_dir, "train").unwrap_err(),
            CxrError::Io(_)
        ));

        write_synthetic_cache(&cache_dir, 2, 2, &["A", "B"]);
        let mut summary = read_cache_summary(&cache_dir).unwrap();
        summary.splits.get_mut("train").unwrap().metadata_path =
            "missing-metadata.jsonl".to_string();
        write_json(&cache_dir.join("cache-metadata.json"), &summary).unwrap();
        let validation = validate_cache_cxr(&ValidateCacheConfig {
            cache_dir,
            split: Some("train".to_string()),
            expected_targets: None,
            expected_image_shape: None,
            plan_path: None,
            report_path: None,
            json_path: None,
        })
        .unwrap();
        assert!(validation
            .errors
            .iter()
            .any(|error| error.contains("missing or unreadable metadata")));

        let csv_error = CxrError::from(csv::Error::from(std::io::Error::new(
            std::io::ErrorKind::Other,
            "csv-display",
        )));
        assert!(csv_error.to_string().contains("csv-display"));
        let json_error = CxrError::from(serde_json::from_str::<CxrRecord>("not json").unwrap_err());
        assert!(!json_error.to_string().is_empty());
        let image_error = CxrError::from(image::open(root.join("missing.png")).unwrap_err());
        assert!(image_error.to_string().contains("No such file"));
        let toml_error = CxrError::from(toml::from_str::<toml::Value>("[").unwrap_err());
        assert!(!toml_error.to_string().is_empty());
    }

    struct Fixture {
        manifest: PathBuf,
        splits: PathBuf,
        plan: PathBuf,
        cache_dir: PathBuf,
    }

    fn build_fixture(root: &Path) -> Fixture {
        fs::create_dir_all(root.join("images")).unwrap();
        write_png(&root.join("images/p1_i1.png"), 10);
        write_png(&root.join("images/p1_i2.png"), 20);
        write_png(&root.join("images/p2_i1.png"), 30);
        write_png(&root.join("images/p3_i1.png"), 40);
        fs::write(
            root.join("metadata.csv"),
            "dicom_id,subject_id,study_id,ViewPosition,Rows,Columns\np1_i1,1,10,PA,4,4\np1_i2,1,10,PA,4,4\np2_i1,2,20,AP,4,4\np3_i1,3,30,LL,4,4\n",
        )
        .unwrap();
        fs::write(
            root.join("labels.csv"),
            "subject_id,study_id,Pneumonia,No Finding\n1,10,1,0\n2,20,0,1\n3,30,-1,\n",
        )
        .unwrap();
        let plan = root.join("plan.toml");
        fs::write(&plan, "name = \"cxr-test\"\n[image]\nsize = [8, 8]\n").unwrap();

        let manifest = root.join("manifest.jsonl");
        let index = index_cxr(&IndexConfig {
            images_root: root.join("images"),
            metadata_path: Some(root.join("metadata.csv")),
            labels_path: Some(root.join("labels.csv")),
            reports_root: Some(root.join("reports")),
            out_path: manifest.clone(),
        })
        .unwrap();
        assert_eq!(index.records, 4);

        let splits = root.join("splits.json");
        let split = split_cxr(&SplitConfig {
            manifest_path: manifest.clone(),
            by: "patient_id".to_string(),
            train: 0.5,
            val: 0.25,
            test: 0.25,
            stratify: vec!["Pneumonia".to_string()],
            out_path: splits.clone(),
            seed: 7,
        })
        .unwrap();
        assert_eq!(split.patient_overlap_count, 0);

        Fixture {
            manifest,
            splits,
            plan,
            cache_dir: root.join("cache"),
        }
    }

    fn unique_test_dir() -> PathBuf {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let sequence = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "medkit-cxr-test-{}-{}-{}",
            std::process::id(),
            sequence,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_png(path: &Path, value: u8) {
        let image = image::GrayImage::from_pixel(4, 4, image::Luma([value]));
        image.save(path).unwrap();
    }

    fn write_rgb_png(path: &Path, value: [u8; 3]) {
        let image = image::RgbImage::from_pixel(4, 4, image::Rgb(value));
        image.save(path).unwrap();
    }

    fn write_synthetic_cache(
        cache_dir: &Path,
        samples: usize,
        image_size: usize,
        targets: &[&str],
    ) -> CacheSummary {
        fs::create_dir_all(cache_dir).unwrap();
        let images_path = cache_dir.join("train-images.float32.dat");
        let labels_path = cache_dir.join("train-labels.float32.dat");
        let masks_path = cache_dir.join("train-masks.float32.dat");
        let metadata_path = cache_dir.join("train-metadata.jsonl");
        write_f32_values(
            &images_path,
            &(0..samples * image_size * image_size)
                .map(|value| value as f32)
                .collect::<Vec<_>>(),
        );
        write_f32_values(
            &labels_path,
            &(0..samples * targets.len())
                .map(|value| value as f32)
                .collect::<Vec<_>>(),
        );
        write_f32_values(
            &masks_path,
            &(0..samples * targets.len())
                .map(|value| 100.0 + value as f32)
                .collect::<Vec<_>>(),
        );
        let mut metadata = BufWriter::new(File::create(&metadata_path).unwrap());
        for index in 0..samples {
            serde_json::to_writer(
                &mut metadata,
                &CxrRecord {
                    sample_id: format!("sample-{index}"),
                    patient_id: format!("patient-{index}"),
                    study_id: format!("study-{index}"),
                    image_id: format!("image-{index}"),
                    image_path: format!("image-{index}.png"),
                    source_format: "png".to_string(),
                    modality: Some("CR".to_string()),
                    view_position: Some("PA".to_string()),
                    laterality: None,
                    width: Some(image_size as u32),
                    height: Some(image_size as u32),
                    photometric_interpretation: Some("MONOCHROME2".to_string()),
                    labels: BTreeMap::new(),
                    label_source: None,
                    report_path: None,
                    split: Some("train".to_string()),
                    sha256: None,
                },
            )
            .unwrap();
            metadata.write_all(b"\n").unwrap();
        }
        metadata.write_all(b"\n").unwrap();
        metadata.flush().unwrap();

        let split = CacheSplitSummary {
            samples,
            shape: [samples, 1, image_size, image_size],
            images_path: "train-images.float32.dat".to_string(),
            labels_path: "train-labels.float32.dat".to_string(),
            masks_path: "train-masks.float32.dat".to_string(),
            metadata_path: "train-metadata.jsonl".to_string(),
        };
        let summary = CacheSummary {
            cache_schema_version: CXR_CACHE_SCHEMA_VERSION,
            report_schema_version: CXR_REPORT_SCHEMA_VERSION,
            cache_dir: cache_dir.display().to_string(),
            image_size,
            channels: 1,
            dtype: "float32".to_string(),
            targets: targets.iter().map(|target| (*target).to_string()).collect(),
            label_policy: LabelPolicy::default(),
            normalization: Normalization {
                mean: 0.0,
                std: 1.0,
            },
            transform_plan_hash: "hash".to_string(),
            transform_fingerprint: "hash".to_string(),
            source_manifest_checksum: "manifest".to_string(),
            split_names: vec!["train".to_string()],
            image_size_policy: ImageSizePolicy {
                channels: 1,
                height: image_size,
                width: image_size,
                dtype: "float32".to_string(),
                transform: "synthetic".to_string(),
            },
            splits: BTreeMap::from([("train".to_string(), split)]),
            failed_samples: Vec::new(),
            cache_size_bytes: directory_size(cache_dir).unwrap(),
        };
        write_json(&cache_dir.join("cache-metadata.json"), &summary).unwrap();
        summary
    }

    fn write_f32_values(path: &Path, values: &[f32]) {
        let mut writer = BufWriter::new(File::create(path).unwrap());
        for value in values {
            writer.write_all(&value.to_le_bytes()).unwrap();
        }
        writer.flush().unwrap();
    }
}
