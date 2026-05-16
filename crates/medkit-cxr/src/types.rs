use std::{collections::BTreeMap, path::PathBuf};

use serde::{Deserialize, Serialize};

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
    pub(crate) cache_dir: PathBuf,
    pub(crate) split: String,
    pub(crate) summary: CacheSummary,
    pub(crate) split_summary: CacheSplitSummary,
    pub(crate) records: Vec<CxrRecord>,
    pub(crate) image_values_per_sample: usize,
    pub(crate) target_count: usize,
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
