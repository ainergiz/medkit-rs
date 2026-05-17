use std::{collections::BTreeMap, path::PathBuf};

use serde::{Deserialize, Serialize};

pub const CXR_CACHE_SCHEMA_VERSION: u32 = 1;
pub const CXR_REPORT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct IndexConfig {
    pub images_root: PathBuf,
    pub dicom_index_path: Option<PathBuf>,
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

#[derive(Debug, Clone)]
pub struct IngestConfig {
    pub raw_root: PathBuf,
    pub recipe_path: PathBuf,
    pub labels_path: PathBuf,
    pub cache_dir: PathBuf,
    pub workdir: PathBuf,
    pub report_path: PathBuf,
    pub dry_run: bool,
    pub workers: usize,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub series_instance_uid: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sop_instance_uid: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transfer_syntax_uid: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pixel_hash: Option<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dicom_index_path: Option<String>,
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
    pub recipe_hash: String,
    #[serde(default)]
    pub recipe_path: String,
    #[serde(default)]
    pub source_manifest_checksum: String,
    #[serde(default)]
    pub split_names: Vec<String>,
    #[serde(default)]
    pub image_size_policy: ImageSizePolicy,
    #[serde(default)]
    pub dicom_presentation_policy: DicomPresentationPolicy,
    #[serde(default)]
    pub transfer_syntax_policy: TransferSyntaxPolicy,
    #[serde(default)]
    pub split_policy: SplitPolicyMetadata,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ImageSizePolicy {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub dtype: String,
    pub transform: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DicomPresentationPolicy {
    pub apply_rescale: bool,
    pub voi: String,
    pub invert_monochrome1: bool,
    pub output: String,
    pub decoder_backend: String,
    pub decoder_version: String,
}

impl Default for DicomPresentationPolicy {
    fn default() -> Self {
        Self {
            apply_rescale: true,
            voi: "auto".to_string(),
            invert_monochrome1: true,
            output: "mono8".to_string(),
            decoder_backend: "medkit-native".to_string(),
            decoder_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TransferSyntaxPolicy {
    pub allow_transfer_syntaxes: Vec<String>,
    pub unsupported_transfer_syntax: String,
}

impl Default for TransferSyntaxPolicy {
    fn default() -> Self {
        Self {
            allow_transfer_syntaxes: vec![
                medkit_dicom::IMPLICIT_VR_LITTLE_ENDIAN.to_string(),
                medkit_dicom::EXPLICIT_VR_LITTLE_ENDIAN.to_string(),
                medkit_dicom::EXPLICIT_VR_BIG_ENDIAN.to_string(),
                medkit_dicom::RLE_LOSSLESS.to_string(),
            ],
            unsupported_transfer_syntax: "fail".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SplitPolicyMetadata {
    pub by: String,
    pub train: f64,
    pub val: f64,
    pub test: f64,
    pub stratify: Vec<String>,
    pub seed: u64,
}

impl Default for SplitPolicyMetadata {
    fn default() -> Self {
        Self {
            by: "patient_id".to_string(),
            train: 0.8,
            val: 0.1,
            test: 0.1,
            stratify: Vec::new(),
            seed: 0,
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
    pub recipe_hash: String,
    pub source_manifest_checksum: String,
    pub cache_size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestPaths {
    pub recipe: String,
    pub raw_dicom_root: String,
    pub labels: String,
    pub workdir: String,
    pub dicom_index: String,
    pub dicom_scan_report: String,
    pub recipe_dicom_index: String,
    pub manifest: String,
    pub validation_report: String,
    pub splits: String,
    pub cache_dir: String,
    pub cache_validation_report: String,
    pub cache_validation_json: String,
    pub ingest_report: String,
    pub ingest_summary_json: String,
    pub resume_state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestSummary {
    pub dry_run: bool,
    pub status: String,
    pub recipe_name: String,
    pub recipe_hash: String,
    pub paths: IngestPaths,
    pub planned_actions: Vec<String>,
    pub validation_rules: Vec<String>,
    pub counts: IngestCounts,
    pub modality_distribution: BTreeMap<String, usize>,
    pub view_position_distribution: BTreeMap<String, usize>,
    pub transfer_syntax_distribution: BTreeMap<String, usize>,
    pub rows_summary: NumericSummary,
    pub columns_summary: NumericSummary,
    pub pixel_spacing_summary: PixelSpacingSummary,
    pub missing_identifier_counts: BTreeMap<String, usize>,
    pub missing_label_counts: BTreeMap<String, usize>,
    pub label_distribution: BTreeMap<String, LabelCount>,
    pub label_distribution_by_split: BTreeMap<String, BTreeMap<String, LabelCount>>,
    pub patient_counts_by_split: BTreeMap<String, usize>,
    pub patient_overlap_count: usize,
    pub skipped_samples: Vec<IngestSampleIssue>,
    pub failed_preprocessing_samples: Vec<String>,
    pub scan_error_counts: BTreeMap<String, usize>,
    pub warning_counts: BTreeMap<String, usize>,
    pub duplicate_sop_instance_uid_count: usize,
    pub duplicate_pixel_hash_count: usize,
    pub cache_transform_fingerprint: String,
    pub cache_validation_status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IngestCounts {
    pub patients: usize,
    pub studies: usize,
    pub series: usize,
    pub images: usize,
    pub dicom_records_scanned: usize,
    pub manifest_records: usize,
    pub unsupported_or_skipped_images: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NumericSummary {
    pub count: usize,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub mean: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PixelSpacingSummary {
    pub row_spacing: NumericSummary,
    pub column_spacing: NumericSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestSampleIssue {
    pub path: String,
    pub reason: String,
    pub code: String,
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
