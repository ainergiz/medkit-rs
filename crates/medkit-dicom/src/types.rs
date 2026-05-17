use std::{collections::BTreeMap, path::PathBuf};

use serde::{Deserialize, Serialize};

pub const EXPLICIT_VR_LITTLE_ENDIAN: &str = "1.2.840.10008.1.2.1";
pub const IMPLICIT_VR_LITTLE_ENDIAN: &str = "1.2.840.10008.1.2";
pub const EXPLICIT_VR_BIG_ENDIAN: &str = "1.2.840.10008.1.2.2";
pub const RLE_LOSSLESS: &str = "1.2.840.10008.1.2.5";
pub const JPEG_BASELINE_8BIT: &str = "1.2.840.10008.1.2.4.50";
pub const JPEG_EXTENDED_12BIT: &str = "1.2.840.10008.1.2.4.51";
pub const JPEG_LOSSLESS: &str = "1.2.840.10008.1.2.4.57";
pub const JPEG_LOSSLESS_SV1: &str = "1.2.840.10008.1.2.4.70";
pub const JPEG_LS_LOSSLESS: &str = "1.2.840.10008.1.2.4.80";
pub const JPEG_LS_LOSSY: &str = "1.2.840.10008.1.2.4.81";
pub const JPEG_2000_LOSSLESS: &str = "1.2.840.10008.1.2.4.90";
pub const JPEG_2000: &str = "1.2.840.10008.1.2.4.91";

pub fn is_native_transfer_syntax(transfer_syntax_uid: &str) -> bool {
    matches!(
        transfer_syntax_uid,
        EXPLICIT_VR_LITTLE_ENDIAN
            | IMPLICIT_VR_LITTLE_ENDIAN
            | EXPLICIT_VR_BIG_ENDIAN
            | RLE_LOSSLESS
            | JPEG_BASELINE_8BIT
    )
}

pub fn is_dicom_rs_transfer_syntax(transfer_syntax_uid: &str) -> bool {
    if !cfg!(feature = "dicom-rs-codecs") {
        return false;
    }
    matches!(
        transfer_syntax_uid,
        EXPLICIT_VR_LITTLE_ENDIAN
            | IMPLICIT_VR_LITTLE_ENDIAN
            | EXPLICIT_VR_BIG_ENDIAN
            | RLE_LOSSLESS
            | JPEG_BASELINE_8BIT
            | JPEG_EXTENDED_12BIT
            | JPEG_LOSSLESS
            | JPEG_LOSSLESS_SV1
    )
}

pub fn is_known_explicit_vr_little_endian_transfer_syntax(transfer_syntax_uid: &str) -> bool {
    matches!(
        transfer_syntax_uid,
        EXPLICIT_VR_LITTLE_ENDIAN
            | RLE_LOSSLESS
            | JPEG_BASELINE_8BIT
            | JPEG_EXTENDED_12BIT
            | JPEG_LOSSLESS
            | JPEG_LOSSLESS_SV1
            | JPEG_LS_LOSSLESS
            | JPEG_LS_LOSSY
            | JPEG_2000_LOSSLESS
            | JPEG_2000
    )
}

pub fn is_uncompressed_transfer_syntax(transfer_syntax_uid: &str) -> bool {
    matches!(
        transfer_syntax_uid,
        EXPLICIT_VR_LITTLE_ENDIAN | IMPLICIT_VR_LITTLE_ENDIAN | EXPLICIT_VR_BIG_ENDIAN
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DicomScanConfig {
    pub root: PathBuf,
    pub out_path: PathBuf,
    pub report_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DicomBrowseConfig {
    pub root: PathBuf,
    pub group: Vec<String>,
    pub out_path: PathBuf,
    pub report_path: PathBuf,
    pub workers: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DicomFileConfig {
    pub path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DicomViewConfig {
    pub path: PathBuf,
    pub width: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DicomInventoryRecord {
    pub path: String,
    pub sha256: String,
    pub patient_id: Option<String>,
    pub study_instance_uid: Option<String>,
    pub series_instance_uid: Option<String>,
    pub sop_instance_uid: Option<String>,
    pub modality: Option<String>,
    pub body_part_examined: Option<String>,
    pub view_position: Option<String>,
    pub laterality: Option<String>,
    pub rows: Option<u16>,
    pub columns: Option<u16>,
    pub samples_per_pixel: Option<u16>,
    pub bits_allocated: Option<u16>,
    pub bits_stored: Option<u16>,
    pub high_bit: Option<u16>,
    pub pixel_representation: Option<String>,
    pub photometric_interpretation: Option<String>,
    pub transfer_syntax_uid: String,
    pub pixel_spacing: Option<[f32; 2]>,
    pub imager_pixel_spacing: Option<[f32; 2]>,
    pub rescale_intercept: Option<f32>,
    pub rescale_slope: Option<f32>,
    pub window_center: Option<f32>,
    pub window_width: Option<f32>,
    pub pixel_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoder_backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoder_version: Option<String>,
    pub warnings: Vec<DicomWarning>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DicomWarning {
    pub code: String,
    pub message: String,
}

impl DicomWarning {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DicomScanError {
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DicomScanSummary {
    pub root: String,
    pub records: usize,
    pub errors: Vec<DicomScanError>,
    pub warnings: usize,
    pub duplicate_sop_instance_uids: usize,
    pub duplicate_pixel_hashes: usize,
    pub out_path: String,
    pub report_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DicomInspectReport {
    pub record: DicomInventoryRecord,
    pub elements: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DicomGraphSummary {
    pub root: String,
    pub patients: usize,
    pub studies: usize,
    pub series: usize,
    pub instances: usize,
    pub duplicate_sop_instance_uids: usize,
    pub duplicate_pixel_hashes: usize,
    pub warnings: Vec<DicomGraphWarning>,
    pub patients_detail: Vec<DicomPatientNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DicomGraphWarning {
    pub code: String,
    pub message: String,
    pub path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DicomPatientNode {
    pub patient_id: String,
    pub studies: Vec<DicomStudyNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DicomStudyNode {
    pub study_instance_uid: String,
    pub series: Vec<DicomSeriesNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DicomSeriesNode {
    pub series_instance_uid: String,
    pub modality: Option<String>,
    pub rows: Option<u16>,
    pub columns: Option<u16>,
    pub instances: Vec<DicomInstanceNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DicomInstanceNode {
    pub sop_instance_uid: Option<String>,
    pub path: String,
    pub modality: Option<String>,
    pub rows: Option<u16>,
    pub columns: Option<u16>,
    pub pixel_hash: Option<String>,
}
