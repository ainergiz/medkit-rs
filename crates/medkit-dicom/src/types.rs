use std::{collections::BTreeMap, path::PathBuf};

use serde::{Deserialize, Serialize};

pub const EXPLICIT_VR_LITTLE_ENDIAN: &str = "1.2.840.10008.1.2.1";
pub const IMPLICIT_VR_LITTLE_ENDIAN: &str = "1.2.840.10008.1.2";
pub const EXPLICIT_VR_BIG_ENDIAN: &str = "1.2.840.10008.1.2.2";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DicomScanConfig {
    pub root: PathBuf,
    pub out_path: PathBuf,
    pub report_path: PathBuf,
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
