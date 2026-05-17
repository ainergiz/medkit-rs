use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use medkit_cache::{prepare_cache, CacheManifest, PrepareConfig};
use medkit_cxr::{split_cxr, CxrRecord, SplitConfig};
use medkit_dataset::{
    validate_dataset, write_manifest_json, write_report, DatasetLayout, ValidationConfig,
};
use medkit_sampler::{ForegroundPrefix, LoadedCachedCase};
use medkit_transform::{TransformPlan, Volume3D, VolumeGeometry};
use serde::{Deserialize, Serialize};

use crate::Result;

const HEADER_LEN: usize = 348;

/// Canonical CXR manifest sizes used for scale fixture runs.
pub const CXR_MANIFEST_SCALE_RECORD_COUNTS: [usize; 3] = [1_000, 10_000, 100_000];

/// Configuration for a synthetic nnU-Net-shaped CT segmentation fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticFixtureConfig {
    /// Root directory that will contain `imagesTr`, `labelsTr`, and benchmark outputs.
    pub root: PathBuf,
    /// Number of image/label cases to generate.
    pub cases: usize,
    /// Source volume shape in x, y, z order.
    pub shape: [usize; 3],
    /// Source voxel spacing in x, y, z order.
    pub spacing: [f32; 3],
    /// Fixed pad/crop size used in the generated transform plan.
    pub cache_shape: [usize; 3],
    /// Target spacing used in the generated transform plan.
    pub resample_spacing: [f64; 3],
}

impl SyntheticFixtureConfig {
    /// Creates a fixture config with benchmark-friendly defaults.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            cases: 4,
            shape: [64, 64, 64],
            spacing: [1.0, 1.0, 1.0],
            cache_shape: [64, 64, 64],
            resample_spacing: [1.0, 1.0, 1.0],
        }
    }
}

/// Paths produced for a synthetic benchmark fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntheticFixture {
    /// Root directory.
    pub root: PathBuf,
    /// Synthetic image directory.
    pub images_dir: PathBuf,
    /// Synthetic label directory.
    pub labels_dir: PathBuf,
    /// Dataset validation manifest path.
    pub manifest_path: PathBuf,
    /// Dataset validation text report path.
    pub report_path: PathBuf,
    /// Transform plan path.
    pub plan_path: PathBuf,
    /// Cache output directory.
    pub cache_dir: PathBuf,
}

/// A synthetic fixture plus prepared medkit cache metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct CachedBenchmarkFixture {
    /// Synthetic fixture paths.
    pub fixture: SyntheticFixture,
    /// Cache manifest produced by `medkit-cache`.
    pub cache_manifest: CacheManifest,
}

/// Returns a unique temporary fixture root.
pub fn temp_fixture_root(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after UNIX_EPOCH")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "medkit-benchmarks-{name}-{}-{nanos}",
        std::process::id()
    ))
}

/// Configuration for a synthetic CXR DICOM manifest scale fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct CxrManifestScaleConfig {
    /// Root directory that will contain the manifest, split file, and report artifacts.
    pub root: PathBuf,
    /// Number of manifest records to generate.
    pub records: usize,
    /// Label targets to include in each synthetic record.
    pub targets: Vec<String>,
    /// Image size in width, height order.
    pub image_size: [u32; 2],
    /// Train split ratio.
    pub train: f64,
    /// Validation split ratio.
    pub val: f64,
    /// Test split ratio.
    pub test: f64,
    /// Patient-level split seed.
    pub seed: u64,
}

impl CxrManifestScaleConfig {
    /// Creates a CXR scale fixture config with benchmark-friendly defaults.
    pub fn new(root: impl Into<PathBuf>, records: usize) -> Self {
        Self {
            root: root.into(),
            records,
            targets: vec!["Pneumonia".to_string(), "No Finding".to_string()],
            image_size: [512, 512],
            train: 0.8,
            val: 0.1,
            test: 0.1,
            seed: 0,
        }
    }
}

/// Paths and workload dimensions produced for a CXR manifest scale fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CxrManifestScaleFixture {
    /// Root directory.
    pub root: PathBuf,
    /// JSONL CXR manifest path.
    pub manifest_path: PathBuf,
    /// Patient-level split file path.
    pub splits_path: PathBuf,
    /// Default benchmark report path.
    pub report_path: PathBuf,
    /// Number of manifest records.
    pub records: usize,
    /// Number of synthetic patients.
    pub patients: usize,
    /// Label targets included in the fixture.
    pub targets: Vec<String>,
    /// Image size in width, height order.
    pub image_size: [u32; 2],
}

/// Timings and byte counts measured by a CXR scale benchmark run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CxrScaleBenchmarkMetrics {
    /// DICOM records scanned.
    pub scan_records: usize,
    /// Scan elapsed wall time in milliseconds.
    pub scan_elapsed_ms: f64,
    /// Images preprocessed into cache tensors.
    pub preprocessed_images: usize,
    /// Preprocessing elapsed wall time in milliseconds.
    pub preprocessing_elapsed_ms: f64,
    /// Final cache size in bytes.
    pub cache_size_bytes: u64,
    /// Cache or manifest validation elapsed wall time in milliseconds.
    pub validation_elapsed_ms: f64,
    /// Python batches read.
    pub python_batches: usize,
    /// Python samples read.
    pub python_samples: usize,
    /// Python batch-read elapsed wall time in milliseconds.
    pub python_elapsed_ms: f64,
}

/// JSON benchmark report for CXR DICOM/manifest scale runs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CxrScaleBenchmarkReport {
    /// Report schema version.
    pub schema_version: u32,
    /// Workload dimensions and artifact paths.
    pub workload: CxrScaleWorkload,
    /// DICOM scan throughput.
    pub scan: CxrScanThroughput,
    /// Preprocessing/cache-build throughput.
    pub preprocessing: CxrPreprocessingThroughput,
    /// Final cache size in bytes.
    pub cache_size_bytes: u64,
    /// Validation timing.
    pub validation: CxrValidationTiming,
    /// Python batch-read throughput.
    pub python_batch: CxrPythonBatchThroughput,
}

/// Workload dimensions and artifact paths in a CXR scale benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CxrScaleWorkload {
    /// Number of manifest records.
    pub records: usize,
    /// Number of synthetic patients.
    pub patients: usize,
    /// Image size in width, height order.
    pub image_size: [u32; 2],
    /// Label targets.
    pub targets: Vec<String>,
    /// JSONL CXR manifest path.
    pub manifest_path: String,
    /// Patient-level split file path.
    pub splits_path: String,
}

/// DICOM scan throughput metric.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CxrScanThroughput {
    /// Records scanned.
    pub records: usize,
    /// Elapsed wall time in milliseconds.
    pub elapsed_ms: f64,
    /// Records scanned per second.
    pub records_per_second: f64,
}

/// CXR preprocessing throughput metric.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CxrPreprocessingThroughput {
    /// Images preprocessed.
    pub images: usize,
    /// Elapsed wall time in milliseconds.
    pub elapsed_ms: f64,
    /// Images preprocessed per second.
    pub images_per_second: f64,
}

/// Validation elapsed-time metric.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CxrValidationTiming {
    /// Elapsed wall time in milliseconds.
    pub elapsed_ms: f64,
}

/// Python cache batch-read throughput metric.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CxrPythonBatchThroughput {
    /// Batches read from Python.
    pub batches: usize,
    /// Samples read from Python.
    pub samples: usize,
    /// Elapsed wall time in milliseconds.
    pub elapsed_ms: f64,
    /// Python batches read per second.
    pub batches_per_second: f64,
    /// Python samples read per second.
    pub samples_per_second: f64,
}

/// Generates a synthetic CXR manifest and deterministic patient-level splits.
pub fn create_cxr_manifest_scale_fixture(
    config: &CxrManifestScaleConfig,
) -> Result<CxrManifestScaleFixture> {
    validate_cxr_manifest_scale_config(config)?;
    fs::create_dir_all(&config.root)?;
    let manifest_path = config.root.join("cxr-manifest.jsonl");
    let splits_path = config.root.join("cxr-splits.json");
    let report_path = config.root.join("cxr-scale-benchmark-report.json");

    write_cxr_scale_manifest(&manifest_path, config)?;
    split_cxr(&SplitConfig {
        manifest_path: manifest_path.clone(),
        by: "patient_id".to_string(),
        train: config.train,
        val: config.val,
        test: config.test,
        stratify: Vec::new(),
        out_path: splits_path.clone(),
        seed: config.seed,
    })?;

    Ok(CxrManifestScaleFixture {
        root: config.root.clone(),
        manifest_path,
        splits_path,
        report_path,
        records: config.records,
        patients: cxr_scale_patient_count(config.records),
        targets: config.targets.clone(),
        image_size: config.image_size,
    })
}

/// Builds a JSON report for measured CXR DICOM/manifest scale metrics.
pub fn cxr_scale_benchmark_report(
    fixture: &CxrManifestScaleFixture,
    metrics: CxrScaleBenchmarkMetrics,
) -> CxrScaleBenchmarkReport {
    CxrScaleBenchmarkReport {
        schema_version: 1,
        workload: CxrScaleWorkload {
            records: fixture.records,
            patients: fixture.patients,
            image_size: fixture.image_size,
            targets: fixture.targets.clone(),
            manifest_path: fixture.manifest_path.display().to_string(),
            splits_path: fixture.splits_path.display().to_string(),
        },
        scan: CxrScanThroughput {
            records: metrics.scan_records,
            elapsed_ms: metrics.scan_elapsed_ms,
            records_per_second: per_second(metrics.scan_records, metrics.scan_elapsed_ms),
        },
        preprocessing: CxrPreprocessingThroughput {
            images: metrics.preprocessed_images,
            elapsed_ms: metrics.preprocessing_elapsed_ms,
            images_per_second: per_second(
                metrics.preprocessed_images,
                metrics.preprocessing_elapsed_ms,
            ),
        },
        cache_size_bytes: metrics.cache_size_bytes,
        validation: CxrValidationTiming {
            elapsed_ms: metrics.validation_elapsed_ms,
        },
        python_batch: CxrPythonBatchThroughput {
            batches: metrics.python_batches,
            samples: metrics.python_samples,
            elapsed_ms: metrics.python_elapsed_ms,
            batches_per_second: per_second(metrics.python_batches, metrics.python_elapsed_ms),
            samples_per_second: per_second(metrics.python_samples, metrics.python_elapsed_ms),
        },
    }
}

/// Writes a CXR scale benchmark report as pretty JSON.
pub fn write_cxr_scale_benchmark_report(
    path: &Path,
    report: &CxrScaleBenchmarkReport,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(report)?)?;
    Ok(())
}

fn validate_cxr_manifest_scale_config(config: &CxrManifestScaleConfig) -> Result<()> {
    if config.records == 0 {
        return Err("records must be greater than zero".into());
    }
    if config.targets.is_empty() {
        return Err("targets must contain at least one label target".into());
    }
    if config.targets.iter().any(|target| target.trim().is_empty()) {
        return Err("targets must not contain empty names".into());
    }
    if config.image_size.contains(&0) {
        return Err(format!("image_size must be non-zero, got {:?}", config.image_size).into());
    }
    let ratio_sum = config.train + config.val + config.test;
    if !ratio_sum.is_finite() || (ratio_sum - 1.0).abs() > 1.0e-6 {
        return Err(format!("train+val+test must equal 1.0, got {ratio_sum}").into());
    }
    Ok(())
}

fn write_cxr_scale_manifest(path: &Path, config: &CxrManifestScaleConfig) -> Result<()> {
    let mut file = File::create(path)?;
    for index in 0..config.records {
        let record = synthetic_cxr_scale_record(config, index);
        serde_json::to_writer(&mut file, &record)?;
        file.write_all(b"\n")?;
    }
    Ok(())
}

fn synthetic_cxr_scale_record(config: &CxrManifestScaleConfig, index: usize) -> CxrRecord {
    let patient_index = index / 2;
    let study_index = index / 2;
    let series_index = index / 4;
    let image_id = format!("synthetic-cxr-{index:08}");
    let patient_id = format!("synthetic-patient-{patient_index:06}");
    let study_id = format!("synthetic-study-{study_index:06}");
    let sample_id = format!("{patient_id}/{study_id}/{image_id}");
    let image_path = config
        .root
        .join("raw-dicom")
        .join(&patient_id)
        .join(format!("{image_id}.dcm"))
        .display()
        .to_string();
    CxrRecord {
        sample_id,
        patient_id,
        study_id,
        image_id,
        image_path,
        source_format: "dicom".to_string(),
        modality: Some(if index % 3 == 0 { "DX" } else { "CR" }.to_string()),
        view_position: Some(if index % 2 == 0 { "PA" } else { "AP" }.to_string()),
        laterality: Some(if index % 2 == 0 { "L" } else { "R" }.to_string()),
        width: Some(config.image_size[0]),
        height: Some(config.image_size[1]),
        photometric_interpretation: Some("MONOCHROME2".to_string()),
        series_instance_uid: Some(format!("1.2.826.0.1.3680043.10.5432.1.{series_index}")),
        sop_instance_uid: Some(format!("1.2.826.0.1.3680043.10.5432.2.{index}")),
        transfer_syntax_uid: Some("1.2.840.10008.1.2.1".to_string()),
        pixel_hash: Some(format!("synthetic-pixel-hash-{index:016x}")),
        labels: synthetic_cxr_scale_labels(&config.targets, index),
        label_source: Some("synthetic-scale-fixture".to_string()),
        report_path: None,
        split: None,
        sha256: Some(format!("synthetic-dicom-hash-{index:016x}")),
    }
}

fn synthetic_cxr_scale_labels(targets: &[String], index: usize) -> BTreeMap<String, Option<i8>> {
    targets
        .iter()
        .enumerate()
        .map(|(target_index, target)| {
            let value = match (index + target_index) % 5 {
                0 => Some(1),
                1 | 2 => Some(0),
                3 => Some(-1),
                _ => None,
            };
            (target.clone(), value)
        })
        .collect()
}

fn cxr_scale_patient_count(records: usize) -> usize {
    records.div_ceil(2)
}

fn per_second(items: usize, elapsed_ms: f64) -> f64 {
    if items == 0 || !elapsed_ms.is_finite() || elapsed_ms <= 0.0 {
        0.0
    } else {
        items as f64 / (elapsed_ms / 1000.0)
    }
}

/// Generates synthetic image/label NIfTI files and a transform plan.
pub fn create_synthetic_fixture(config: &SyntheticFixtureConfig) -> Result<SyntheticFixture> {
    validate_config(config)?;
    let images_dir = config.root.join("imagesTr");
    let labels_dir = config.root.join("labelsTr");
    fs::create_dir_all(&images_dir)?;
    fs::create_dir_all(&labels_dir)?;

    for case_index in 0..config.cases {
        let case_id = format!("case_{case_index:04}");
        let (image, label, _) =
            synthetic_volume_pair(config.shape, config.spacing.map(f64::from), case_index)?;
        let image_path = images_dir.join(format!("{case_id}_0000.nii"));
        let label_path = labels_dir.join(format!("{case_id}.nii"));
        let image_i16 = image
            .data
            .iter()
            .map(|value| value.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16)
            .collect::<Vec<_>>();
        let label_u8 = label
            .data
            .iter()
            .map(|value| (*value).min(u8::MAX as u16) as u8)
            .collect::<Vec<_>>();
        write_i16_nifti(&image_path, config.shape, config.spacing, &image_i16)?;
        write_u8_nifti(&label_path, config.shape, config.spacing, &label_u8)?;
    }

    let fixture = SyntheticFixture {
        root: config.root.clone(),
        images_dir,
        labels_dir,
        manifest_path: config.root.join("manifest.json"),
        report_path: config.root.join("report.txt"),
        plan_path: config.root.join("ct-segmentation.toml"),
        cache_dir: config.root.join(".medkit").join("cache"),
    };
    fs::write(&fixture.plan_path, transform_plan_toml(config))?;
    Ok(fixture)
}

/// Generates a fixture, validates it, and prepares the medkit cache.
pub fn build_cached_fixture(config: &SyntheticFixtureConfig) -> Result<CachedBenchmarkFixture> {
    let fixture = create_synthetic_fixture(config)?;
    let validation =
        validate_dataset(&ValidationConfig::new(&fixture.root).layout(DatasetLayout::Nnunet))?;
    write_manifest_json(&validation, &fixture.manifest_path)?;
    write_report(&validation, &fixture.report_path)?;
    let cache_manifest = prepare_cache(&PrepareConfig {
        dataset_root: fixture.root.clone(),
        manifest_path: fixture.manifest_path.clone(),
        plan_path: fixture.plan_path.clone(),
        cache_dir: fixture.cache_dir.clone(),
        chunk_shape: None,
    })?;
    Ok(CachedBenchmarkFixture {
        fixture,
        cache_manifest,
    })
}

/// Creates an in-memory synthetic image/label pair with explicit geometry.
pub fn synthetic_volume_pair(
    shape: [usize; 3],
    spacing: [f64; 3],
    case_index: usize,
) -> Result<(Volume3D<f32>, Volume3D<u16>, VolumeGeometry)> {
    if shape.contains(&0) {
        return Err(format!("shape must be non-zero, got {shape:?}").into());
    }
    let mut image = Vec::with_capacity(shape[0] * shape[1] * shape[2]);
    let mut label = Vec::with_capacity(image.capacity());
    let center = [
        (shape[0] - 1) as f32 * 0.5,
        (shape[1] - 1) as f32 * 0.5,
        (shape[2] - 1) as f32 * 0.5,
    ];
    let radius = shape.iter().copied().min().unwrap_or(1) as f32 * 0.18;
    for z in 0..shape[2] {
        for y in 0..shape[1] {
            for x in 0..shape[0] {
                let dx = x as f32 - center[0];
                let dy = y as f32 - center[1];
                let dz = z as f32 - center[2];
                let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                image.push(
                    -850.0
                        + x as f32 * 3.0
                        + y as f32 * 1.5
                        + z as f32 * 0.75
                        + case_index as f32 * 17.0,
                );
                label.push(u16::from(distance <= radius));
            }
        }
    }
    Ok((
        Volume3D::new(shape, image)?,
        Volume3D::new(shape, label)?,
        VolumeGeometry::identity(shape, spacing)?,
    ))
}

/// Creates an in-memory loaded cached case for sampler benchmarks.
pub fn synthetic_loaded_case(shape: [usize; 3]) -> Result<LoadedCachedCase> {
    let (image, label, geometry) = synthetic_volume_pair(shape, [1.0, 1.0, 1.0], 0)?;
    let foreground_indices: Vec<usize> = label
        .data
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value != 0).then_some(index))
        .collect();
    let foreground_prefix = ForegroundPrefix::from_label(&label)?;
    Ok(LoadedCachedCase {
        metadata: medkit_cache::CachedCase {
            case_id: "synthetic".to_string(),
            cache_key: "synthetic-key".to_string(),
            source_metadata_hash: "source".to_string(),
            transform_plan_hash: "plan".to_string(),
            image_path: "image.nii".to_string(),
            label_path: "label.nii".to_string(),
            source_geometry: geometry,
            output_geometry: geometry,
            image_cache_path: "image.f32.raw".to_string(),
            label_cache_path: "label.u16.raw".to_string(),
            foreground_indices_path: None,
            foreground_prefix_path: None,
            foreground_prefix_shape: None,
            image_chunk_cache_path: None,
            label_chunk_cache_path: None,
            shape,
            chunk_shape: shape,
            chunk_grid: None,
            crop_origin: [0, 0, 0],
            applied_operations: vec!["synthetic".to_string()],
            foreground_voxels: foreground_indices.len(),
            bytes_written: shape[0] * shape[1] * shape[2] * 6,
        },
        image,
        label,
        foreground_indices,
        foreground_prefix,
    })
}

/// Creates a default transform plan for microbenchmarks.
pub fn transform_plan(config: &SyntheticFixtureConfig) -> Result<TransformPlan> {
    Ok(TransformPlan::from_toml_str(&transform_plan_toml(config))?)
}

fn validate_config(config: &SyntheticFixtureConfig) -> Result<()> {
    if config.cases == 0 {
        return Err("cases must be greater than zero".into());
    }
    if config.shape.contains(&0) {
        return Err(format!("shape must be non-zero, got {:?}", config.shape).into());
    }
    if config.cache_shape.contains(&0) {
        return Err(format!("cache shape must be non-zero, got {:?}", config.cache_shape).into());
    }
    if config
        .spacing
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(format!(
            "spacing must be finite and positive, got {:?}",
            config.spacing
        )
        .into());
    }
    if config
        .resample_spacing
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(format!(
            "resample spacing must be finite and positive, got {:?}",
            config.resample_spacing
        )
        .into());
    }
    Ok(())
}

fn transform_plan_toml(config: &SyntheticFixtureConfig) -> String {
    format!(
        r#"name = "ct-segmentation-benchmark"
image_interpolation = "linear"
label_interpolation = "nearest"

[[operations]]
op = "resample"
spacing = [{:.8}, {:.8}, {:.8}]

[[operations]]
op = "ct_window"
min = -1000.0
max = 1000.0

[[operations]]
op = "min_max_normalize"

[[operations]]
op = "crop_foreground"
margin = 4

[[operations]]
op = "pad_crop"
size = [{}, {}, {}]
"#,
        config.resample_spacing[0],
        config.resample_spacing[1],
        config.resample_spacing[2],
        config.cache_shape[0],
        config.cache_shape[1],
        config.cache_shape[2]
    )
}

fn write_i16_nifti(
    path: &Path,
    shape: [usize; 3],
    spacing: [f32; 3],
    values: &[i16],
) -> Result<()> {
    let mut bytes = header(shape, spacing, 4, 16);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn write_u8_nifti(path: &Path, shape: [usize; 3], spacing: [f32; 3], values: &[u8]) -> Result<()> {
    let mut bytes = header(shape, spacing, 2, 8);
    bytes.extend_from_slice(values);
    fs::write(path, bytes)?;
    Ok(())
}

fn header(shape: [usize; 3], spacing: [f32; 3], datatype: i16, bitpix: i16) -> Vec<u8> {
    let mut bytes = [0_u8; HEADER_LEN];
    put_i32(&mut bytes, 0, 348);
    put_i16(&mut bytes, 40, 3);
    put_i16(&mut bytes, 42, shape[0] as i16);
    put_i16(&mut bytes, 44, shape[1] as i16);
    put_i16(&mut bytes, 46, shape[2] as i16);
    put_i16(&mut bytes, 70, datatype);
    put_i16(&mut bytes, 72, bitpix);
    put_f32(&mut bytes, 76, 1.0);
    put_f32(&mut bytes, 80, spacing[0]);
    put_f32(&mut bytes, 84, spacing[1]);
    put_f32(&mut bytes, 88, spacing[2]);
    put_f32(&mut bytes, 108, 352.0);
    bytes[344..348].copy_from_slice(b"n+1\0");
    let mut out = bytes.to_vec();
    out.extend_from_slice(&[0, 0, 0, 0]);
    out
}

fn put_i32(bytes: &mut [u8; HEADER_LEN], offset: usize, value: i32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_i16(bytes: &mut [u8; HEADER_LEN], offset: usize, value: i16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_f32(bytes: &mut [u8; HEADER_LEN], offset: usize, value: f32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
