use std::{
    error::Error,
    fs, io,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use medkit_cache::{CacheError, CacheManifest, CacheSummary, CachedCase};
use medkit_sampler::{
    extract_patch_pair, extract_patch_pair_into, load_cached_cases, plan_batches, sample_cache,
    CachedImageVolume, ForegroundPrefix, LoadedCachedCase, PatchRecord, SampleConfig, SamplerError,
    SamplingStrategy,
};
use medkit_transform::{TransformPlan, Volume3D, VolumeGeometry};

#[test]
fn sampler_errors_report_sources_and_messages() {
    let io_error = SamplerError::Io {
        path: PathBuf::from("samples.jsonl"),
        source: io::Error::new(io::ErrorKind::PermissionDenied, "denied"),
    };
    assert_eq!(
        io_error.to_string(),
        "filesystem error at samples.jsonl: denied"
    );
    assert_eq!(io_error.source().unwrap().to_string(), "denied");

    let cache_error = SamplerError::Cache(CacheError::InvalidInput {
        reason: "bad cache".to_string(),
    });
    assert_eq!(cache_error.to_string(), "invalid cache input: bad cache");
    assert!(cache_error.source().is_some());

    let json_source = serde_json::from_str::<PatchRecord>("{").unwrap_err();
    let json_error = SamplerError::Json(json_source);
    assert!(json_error
        .to_string()
        .starts_with("failed to write sample JSONL:"));
    assert!(json_error.source().is_some());

    let invalid = SamplerError::InvalidInput {
        reason: "patch size must be non-zero".to_string(),
    };
    assert_eq!(
        invalid.to_string(),
        "invalid sampler input: patch size must be non-zero"
    );
    assert!(invalid.source().is_none());
}

#[test]
fn planning_and_patch_guards_return_invalid_input() {
    let batch_error = plan_batches(Vec::new(), 0).unwrap_err();
    assert_invalid_input(batch_error, "batch size must be greater than zero");

    let case = loaded_case();
    let size_error = extract_patch_pair(&case, [0, 0, 0], [0, 1, 1]).unwrap_err();
    assert_invalid_input(size_error, "patch size must be non-zero");

    let start_error = extract_patch_pair(&case, [3, 0, 0], [2, 1, 1]).unwrap_err();
    assert_invalid_input(start_error, "patch start");

    let shape_error = extract_patch_pair(&case, [0, 0, 0], [5, 1, 1]).unwrap_err();
    assert_invalid_input(shape_error, "patch size");

    let mut short_image = vec![0.0; 7];
    let mut label = vec![0_u16; 8];
    let image_error =
        extract_patch_pair_into(&case, [0, 0, 0], [2, 2, 2], &mut short_image, &mut label)
            .unwrap_err();
    assert_invalid_input(image_error, "image output buffer");

    let mut image = vec![0.0; 8];
    let mut short_label = vec![0_u16; 7];
    let label_error =
        extract_patch_pair_into(&case, [0, 0, 0], [2, 2, 2], &mut image, &mut short_label)
            .unwrap_err();
    assert_invalid_input(label_error, "label output buffer");

    let prefix_error = ForegroundPrefix::from_values([2, 2, 2], vec![0; 3]).unwrap_err();
    assert_invalid_input(prefix_error, "foreground prefix");

    let prefix =
        ForegroundPrefix::from_label(&Volume3D::new([2, 2, 2], vec![0_u16; 8]).unwrap()).unwrap();
    let bounds_error = prefix.count_checked([2, 0, 0], [1, 1, 1]).unwrap_err();
    assert_invalid_input(bounds_error, "patch start");
}

#[test]
fn sample_cache_writes_jsonl_and_falls_back_to_label_foreground() {
    let cache_dir = temp_dir("sample-cache-success");
    let shape = [3, 1, 1];
    write_f32_raw(&cache_dir.join("image.f32.raw"), &[0.0, 1.0, 2.0]);
    write_u16_raw(&cache_dir.join("label.u16.raw"), &[0, 9, 0]);
    write_cache_manifest(&cache_dir, vec![cached_case(&cache_dir, shape)]);
    let out_path = cache_dir.join("out").join("samples.jsonl");

    let summary = sample_cache(&SampleConfig {
        cache_dir: cache_dir.clone(),
        patch_size: [1, 1, 1],
        strategy: SamplingStrategy::ForegroundBalanced,
        count: 4,
        out_path: out_path.clone(),
        seed: 7,
        epoch: 2,
        worker: 3,
    })
    .unwrap();

    assert_eq!(summary.records, 4);
    assert_eq!(summary.foreground_records + summary.background_records, 4);
    let records = fs::read_to_string(out_path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<PatchRecord>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(records.len(), 4);
    assert_eq!(records[0].patch_start, [1, 0, 0]);
    assert!(records[0].has_foreground);
    assert_eq!(records[2].patch_start, [1, 0, 0]);
    assert!(records[2].has_foreground);
    assert!(records
        .iter()
        .all(|record| record.case_id == "case" && record.epoch == 2 && record.worker == 3));
}

#[test]
fn sample_cache_writes_output_without_parent_directory() {
    let cache_dir = temp_dir("sample-cache-current-dir");
    let shape = [1, 1, 1];
    write_f32_raw(&cache_dir.join("image.f32.raw"), &[1.0]);
    write_u16_raw(&cache_dir.join("label.u16.raw"), &[1]);
    write_cache_manifest(&cache_dir, vec![cached_case(&cache_dir, shape)]);
    let original_dir = std::env::current_dir().unwrap();
    std::env::set_current_dir(&cache_dir).unwrap();

    let summary = sample_cache(&SampleConfig {
        cache_dir: cache_dir.clone(),
        patch_size: [1, 1, 1],
        strategy: SamplingStrategy::ForegroundBalanced,
        count: 1,
        out_path: PathBuf::from("samples.jsonl"),
        seed: 0,
        epoch: 0,
        worker: 0,
    })
    .unwrap();

    std::env::set_current_dir(original_dir).unwrap();
    assert_eq!(summary.records, 1);
    assert!(cache_dir.join("samples.jsonl").is_file());
}

#[test]
fn sample_cache_rejects_empty_cache_manifest() {
    let cache_dir = temp_dir("sample-cache-empty");
    write_cache_manifest(&cache_dir, Vec::new());

    let error = sample_cache(&SampleConfig {
        cache_dir: cache_dir.clone(),
        patch_size: [1, 1, 1],
        strategy: SamplingStrategy::ForegroundBalanced,
        count: 1,
        out_path: cache_dir.join("samples.jsonl"),
        seed: 0,
        epoch: 0,
        worker: 0,
    })
    .unwrap_err();

    assert_invalid_input(error, "cache contains no cases");
}

#[test]
fn sample_cache_reports_output_parent_creation_errors() {
    let cache_dir = temp_dir("sample-cache-parent-file");
    let shape = [1, 1, 1];
    write_u16_raw(&cache_dir.join("label.u16.raw"), &[0]);
    write_cache_manifest(&cache_dir, vec![cached_case(&cache_dir, shape)]);
    let parent = cache_dir.join("not-a-directory");
    fs::write(&parent, b"file blocks directory creation").unwrap();

    let error = sample_cache(&SampleConfig {
        cache_dir: cache_dir.clone(),
        patch_size: [1, 1, 1],
        strategy: SamplingStrategy::ForegroundBalanced,
        count: 1,
        out_path: parent.join("samples.jsonl"),
        seed: 0,
        epoch: 0,
        worker: 0,
    })
    .unwrap_err();

    match &error {
        SamplerError::Io { path, .. } => assert_eq!(path, &parent),
        other => panic!("expected io error, got {other:?}"),
    }
}

#[test]
fn load_cached_cases_reads_volumes_and_derives_foreground_from_label() {
    let cache_dir = temp_dir("load-cache-success-label");
    let shape = [2, 1, 1];
    write_f32_raw(&cache_dir.join("image.f32.raw"), &[1.0, 2.0]);
    write_u16_raw(&cache_dir.join("label.u16.raw"), &[0, 5]);
    write_cache_manifest(&cache_dir, vec![cached_case(&cache_dir, shape)]);

    let cases = load_cached_cases(&cache_dir).unwrap();

    assert_eq!(cases.len(), 1);
    assert_eq!(cases[0].image.data, vec![1.0, 2.0]);
    assert_eq!(cases[0].label.data, vec![0, 5]);
    assert_eq!(cases[0].foreground_indices, vec![1]);
    assert_eq!(
        cases[0]
            .foreground_prefix
            .count_checked([0, 0, 0], [2, 1, 1])
            .unwrap(),
        1
    );
}

#[test]
fn load_cached_cases_reads_persisted_foreground_artifacts() {
    let cache_dir = temp_dir("load-cache-success-artifacts");
    let shape = [2, 1, 1];
    write_f32_raw(&cache_dir.join("image.f32.raw"), &[1.0, 2.0]);
    write_u16_raw(&cache_dir.join("label.u16.raw"), &[0, 5]);
    write_u64_raw(&cache_dir.join("foreground.u64.raw"), &[1]);
    let mut prefix_values = vec![0_u32; 12];
    prefix_values[11] = 1;
    write_u32_raw(&cache_dir.join("prefix.u32.raw"), &prefix_values);
    let mut case = cached_case(&cache_dir, shape);
    case.foreground_indices_path = Some(
        cache_dir
            .join("foreground.u64.raw")
            .to_string_lossy()
            .into_owned(),
    );
    case.foreground_prefix_path = Some(
        cache_dir
            .join("prefix.u32.raw")
            .to_string_lossy()
            .into_owned(),
    );
    case.foreground_prefix_shape = Some([3, 2, 2]);
    write_cache_manifest(&cache_dir, vec![case]);

    let cases = load_cached_cases(&cache_dir).unwrap();

    assert_eq!(cases[0].foreground_indices, vec![1]);
    assert_eq!(
        cases[0]
            .foreground_prefix
            .count_checked([1, 0, 0], [1, 1, 1])
            .unwrap(),
        1
    );
}

#[test]
fn load_cached_cases_rejects_malformed_image_raw_length() {
    let cache_dir = temp_dir("load-cache-bad-image");
    let shape = [2, 1, 1];
    fs::write(cache_dir.join("image.f32.raw"), [1_u8, 2, 3]).unwrap();
    write_u16_raw(&cache_dir.join("label.u16.raw"), &[0, 1]);
    write_cache_manifest(&cache_dir, vec![cached_case(&cache_dir, shape)]);

    let error = load_cached_cases(&cache_dir).unwrap_err();

    assert_invalid_input(error, "f32 cache file");
}

#[test]
fn load_cached_cases_rejects_wrong_image_and_label_value_counts() {
    let cache_dir = temp_dir("load-cache-bad-counts");
    let shape = [2, 1, 1];
    write_f32_raw(&cache_dir.join("image.f32.raw"), &[1.0]);
    write_u16_raw(&cache_dir.join("label.u16.raw"), &[0, 1]);
    write_cache_manifest(&cache_dir, vec![cached_case(&cache_dir, shape)]);
    let image_error = load_cached_cases(&cache_dir).unwrap_err();
    assert_invalid_input(image_error, "invalid cached image");

    write_f32_raw(&cache_dir.join("image.f32.raw"), &[1.0, 2.0]);
    write_u16_raw(&cache_dir.join("label.u16.raw"), &[1]);
    let label_error = load_cached_cases(&cache_dir).unwrap_err();
    assert_invalid_input(label_error, "invalid cached label");
}

#[test]
fn load_cached_cases_rejects_malformed_label_raw_length() {
    let cache_dir = temp_dir("load-cache-bad-label");
    let shape = [2, 1, 1];
    write_f32_raw(&cache_dir.join("image.f32.raw"), &[1.0, 2.0]);
    fs::write(cache_dir.join("label.u16.raw"), [1_u8]).unwrap();
    write_cache_manifest(&cache_dir, vec![cached_case(&cache_dir, shape)]);

    let error = load_cached_cases(&cache_dir).unwrap_err();

    assert_invalid_input(error, "u16 cache file");
}

#[test]
fn sample_cache_rejects_bad_foreground_prefix_shape() {
    let cache_dir = temp_dir("sample-cache-bad-prefix");
    let shape = [2, 1, 1];
    let mut case = cached_case(&cache_dir, shape);
    case.foreground_indices_path = Some(
        cache_dir
            .join("foreground.u64.raw")
            .to_string_lossy()
            .into_owned(),
    );
    case.foreground_prefix_path = Some(
        cache_dir
            .join("prefix.u32.raw")
            .to_string_lossy()
            .into_owned(),
    );
    case.foreground_prefix_shape = Some([2, 2, 2]);
    write_cache_manifest(&cache_dir, vec![case]);

    let error = sample_cache(&SampleConfig {
        cache_dir: cache_dir.clone(),
        patch_size: [1, 1, 1],
        strategy: SamplingStrategy::ForegroundBalanced,
        count: 1,
        out_path: cache_dir.join("samples.jsonl"),
        seed: 0,
        epoch: 0,
        worker: 0,
    })
    .unwrap_err();

    assert_invalid_input(error, "foreground prefix shape");
}

#[test]
fn sample_cache_rejects_malformed_foreground_artifacts() {
    let cache_dir = temp_dir("sample-cache-bad-artifacts");
    let shape = [2, 1, 1];
    let mut case = cached_case(&cache_dir, shape);
    case.foreground_indices_path = Some(
        cache_dir
            .join("foreground.u64.raw")
            .to_string_lossy()
            .into_owned(),
    );
    case.foreground_prefix_path = Some(
        cache_dir
            .join("prefix.u32.raw")
            .to_string_lossy()
            .into_owned(),
    );
    fs::write(cache_dir.join("foreground.u64.raw"), [1_u8]).unwrap();
    write_u32_raw(&cache_dir.join("prefix.u32.raw"), &[0; 8]);
    write_cache_manifest(&cache_dir, vec![case.clone()]);

    let index_error = sample_cache_config(&cache_dir).unwrap_err();
    assert_invalid_input(index_error, "u64 index cache file");

    write_u64_raw(&cache_dir.join("foreground.u64.raw"), &[1]);
    fs::write(cache_dir.join("prefix.u32.raw"), [1_u8]).unwrap();
    write_cache_manifest(&cache_dir, vec![case]);

    let prefix_error = sample_cache_config(&cache_dir).unwrap_err();
    assert_invalid_input(prefix_error, "u32 cache file");
}

fn loaded_case() -> LoadedCachedCase {
    let image =
        CachedImageVolume::new([4, 4, 4], 1, (0..64).map(|value| value as f32).collect()).unwrap();
    let mut label_values = vec![0_u16; 64];
    label_values[21] = 1;
    let label = Volume3D::new([4, 4, 4], label_values).unwrap();
    let foreground_prefix = ForegroundPrefix::from_label(&label).unwrap();
    LoadedCachedCase {
        metadata: cached_case(Path::new("."), [4, 4, 4]),
        image,
        label,
        foreground_indices: vec![21],
        foreground_prefix,
    }
}

fn assert_invalid_input(error: SamplerError, expected: &str) {
    match error {
        SamplerError::InvalidInput { reason } => assert!(
            reason.contains(expected),
            "expected reason containing {expected:?}, got {reason:?}"
        ),
        other => panic!("expected invalid input, got {other:?}"),
    }
}

fn sample_cache_config(
    cache_dir: &Path,
) -> std::result::Result<medkit_sampler::SampleSummary, SamplerError> {
    sample_cache(&SampleConfig {
        cache_dir: cache_dir.to_path_buf(),
        patch_size: [1, 1, 1],
        strategy: SamplingStrategy::ForegroundBalanced,
        count: 1,
        out_path: cache_dir.join("samples.jsonl"),
        seed: 0,
        epoch: 0,
        worker: 0,
    })
}

fn temp_dir(case: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "medkit-sampler-{case}-{}-{nanos}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_cache_manifest(cache_dir: &Path, cases: Vec<CachedCase>) {
    let manifest = CacheManifest {
        version: 1,
        cache_dir: cache_dir.to_string_lossy().into_owned(),
        dataset_manifest_path: "dataset_manifest.json".to_string(),
        transform_plan_hash: "plan-hash".to_string(),
        transform_plan: TransformPlan::ct_segmentation_default(),
        summary: CacheSummary {
            input_cases: cases.len(),
            cached_cases: cases.len(),
            failed_cases: 0,
            foreground_voxels: cases.iter().map(|case| case.foreground_voxels).sum(),
            bytes_written: cases.iter().map(|case| case.bytes_written).sum(),
        },
        cases,
    };
    fs::write(
        cache_dir.join("cache_manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
}

fn cached_case(cache_dir: &Path, shape: [usize; 3]) -> CachedCase {
    let geometry = VolumeGeometry::identity(shape, [1.0, 1.0, 1.0]).unwrap();
    CachedCase {
        case_id: "case".to_string(),
        cache_key: "case-key".to_string(),
        source_metadata_hash: "source".to_string(),
        transform_plan_hash: "plan".to_string(),
        image_path: "image.nii".to_string(),
        image_paths: vec!["image.nii".to_string()],
        image_channel_count: 1,
        label_path: "label.nii".to_string(),
        source_geometry: geometry,
        output_geometry: geometry,
        image_cache_path: cache_dir
            .join("image.f32.raw")
            .to_string_lossy()
            .into_owned(),
        image_cache_sha256: String::new(),
        label_cache_path: cache_dir
            .join("label.u16.raw")
            .to_string_lossy()
            .into_owned(),
        label_cache_sha256: String::new(),
        foreground_indices_path: None,
        foreground_indices_sha256: None,
        foreground_prefix_path: None,
        foreground_prefix_sha256: None,
        foreground_prefix_shape: None,
        image_chunk_cache_path: None,
        image_chunk_cache_sha256: None,
        label_chunk_cache_path: None,
        label_chunk_cache_sha256: None,
        shape,
        chunk_shape: shape,
        chunk_grid: None,
        crop_origin: [0, 0, 0],
        applied_operations: vec!["test".to_string()],
        foreground_voxels: 1,
        bytes_written: shape[0] * shape[1] * shape[2] * 6,
    }
}

fn write_f32_raw(path: &Path, values: &[f32]) {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).unwrap();
}

fn write_u16_raw(path: &Path, values: &[u16]) {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).unwrap();
}

fn write_u32_raw(path: &Path, values: &[u32]) {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).unwrap();
}

fn write_u64_raw(path: &Path, values: &[u64]) {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).unwrap();
}
