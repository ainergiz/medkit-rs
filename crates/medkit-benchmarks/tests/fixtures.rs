use std::fs;

use medkit_benchmarks::fixtures::{
    build_cached_fixture, create_cxr_manifest_scale_fixture, create_synthetic_fixture,
    cxr_scale_benchmark_report, synthetic_loaded_case, temp_fixture_root,
    write_cxr_scale_benchmark_report, CxrManifestScaleConfig, CxrScaleBenchmarkMetrics,
    SyntheticFixtureConfig, CXR_MANIFEST_SCALE_RECORD_COUNTS,
};
use medkit_cxr::{read_manifest, SplitFile};

#[test]
fn synthetic_fixture_writes_nn_unet_shaped_inputs_and_plan() {
    let mut config = SyntheticFixtureConfig::new(temp_fixture_root("synthetic-fixture"));
    config.cases = 2;
    config.shape = [8, 8, 8];
    config.cache_shape = [8, 8, 8];

    let fixture = create_synthetic_fixture(&config).unwrap();

    assert!(fixture.images_dir.join("case_0000_0000.nii").exists());
    assert!(fixture.labels_dir.join("case_0000.nii").exists());
    assert!(fixture.images_dir.join("case_0001_0000.nii").exists());
    assert!(fixture.labels_dir.join("case_0001.nii").exists());
    let plan = fs::read_to_string(fixture.plan_path).unwrap();
    assert!(plan.contains("op = \"resample\""));
    assert!(plan.contains("size = [8, 8, 8]"));
}

#[test]
fn cached_fixture_prepares_reusable_cache_manifest() {
    let mut config = SyntheticFixtureConfig::new(temp_fixture_root("cached-fixture"));
    config.cases = 2;
    config.shape = [12, 12, 12];
    config.cache_shape = [12, 12, 12];

    let fixture = build_cached_fixture(&config).unwrap();

    assert_eq!(fixture.cache_manifest.summary.input_cases, 2);
    assert_eq!(fixture.cache_manifest.summary.cached_cases, 2);
    assert!(fixture.fixture.manifest_path.exists());
    assert!(fixture.fixture.report_path.exists());
    assert!(fixture
        .fixture
        .cache_dir
        .join("cache_manifest.json")
        .exists());
    assert!(fixture.cache_manifest.cases[0]
        .applied_operations
        .contains(&"resample".to_string()));
}

#[test]
fn synthetic_loaded_case_supports_sampler_microbenchmarks() {
    let case = synthetic_loaded_case([16, 16, 16]).unwrap();

    assert_eq!(case.image.shape, [16, 16, 16]);
    assert_eq!(case.label.shape, [16, 16, 16]);
    assert!(!case.foreground_indices.is_empty());
    assert_eq!(
        case.metadata.foreground_voxels,
        case.foreground_indices.len()
    );
}

#[test]
fn cxr_manifest_scale_sizes_match_dicom_iteration_plan() {
    assert_eq!(CXR_MANIFEST_SCALE_RECORD_COUNTS, [1_000, 10_000, 100_000]);
}

#[test]
fn cxr_manifest_scale_fixture_writes_manifest_splits_and_report() {
    let mut config = CxrManifestScaleConfig::new(temp_fixture_root("cxr-scale-fixture"), 9);
    config.image_size = [384, 384];

    let fixture = create_cxr_manifest_scale_fixture(&config).unwrap();

    let records = read_manifest(&fixture.manifest_path).unwrap();
    assert_eq!(records.len(), 9);
    assert_eq!(records[0].source_format, "dicom");
    assert_eq!(records[0].width, Some(384));
    assert_eq!(records[0].height, Some(384));
    assert!(records.iter().all(|record| record.split.is_some()));
    assert_eq!(fixture.patients, 5);

    let split_file: SplitFile =
        serde_json::from_str(&fs::read_to_string(&fixture.splits_path).unwrap()).unwrap();
    let split_total = split_file.train.len() + split_file.val.len() + split_file.test.len();
    assert_eq!(split_total, 9);
    assert_eq!(split_file.split_audit.patient_overlap_count, 0);

    let report = cxr_scale_benchmark_report(
        &fixture,
        CxrScaleBenchmarkMetrics {
            scan_records: 9,
            scan_elapsed_ms: 2.0,
            preprocessed_images: 6,
            preprocessing_elapsed_ms: 3.0,
            cache_size_bytes: 12_345,
            validation_elapsed_ms: 4.0,
            python_batches: 4,
            python_samples: 8,
            python_elapsed_ms: 4.0,
        },
    );
    write_cxr_scale_benchmark_report(&fixture.report_path, &report).unwrap();

    let stored: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&fixture.report_path).unwrap()).unwrap();
    assert_eq!(stored["workload"]["records"], 9);
    assert_eq!(stored["scan"]["records_per_second"], 4500.0);
    assert_eq!(stored["preprocessing"]["images_per_second"], 2000.0);
    assert_eq!(stored["cache_size_bytes"], 12_345);
    assert_eq!(stored["validation"]["elapsed_ms"], 4.0);
    assert_eq!(stored["python_batch"]["batches_per_second"], 1000.0);
    assert_eq!(stored["python_batch"]["samples_per_second"], 2000.0);
}

#[test]
fn cxr_manifest_scale_fixture_rejects_invalid_record_count() {
    let config = CxrManifestScaleConfig::new(temp_fixture_root("cxr-scale-invalid"), 0);

    let error = create_cxr_manifest_scale_fixture(&config).unwrap_err();

    assert!(error
        .to_string()
        .contains("records must be greater than zero"));
}
