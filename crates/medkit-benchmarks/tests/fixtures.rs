use std::fs;

use medkit_benchmarks::fixtures::{
    build_cached_fixture, create_synthetic_fixture, synthetic_loaded_case, temp_fixture_root,
    SyntheticFixtureConfig,
};

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
