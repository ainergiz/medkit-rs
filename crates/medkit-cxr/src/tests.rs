use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use crate::{
    cache::{image_size_from_plan, read_split_file, records_for_split},
    manifest::{
        is_frontal, is_image_path, parse_label_value, patient_from_filename, read_label_csv,
        write_manifest,
    },
    split::{initial_patient_counts, normalize_stratify_value},
    util::{directory_size, *},
    *,
};

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
fn relative_cache_dirs_store_portable_split_paths_and_validate() {
    let root = unique_test_dir();
    let Fixture {
        manifest,
        splits,
        plan,
        ..
    } = build_fixture(&root);
    let cache_dir = unique_relative_test_dir();

    let cache = cache_cxr(&CacheConfig {
        manifest_path: manifest,
        splits_path: splits,
        plan_path: plan.clone(),
        cache_dir: cache_dir.clone(),
    })
    .unwrap();
    let train = cache.splits.get("train").unwrap();
    assert_eq!(train.images_path, "train-images.float32.dat");
    assert_eq!(train.labels_path, "train-labels.float32.dat");
    assert_eq!(train.masks_path, "train-masks.float32.dat");
    assert_eq!(train.metadata_path, "train-metadata.jsonl");

    let validation = validate_cache_cxr(&ValidateCacheConfig {
        cache_dir: cache_dir.clone(),
        split: Some("train".to_string()),
        expected_targets: Some(cache.targets.clone()),
        expected_image_shape: None,
        plan_path: Some(plan),
        report_path: None,
        json_path: None,
    })
    .unwrap();
    assert_eq!(validation.status, "ok");

    let reader = CxrCacheReader::open(&cache_dir, "train").unwrap();
    assert!(reader.samples() > 0);
    let _ = fs::remove_dir_all(cache_dir);
}

#[test]
fn dicom_index_manifest_validates_splits_caches_and_reads_batch() {
    let root = unique_test_dir();
    let raw = root.join("raw-dicom");
    let dicom = raw.join("patient-1/image.dc");
    write_dicom_fixture(&dicom, "DICOM-P1", "1.2.826.0.1", "1.2.826.0.1.1");

    let dicom_index = root.join("dicom-index.jsonl");
    let dicom_report = root.join("dicom-report.md");
    let scan_config = medkit_dicom::DicomScanConfig {
        root: raw,
        out_path: dicom_index.clone(),
        report_path: dicom_report,
    };
    let (scan_summary, scan_records) = medkit_dicom::scan_dicom(&scan_config).unwrap();
    medkit_dicom::write_scan_outputs(
        &scan_summary,
        &scan_records,
        &scan_config.out_path,
        &scan_config.report_path,
    )
    .unwrap();

    fs::write(
        root.join("labels.csv"),
        "patient_id,study_instance_uid,Pneumonia\nDICOM-P1,1.2.826.0.1,1\n",
    )
    .unwrap();
    let manifest = root.join("manifest.jsonl");
    let index = index_cxr(&IndexConfig {
        images_root: root.clone(),
        dicom_index_path: Some(dicom_index),
        metadata_path: None,
        labels_path: Some(root.join("labels.csv")),
        reports_root: None,
        out_path: manifest.clone(),
    })
    .unwrap();
    assert_eq!(index.records, 1);
    assert!(index.dicom_index_path.is_some());

    let records = read_manifest(&manifest).unwrap();
    assert_eq!(records[0].source_format, "dicom");
    assert_eq!(records[0].patient_id, "DICOM-P1");
    assert_eq!(records[0].study_id, "1.2.826.0.1");
    assert_eq!(records[0].width, Some(2));
    assert_eq!(records[0].height, Some(2));
    assert_eq!(
        records[0].transfer_syntax_uid.as_deref(),
        Some(medkit_dicom::EXPLICIT_VR_LITTLE_ENDIAN)
    );
    assert_eq!(records[0].labels["Pneumonia"], Some(1));

    let validation = validate_cxr(&ValidateConfig {
        manifest_path: manifest.clone(),
        require_frontal: true,
        check_patient_leakage: true,
        check_duplicates: true,
        report_path: root.join("validation.md"),
    })
    .unwrap();
    assert_eq!(validation.readable_images, 1);
    assert_eq!(validation.unreadable_images, 0);

    let splits = root.join("splits.json");
    split_cxr(&SplitConfig {
        manifest_path: manifest.clone(),
        by: "patient_id".to_string(),
        train: 1.0,
        val: 0.0,
        test: 0.0,
        stratify: Vec::new(),
        out_path: splits.clone(),
        seed: 0,
    })
    .unwrap();

    let plan = root.join("plan.toml");
    fs::write(&plan, "name = \"cxr-dicom-test\"\n[image]\nsize = [4, 4]\n").unwrap();
    let cache_dir = root.join("cache");
    let cache = cache_cxr(&CacheConfig {
        manifest_path: manifest,
        splits_path: splits,
        plan_path: plan,
        cache_dir: cache_dir.clone(),
    })
    .unwrap();
    assert!(cache.failed_samples.is_empty());
    assert!(cache
        .image_size_policy
        .transform
        .contains("medkit-dicom presentation"));

    let reader = CxrCacheReader::open(cache_dir, "train").unwrap();
    let batch = reader.read_batch(0, 1).unwrap();
    assert_eq!(batch.samples, 1);
    assert_eq!(batch.image_shape, [1, 1, 4, 4]);
    assert_eq!(batch.labels, vec![1.0]);
    assert_eq!(batch.masks, vec![1.0]);
    assert!(batch
        .images
        .iter()
        .any(|value| value.is_finite() && *value != 0.0));
}

#[test]
fn recipe_driven_dicom_ingest_writes_artifacts_metadata_report_and_is_deterministic() {
    let root = unique_test_dir();
    let raw = root.join("raw-dicom");
    write_dicom_fixture_custom(
        &raw.join("p1/image.dc"),
        DicomFixtureSpec {
            patient_id: "P1",
            study_uid: "1.2.826.0.1",
            sop_uid: "1.2.826.0.1.1",
            view_position: Some("PA"),
            transfer_syntax: medkit_dicom::EXPLICIT_VR_LITTLE_ENDIAN,
            pixels: vec![0, 64, 128, 255],
            ..DicomFixtureSpec::default()
        },
    );
    write_dicom_fixture_custom(
        &raw.join("p2/image.dc"),
        DicomFixtureSpec {
            patient_id: "P2",
            study_uid: "1.2.826.0.2",
            sop_uid: "1.2.826.0.2.1",
            view_position: Some("AP"),
            transfer_syntax: medkit_dicom::EXPLICIT_VR_LITTLE_ENDIAN,
            pixels: vec![0, 64, 128, 255],
            ..DicomFixtureSpec::default()
        },
    );
    write_dicom_fixture_custom(
        &raw.join("p3/missing-view.dc"),
        DicomFixtureSpec {
            patient_id: "P3",
            study_uid: "1.2.826.0.3",
            sop_uid: "1.2.826.0.3.1",
            view_position: None,
            transfer_syntax: medkit_dicom::EXPLICIT_VR_LITTLE_ENDIAN,
            pixels: vec![9, 10, 11, 12],
            ..DicomFixtureSpec::default()
        },
    );
    write_dicom_fixture_custom(
        &raw.join("p4/unsupported.dc"),
        DicomFixtureSpec {
            patient_id: "P4",
            study_uid: "1.2.826.0.4",
            sop_uid: "1.2.826.0.4.1",
            view_position: Some("PA"),
            transfer_syntax: "1.2.840.10008.1.2.4.91",
            pixels: vec![1, 2, 3, 4],
            ..DicomFixtureSpec::default()
        },
    );
    write_dicom_fixture_custom(
        &raw.join("p5/rle.dc"),
        DicomFixtureSpec {
            patient_id: "P5",
            study_uid: "1.2.826.0.5",
            sop_uid: "1.2.826.0.5.1",
            view_position: Some("PA"),
            transfer_syntax: medkit_dicom::RLE_LOSSLESS,
            pixels: rle_single_segment_pixels(&[5, 10, 15, 20]),
            ..DicomFixtureSpec::default()
        },
    );
    fs::write(
        root.join("labels.csv"),
        "patient_id,study_instance_uid,No Finding,Pneumonia\nP1,1.2.826.0.1,0,1\nP5,1.2.826.0.5,1,0\n",
    )
    .unwrap();
    let recipe = root.join("recipe.toml");
    write_recipe(
        &recipe,
        r#"
name = "cxr-dicom-4"

[dicom]
modalities = ["CR", "DX"]
views = ["PA", "AP"]
require_single_frame = true
allow_transfer_syntaxes = ["1.2.840.10008.1.2.1", "1.2.840.10008.1.2.5"]
unsupported_transfer_syntax = "warn"

[presentation]
apply_rescale = true
voi = "auto"
invert_monochrome1 = true
output = "mono8"

[image]
size = [4, 4]
resize = "fit"
pad_value = 0
normalize = "train_split_mean_std"

[labels]
targets = ["No Finding", "Pneumonia"]
uncertain = "ignore"
missing = "ignore"

[split]
by = "patient_id"
train = 1.0
val = 0.0
test = 0.0
stratify = []
seed = 0
"#,
    );

    let config = IngestConfig {
        raw_root: raw.clone(),
        recipe_path: recipe.clone(),
        labels_path: root.join("labels.csv"),
        cache_dir: root.join("cache"),
        workdir: root.join("work"),
        report_path: root.join("ingestion-report.md"),
        dry_run: false,
        workers: 2,
    };
    let first = ingest_cxr_dicom(&config).unwrap();
    assert_eq!(first.status, "ok");
    assert_eq!(first.counts.dicom_records_scanned, 5);
    assert_eq!(first.counts.manifest_records, 3);
    assert_eq!(first.counts.unsupported_or_skipped_images, 2);
    assert_eq!(first.missing_label_counts["No Finding"], 1);
    assert_eq!(first.missing_label_counts["Pneumonia"], 1);
    assert_eq!(first.duplicate_pixel_hash_count, 1);
    assert_eq!(first.cache_validation_status, "ok");
    assert_eq!(
        first.transfer_syntax_distribution[medkit_dicom::RLE_LOSSLESS],
        1
    );
    assert!(first
        .skipped_samples
        .iter()
        .any(|issue| issue.code == "unsupported_transfer_syntax"));
    assert!(first
        .skipped_samples
        .iter()
        .any(|issue| issue.code == "filtered_view_position"));

    for path in [
        &first.paths.dicom_index,
        &first.paths.recipe_dicom_index,
        &first.paths.manifest,
        &first.paths.validation_report,
        &first.paths.splits,
        &first.paths.cache_validation_report,
        &first.paths.cache_validation_json,
        &first.paths.ingest_report,
        &first.paths.ingest_summary_json,
        &first.paths.resume_state,
    ] {
        assert!(Path::new(path).exists(), "missing artifact {path}");
    }
    let metadata = read_cache_summary(&config.cache_dir).unwrap();
    assert_eq!(metadata.recipe_hash, first.recipe_hash);
    assert_eq!(metadata.recipe_path, recipe.display().to_string());
    assert_eq!(metadata.dicom_presentation_policy.output, "mono8");
    assert_eq!(
        metadata.transfer_syntax_policy.unsupported_transfer_syntax,
        "warn"
    );
    assert_eq!(metadata.split_policy.by, "patient_id");
    assert_eq!(
        metadata.source_manifest_checksum,
        hash_file(Path::new(&first.paths.manifest)).unwrap()
    );

    let report = fs::read_to_string(&first.paths.ingest_report).unwrap();
    assert!(report.contains("Transfer Syntax Distribution"));
    assert!(report.contains("Missing Labels"));
    assert!(report.contains("skipped"));
    assert!(report.contains("cache validation status: ok"));
    assert!(report.contains("resume state"));

    let second = ingest_cxr_dicom(&config).unwrap();
    let first_split: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&first.paths.splits).unwrap()).unwrap();
    let second_split: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&second.paths.splits).unwrap()).unwrap();
    assert_eq!(
        read_cache_summary(&config.cache_dir)
            .unwrap()
            .source_manifest_checksum,
        metadata.source_manifest_checksum
    );
    assert_eq!(first_split["train"], second_split["train"]);
    assert_eq!(
        first.cache_transform_fingerprint,
        second.cache_transform_fingerprint
    );
    assert_eq!(second.cache_validation_status, "ok");

    fs::write(
        root.join("labels.csv"),
        "patient_id,study_instance_uid,No Finding,Pneumonia\nP1,1.2.826.0.1,1,0\n",
    )
    .unwrap();
    let stale_workdir = ingest_cxr_dicom(&config).unwrap_err();
    assert!(stale_workdir.to_string().contains("resume state"));
    assert!(config.cache_dir.join("cache-metadata.json").exists());

    fs::write(
        root.join("different-recipe.toml"),
        fs::read_to_string(&recipe)
            .unwrap()
            .replace("[4, 4]", "[8, 8]"),
    )
    .unwrap();
    let stale = validate_cache_cxr(&ValidateCacheConfig {
        cache_dir: config.cache_dir.clone(),
        split: Some("train".to_string()),
        expected_targets: Some(vec!["No Finding".to_string(), "Pneumonia".to_string()]),
        expected_image_shape: None,
        plan_path: Some(root.join("different-recipe.toml")),
        report_path: None,
        json_path: None,
    })
    .unwrap();
    assert_eq!(stale.status, "failed");
    assert!(stale
        .errors
        .iter()
        .any(|error| error.contains("stale transform fingerprint")));
}

#[test]
fn ingest_rejects_invalid_recipes_and_supports_dry_run_without_cache_tensors() {
    let root = unique_test_dir();
    let raw = root.join("raw-dicom");
    write_dicom_fixture(
        &raw.join("p1/image.dc"),
        "P1",
        "1.2.826.0.1",
        "1.2.826.0.1.1",
    );
    fs::write(
        root.join("labels.csv"),
        "patient_id,study_instance_uid,No Finding,Pneumonia\nP1,1.2.826.0.1,0,1\n",
    )
    .unwrap();
    let recipe = root.join("recipe.toml");
    write_recipe(
        &recipe,
        r#"
name = "dry-run"
[image]
size = [4, 4]
[labels]
targets = ["No Finding", "Pneumonia"]
[split]
train = 1.0
val = 0.0
test = 0.0
"#,
    );

    let dry = ingest_cxr_dicom(&IngestConfig {
        raw_root: raw.clone(),
        recipe_path: recipe.clone(),
        labels_path: root.join("labels.csv"),
        cache_dir: root.join("dry-cache"),
        workdir: root.join("dry-work"),
        report_path: root.join("dry-report.md"),
        dry_run: true,
        workers: 1,
    })
    .unwrap();
    assert_eq!(dry.status, "planned");
    assert!(Path::new(&dry.paths.ingest_summary_json).exists());
    assert!(fs::read_to_string(&dry.paths.ingest_report)
        .unwrap()
        .contains("dry run: true"));
    assert!(!root.join("dry-cache/cache-metadata.json").exists());
    let dry_work_blocker = root.join("dry-work-blocker");
    fs::write(&dry_work_blocker, b"file blocks dry-run summary dir").unwrap();
    assert!(ingest_cxr_dicom(&IngestConfig {
        raw_root: raw.clone(),
        recipe_path: recipe.clone(),
        labels_path: root.join("labels.csv"),
        cache_dir: root.join("dry-error-cache"),
        workdir: dry_work_blocker,
        report_path: root.join("dry-error-report.md"),
        dry_run: true,
        workers: 1,
    })
    .unwrap_err()
    .to_string()
    .contains("File exists"));

    let invalid = root.join("invalid.toml");
    write_recipe(
        &invalid,
        r#"
name = "invalid"
[image]
size = [0, 4]
[labels]
targets = ["Pneumonia"]
"#,
    );
    let error = ingest_cxr_dicom(&IngestConfig {
        raw_root: raw.clone(),
        recipe_path: invalid,
        labels_path: root.join("labels.csv"),
        cache_dir: root.join("invalid-cache"),
        workdir: root.join("invalid-work"),
        report_path: root.join("invalid-report.md"),
        dry_run: false,
        workers: 1,
    })
    .unwrap_err();
    assert!(error.to_string().contains("image.size"));
    assert!(!root.join("invalid-cache/cache-metadata.json").exists());

    let fail_recipe = root.join("fail-unsupported.toml");
    write_recipe(
        &fail_recipe,
        r#"
name = "fail-unsupported"
[dicom]
allow_transfer_syntaxes = ["1.2.840.10008.1.2.2"]
unsupported_transfer_syntax = "fail"
[image]
size = [4, 4]
[labels]
targets = ["No Finding", "Pneumonia"]
[split]
train = 1.0
val = 0.0
test = 0.0
"#,
    );
    let unsupported = ingest_cxr_dicom(&IngestConfig {
        raw_root: raw.clone(),
        recipe_path: fail_recipe,
        labels_path: root.join("labels.csv"),
        cache_dir: root.join("fail-cache"),
        workdir: root.join("fail-work"),
        report_path: root.join("fail-report.md"),
        dry_run: false,
        workers: 1,
    })
    .unwrap_err();
    assert!(unsupported
        .to_string()
        .contains("transfer syntax 1.2.840.10008.1.2.1"));
    assert!(!root.join("fail-cache/cache-metadata.json").exists());

    let empty_recipe = root.join("empty-after-filter.toml");
    write_recipe(
        &empty_recipe,
        r#"
name = "empty-after-filter"
[dicom]
allow_transfer_syntaxes = ["1.2.840.10008.1.2.2"]
unsupported_transfer_syntax = "skip"
[image]
size = [4, 4]
[labels]
targets = ["No Finding", "Pneumonia"]
[split]
train = 1.0
val = 0.0
test = 0.0
"#,
    );
    let empty = ingest_cxr_dicom(&IngestConfig {
        raw_root: raw.clone(),
        recipe_path: empty_recipe,
        labels_path: root.join("labels.csv"),
        cache_dir: root.join("empty-cache"),
        workdir: root.join("empty-work"),
        report_path: root.join("empty-report.md"),
        dry_run: false,
        workers: 1,
    })
    .unwrap_err();
    assert!(empty
        .to_string()
        .contains("filters removed all DICOM records"));
    assert!(!root.join("empty-cache/cache-metadata.json").exists());
}

#[test]
fn stratified_patient_split_balances_labels_and_metadata_without_leakage() {
    let root = unique_test_dir();
    let manifest = root.join("manifest.jsonl");
    let mut records = Vec::new();
    for index in 0..12 {
        let finding = Some(if index < 6 { 1 } else { 0 });
        let view = if index % 2 == 0 { "PA" } else { "AP" };
        records.push(test_cxr_record(
            &format!("sample-{index}"),
            &format!("patient-{index}"),
            finding,
            Some(view),
        ));
    }
    write_manifest(&manifest, &records).unwrap();

    let splits_path = root.join("stratified-splits.json");
    let summary = split_cxr(&SplitConfig {
        manifest_path: manifest.clone(),
        by: "patient_id".to_string(),
        train: 0.5,
        val: 0.25,
        test: 0.25,
        stratify: vec!["Finding".to_string(), "view_position".to_string()],
        out_path: splits_path.clone(),
        seed: 11,
    })
    .unwrap();
    assert_eq!(summary.counts["train"], 6);
    assert_eq!(summary.counts["val"], 3);
    assert_eq!(summary.counts["test"], 3);
    assert_eq!(summary.patient_counts["train"], 6);
    assert_eq!(summary.patient_overlap_count, 0);

    let split_file = read_split_file(&splits_path).unwrap();
    let edited = read_manifest(&manifest).unwrap();
    assert_patient_sets_do_not_overlap(&edited);
    assert_eq!(split_file.train.len(), 6);
    assert_eq!(split_file.val.len(), 3);
    assert_eq!(split_file.test.len(), 3);

    let positive_counts = positive_counts_by_split(&edited, "Finding");
    assert_eq!(positive_counts["train"], 3);
    assert_eq!(positive_counts["val"], 2);
    assert_eq!(positive_counts["test"], 1);
    for split in ["train", "val", "test"] {
        let view_counts =
            field_counts_by_split(&edited, split, |record| record.view_position.as_deref());
        assert!(view_counts.contains_key("pa"));
        assert!(view_counts.contains_key("ap"));
    }

    let second_path = root.join("stratified-splits-again.json");
    let second = split_cxr(&SplitConfig {
        manifest_path: manifest.clone(),
        by: "patient".to_string(),
        train: 0.5,
        val: 0.25,
        test: 0.25,
        stratify: vec!["Finding".to_string(), "ViewPosition".to_string()],
        out_path: second_path.clone(),
        seed: 11,
    })
    .unwrap();
    assert_eq!(second.counts, summary.counts);
    assert_eq!(
        read_split_file(&second_path).unwrap().train,
        read_split_file(&splits_path).unwrap().train
    );

    let unknown = split_cxr(&SplitConfig {
        manifest_path: manifest,
        by: "patient_id".to_string(),
        train: 0.5,
        val: 0.25,
        test: 0.25,
        stratify: vec!["DefinitelyMissing".to_string()],
        out_path: root.join("bad-stratified-splits.json"),
        seed: 11,
    })
    .unwrap_err();
    assert!(unknown.to_string().contains("unknown stratify target"));
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
        dicom_index_path: None,
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

    let mut summary_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(cache_dir.join("cache-metadata.json")).unwrap())
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
        dicom_index_path: None,
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
fn cache_build_rejects_failed_samples() {
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
        }
    }
    write_manifest(&manifest, &records).unwrap();

    let error = cache_cxr(&CacheConfig {
        manifest_path: manifest.clone(),
        splits_path: splits,
        plan_path: plan,
        cache_dir: cache_dir.clone(),
    })
    .unwrap_err();
    assert!(error
        .to_string()
        .contains(&format!("failed to preprocess sample {failing_id}")));
    assert!(!cache_dir.join("cache-metadata.json").exists());
}

#[test]
fn cache_build_preserves_non_binary_labels() {
    let root = unique_test_dir();
    let Fixture {
        manifest,
        splits,
        plan,
        cache_dir,
        ..
    } = build_fixture(&root);
    let split_file = read_split_file(&splits).unwrap();
    let custom_id = split_file
        .train
        .first()
        .expect("fixture has train sample")
        .clone();
    let mut records = read_manifest(&manifest).unwrap();
    for record in &mut records {
        if record.sample_id == custom_id {
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
    assert!(cache.failed_samples.is_empty());
    assert!(cache.targets.contains(&"Custom".to_string()));

    let target_index = cache
        .targets
        .iter()
        .position(|target| target == "Custom")
        .unwrap();
    let reader = CxrCacheReader::open(&cache_dir, "train").unwrap();
    let row = reader
        .records_for_range(0, reader.samples())
        .unwrap()
        .into_iter()
        .position(|record| record.sample_id == custom_id)
        .unwrap();
    let mut images = vec![1.0; reader.samples() * 8 * 8];
    let mut labels = vec![0.0; reader.samples() * reader.targets().len()];
    let mut masks = vec![0.0; reader.samples() * reader.targets().len()];
    reader
        .fill_batch(0, reader.samples(), &mut images, &mut labels, &mut masks)
        .unwrap();
    assert!(images[row * 64..row * 64 + 64]
        .iter()
        .any(|value| *value != 0.0));
    assert_eq!(labels[row * reader.targets().len() + target_index], 2.0);
    assert_eq!(masks[row * reader.targets().len() + target_index], 1.0);
}

#[test]
fn cache_build_uses_configured_label_policy_for_uncertain_and_missing_values() {
    let root = unique_test_dir();
    let Fixture {
        manifest,
        plan,
        cache_dir,
        ..
    } = build_fixture(&root);
    let splits = root.join("all-train-splits.json");
    split_cxr(&SplitConfig {
        manifest_path: manifest.clone(),
        by: "patient_id".to_string(),
        train: 1.0,
        val: 0.0,
        test: 0.0,
        stratify: Vec::new(),
        out_path: splits.clone(),
        seed: 0,
    })
    .unwrap();

    let policy = LabelPolicy {
        uncertain: "positive".to_string(),
        missing: "zero".to_string(),
        loss_mask: "uncertain=positive missing=zero".to_string(),
        ..LabelPolicy::default()
    };
    let cache = cache_cxr_with_options(
        &CacheConfig {
            manifest_path: manifest.clone(),
            splits_path: splits.clone(),
            plan_path: plan.clone(),
            cache_dir: cache_dir.clone(),
        },
        &CxrCacheOptions {
            label_policy: policy.clone(),
            ..CxrCacheOptions::default()
        },
    )
    .unwrap();
    assert_eq!(cache.label_policy, policy);

    let reader = CxrCacheReader::open(&cache_dir, "train").unwrap();
    let batch = reader.read_batch(0, reader.samples()).unwrap();
    let row = batch
        .records
        .iter()
        .position(|record| matches!(record.labels.get("Pneumonia").copied().flatten(), Some(-1)))
        .unwrap();
    let no_finding = reader
        .targets()
        .iter()
        .position(|target| target == "No Finding")
        .unwrap();
    let pneumonia = reader
        .targets()
        .iter()
        .position(|target| target == "Pneumonia")
        .unwrap();
    let row_offset = row * reader.targets().len();
    assert_eq!(batch.labels[row_offset + pneumonia], 1.0);
    assert_eq!(batch.masks[row_offset + pneumonia], 1.0);
    assert_eq!(batch.labels[row_offset + no_finding], 0.0);
    assert_eq!(batch.masks[row_offset + no_finding], 1.0);

    let fail_error = cache_cxr_with_options(
        &CacheConfig {
            manifest_path: manifest,
            splits_path: splits,
            plan_path: plan,
            cache_dir: root.join("cache-fail-policy"),
        },
        &CxrCacheOptions {
            label_policy: LabelPolicy {
                uncertain: "fail".to_string(),
                missing: "ignore".to_string(),
                loss_mask: "uncertain=fail missing=ignore".to_string(),
                ..LabelPolicy::default()
            },
            ..CxrCacheOptions::default()
        },
    )
    .unwrap_err();
    assert!(fail_error.to_string().contains("uncertain label"));
}

#[test]
fn malformed_metadata_and_cache_io_errors_surface() {
    let root = unique_test_dir();
    fs::create_dir_all(root.join("images")).unwrap();
    write_png(&root.join("images/img.png"), 10);
    fs::write(root.join("metadata.csv"), "subject_id,study_id\n1,1\n").unwrap();
    let missing_dicom = index_cxr(&IndexConfig {
        images_root: root.join("images"),
        dicom_index_path: None,
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
            series_instance_uid: None,
            sop_instance_uid: None,
            transfer_syntax_uid: None,
            pixel_hash: None,
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
    assert!(directory_size(&root).unwrap() > 0);

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
    summary.splits.get_mut("train").unwrap().metadata_path = "missing-metadata.jsonl".to_string();
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

    let csv_error = CxrError::from(csv::Error::from(std::io::Error::other("csv-display")));
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
        dicom_index_path: None,
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

fn test_cxr_record(
    sample_id: &str,
    patient_id: &str,
    finding: Option<i8>,
    view_position: Option<&str>,
) -> CxrRecord {
    CxrRecord {
        sample_id: sample_id.to_string(),
        patient_id: patient_id.to_string(),
        study_id: format!("study-{sample_id}"),
        image_id: format!("image-{sample_id}"),
        image_path: format!("{sample_id}.png"),
        source_format: "png".to_string(),
        modality: Some("DX".to_string()),
        view_position: view_position.map(ToOwned::to_owned),
        laterality: None,
        width: Some(4),
        height: Some(4),
        photometric_interpretation: Some("MONOCHROME2".to_string()),
        series_instance_uid: None,
        sop_instance_uid: None,
        transfer_syntax_uid: None,
        pixel_hash: None,
        labels: BTreeMap::from([("Finding".to_string(), finding)]),
        label_source: Some("test".to_string()),
        report_path: None,
        split: None,
        sha256: None,
    }
}

fn assert_patient_sets_do_not_overlap(records: &[CxrRecord]) {
    let mut by_split: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for record in records {
        by_split
            .entry(record.split.clone().unwrap())
            .or_default()
            .insert(record.patient_id.clone());
    }
    assert_eq!(overlap_count(&by_split), 0);
}

fn positive_counts_by_split(records: &[CxrRecord], target: &str) -> BTreeMap<String, usize> {
    let mut counts = initial_patient_counts();
    for record in records {
        if matches!(record.labels.get(target).copied().flatten(), Some(1)) {
            *counts.entry(record.split.clone().unwrap()).or_insert(0) += 1;
        }
    }
    counts
}

fn field_counts_by_split(
    records: &[CxrRecord],
    split: &str,
    field: impl Fn(&CxrRecord) -> Option<&str>,
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for record in records
        .iter()
        .filter(|record| record.split.as_deref() == Some(split))
    {
        if let Some(value) = field(record).and_then(normalize_stratify_value) {
            *counts.entry(value).or_insert(0) += 1;
        }
    }
    counts
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

fn unique_relative_test_dir() -> PathBuf {
    static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let sequence = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = PathBuf::from("../../target").join(format!(
        "medkit-cxr-relative-cache-{}-{}-{}",
        std::process::id(),
        sequence,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = fs::remove_dir_all(&path);
    path
}

fn write_png(path: &Path, value: u8) {
    let image = image::GrayImage::from_pixel(4, 4, image::Luma([value]));
    image.save(path).unwrap();
}

#[derive(Clone)]
struct DicomFixtureSpec<'a> {
    patient_id: &'a str,
    study_uid: &'a str,
    series_uid: &'a str,
    sop_uid: &'a str,
    modality: &'a str,
    view_position: Option<&'a str>,
    transfer_syntax: &'a str,
    pixels: Vec<u8>,
}

impl Default for DicomFixtureSpec<'_> {
    fn default() -> Self {
        Self {
            patient_id: "patient-1",
            study_uid: "1.2.826.0.1",
            series_uid: "1.2.826.0.1.99",
            sop_uid: "1.2.826.0.1.1",
            modality: "DX",
            view_position: Some("PA"),
            transfer_syntax: medkit_dicom::EXPLICIT_VR_LITTLE_ENDIAN,
            pixels: vec![0, 64, 128, 255],
        }
    }
}

fn write_dicom_fixture(path: &Path, patient_id: &str, study_uid: &str, sop_uid: &str) {
    write_dicom_fixture_custom(
        path,
        DicomFixtureSpec {
            patient_id,
            study_uid,
            sop_uid,
            ..DicomFixtureSpec::default()
        },
    );
}

fn write_dicom_fixture_custom(path: &Path, spec: DicomFixtureSpec<'_>) {
    let mut bytes = vec![0u8; 128];
    bytes.extend_from_slice(b"DICM");
    push_text(&mut bytes, (0x0002, 0x0010), "UI", spec.transfer_syntax);
    push_text(&mut bytes, (0x0010, 0x0020), "LO", spec.patient_id);
    push_text(&mut bytes, (0x0020, 0x000D), "UI", spec.study_uid);
    push_text(&mut bytes, (0x0020, 0x000E), "UI", spec.series_uid);
    push_text(&mut bytes, (0x0008, 0x0018), "UI", spec.sop_uid);
    push_text(&mut bytes, (0x0008, 0x0060), "CS", spec.modality);
    if let Some(view_position) = spec.view_position {
        push_text(&mut bytes, (0x0018, 0x5101), "CS", view_position);
    }
    push_text(&mut bytes, (0x0028, 0x0030), "DS", "0.5\\0.5");
    push_text(&mut bytes, (0x0028, 0x0004), "CS", "MONOCHROME2");
    push_u16(&mut bytes, (0x0028, 0x0002), 1);
    push_u16(&mut bytes, (0x0028, 0x0010), 2);
    push_u16(&mut bytes, (0x0028, 0x0011), 2);
    push_u16(&mut bytes, (0x0028, 0x0100), 8);
    push_u16(&mut bytes, (0x0028, 0x0101), 8);
    push_u16(&mut bytes, (0x0028, 0x0102), 7);
    push_u16(&mut bytes, (0x0028, 0x0103), 0);
    push_element(&mut bytes, (0x7FE0, 0x0010), "OB", spec.pixels);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
}

fn write_recipe(path: &Path, text: &str) {
    fs::write(path, text.trim_start()).unwrap();
}

fn rle_single_segment_pixels(raw: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0u8; 64];
    bytes[0..4].copy_from_slice(&1u32.to_le_bytes());
    bytes[4..8].copy_from_slice(&64u32.to_le_bytes());
    bytes.push((raw.len() as u8).saturating_sub(1));
    bytes.extend_from_slice(raw);
    bytes
}

fn push_text(out: &mut Vec<u8>, tag: (u16, u16), vr: &str, value: &str) {
    push_element(out, tag, vr, value.as_bytes().to_vec());
}

fn push_u16(out: &mut Vec<u8>, tag: (u16, u16), value: u16) {
    push_element(out, tag, "US", value.to_le_bytes().to_vec());
}

fn push_element(out: &mut Vec<u8>, tag: (u16, u16), vr: &str, mut value: Vec<u8>) {
    if value.len() % 2 == 1 {
        value.push(if vr == "UI" { 0 } else { b' ' });
    }
    out.extend_from_slice(&tag.0.to_le_bytes());
    out.extend_from_slice(&tag.1.to_le_bytes());
    out.extend_from_slice(vr.as_bytes());
    if matches!(vr, "OB" | "OW" | "SQ" | "UN" | "UT") {
        out.extend_from_slice(&[0, 0]);
        out.extend_from_slice(&(value.len() as u32).to_le_bytes());
    } else {
        out.extend_from_slice(&(value.len() as u16).to_le_bytes());
    }
    out.extend_from_slice(&value);
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
                series_instance_uid: None,
                sop_instance_uid: None,
                transfer_syntax_uid: None,
                pixel_hash: None,
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
        recipe_hash: String::new(),
        recipe_path: String::new(),
        source_manifest_checksum: "manifest".to_string(),
        split_names: vec!["train".to_string()],
        image_size_policy: ImageSizePolicy {
            channels: 1,
            height: image_size,
            width: image_size,
            dtype: "float32".to_string(),
            transform: "synthetic".to_string(),
        },
        dicom_presentation_policy: DicomPresentationPolicy::default(),
        transfer_syntax_policy: TransferSyntaxPolicy::default(),
        split_policy: SplitPolicyMetadata::default(),
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
