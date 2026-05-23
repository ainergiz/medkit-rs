use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn cxr_cli_indexes_splits_validates_and_caches_fixture() {
    let root = unique_test_dir();
    let images = root.join("images");
    fs::create_dir_all(&images).unwrap();
    write_png(&images.join("p1_i1.png"), 10);
    write_png(&images.join("p1_i2.png"), 20);
    write_png(&images.join("p2_i1.png"), 30);
    write_png(&images.join("p3_i1.png"), 40);
    fs::write(
        root.join("metadata.csv"),
        "dicom_id,subject_id,study_id,ViewPosition,Rows,Columns\np1_i1,1,10,PA,4,4\np1_i2,1,10,PA,4,4\np2_i1,2,20,AP,4,4\np3_i1,3,30,PA,4,4\n",
    )
    .unwrap();
    fs::write(
        root.join("labels.csv"),
        "subject_id,study_id,Pneumonia,No Finding\n1,10,1,0\n2,20,0,1\n3,30,-1,\n",
    )
    .unwrap();
    fs::write(
        root.join("plan.toml"),
        "name = \"cxr-test\"\n[image]\nsize = [8, 8]\n",
    )
    .unwrap();

    let manifest = root.join("manifest.jsonl");
    run_medkit(&[
        "cxr",
        "manifest",
        "--images",
        images.to_str().unwrap(),
        "--metadata",
        root.join("metadata.csv").to_str().unwrap(),
        "--labels",
        root.join("labels.csv").to_str().unwrap(),
        "--out",
        manifest.to_str().unwrap(),
    ]);
    assert_eq!(fs::read_to_string(&manifest).unwrap().lines().count(), 4);

    let splits = root.join("splits.json");
    run_medkit(&[
        "cxr",
        "split",
        manifest.to_str().unwrap(),
        "--by",
        "patient_id",
        "--train",
        "0.5",
        "--val",
        "0.25",
        "--test",
        "0.25",
        "--out",
        splits.to_str().unwrap(),
    ]);
    let split_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&splits).unwrap()).unwrap();
    assert_eq!(
        split_json["split_audit"]["patient_overlap_count"]
            .as_u64()
            .unwrap(),
        0
    );

    let validation = root.join("validation.md");
    run_medkit(&[
        "cxr",
        "validate",
        manifest.to_str().unwrap(),
        "--require-frontal",
        "--check-patient-leakage",
        "--check-duplicates",
        "--report",
        validation.to_str().unwrap(),
    ]);
    assert!(fs::read_to_string(validation)
        .unwrap()
        .contains("patient overlap count: 0"));

    let cache = root.join("cache");
    run_medkit(&[
        "cxr",
        "cache",
        manifest.to_str().unwrap(),
        "--splits",
        splits.to_str().unwrap(),
        "--plan",
        root.join("plan.toml").to_str().unwrap(),
        "--cache",
        cache.to_str().unwrap(),
        "--targets",
        "No Finding,Pneumonia",
        "--uncertain",
        "positive",
        "--missing",
        "zero",
        "--dicom-apply-rescale",
        "false",
        "--dicom-voi",
        "minmax",
    ]);
    let mut cache_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(cache.join("cache-metadata.json")).unwrap())
            .unwrap();
    assert_eq!(cache_json["cache_schema_version"].as_u64().unwrap(), 1);
    assert_eq!(cache_json["report_schema_version"].as_u64().unwrap(), 1);
    assert_eq!(cache_json["image_size"].as_u64().unwrap(), 8);
    assert_eq!(
        cache_json["label_policy"]["uncertain"].as_str().unwrap(),
        "positive"
    );
    assert_eq!(
        cache_json["label_policy"]["missing"].as_str().unwrap(),
        "zero"
    );
    assert!(!cache_json["dicom_presentation_policy"]["apply_rescale"]
        .as_bool()
        .unwrap());
    assert_eq!(
        cache_json["dicom_presentation_policy"]["voi"]
            .as_str()
            .unwrap(),
        "minmax"
    );
    assert_eq!(cache_json["split_policy"]["train"].as_f64().unwrap(), 0.5);
    assert_eq!(cache_json["split_policy"]["seed"].as_u64().unwrap(), 0);
    assert!(
        cache_json["source_manifest_checksum"]
            .as_str()
            .unwrap()
            .len()
            >= 64
    );
    assert_eq!(
        cache_json["transform_fingerprint"].as_str().unwrap().len(),
        64
    );
    assert_ne!(
        cache_json["transform_fingerprint"].as_str().unwrap(),
        cache_json["transform_plan_hash"].as_str().unwrap()
    );
    assert!(cache.join("train-images.float32.dat").exists());

    let cache_validation = root.join("cache-validation.md");
    let cache_validation_json = root.join("cache-validation.json");
    let train_shape = cache_json["splits"]["train"]["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap().to_string())
        .collect::<Vec<_>>()
        .join(",");
    run_medkit(&[
        "cxr",
        "validate-cache",
        cache.to_str().unwrap(),
        "--split",
        "train",
        "--targets",
        "No Finding,Pneumonia",
        "--image-shape",
        &train_shape,
        "--plan",
        root.join("plan.toml").to_str().unwrap(),
        "--report",
        cache_validation.to_str().unwrap(),
        "--json",
        cache_validation_json.to_str().unwrap(),
    ]);
    assert!(fs::read_to_string(&cache_validation)
        .unwrap()
        .contains("status: ok"));
    let validation_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(cache_validation_json).unwrap()).unwrap();
    assert_eq!(validation_json["status"].as_str().unwrap(), "ok");

    cache_json["report_schema_version"] = serde_json::json!(77);
    fs::write(
        cache.join("cache-metadata.json"),
        serde_json::to_string_pretty(&cache_json).unwrap(),
    )
    .unwrap();
    let warning_output = run_medkit_output(&[
        "cxr",
        "inspect-cache",
        cache.to_str().unwrap(),
        "--split",
        "train",
    ]);
    assert!(warning_output.contains("Warnings: 1"));
    assert!(warning_output.contains("CXR report schema version is 77"));
    cache_json["report_schema_version"] = serde_json::json!(1);
    fs::write(
        cache.join("cache-metadata.json"),
        serde_json::to_string_pretty(&cache_json).unwrap(),
    )
    .unwrap();

    let inspect_output = run_medkit_output(&[
        "cxr",
        "inspect-cache",
        cache.to_str().unwrap(),
        "--targets",
        "No Finding,Pneumonia",
    ]);
    assert!(inspect_output.contains("Status: ok"));
    assert!(inspect_output.contains("Checked splits:"));

    fs::write(
        root.join("different-plan.toml"),
        "name = \"different\"\n[image]\nsize = [16, 16]\n",
    )
    .unwrap();
    let mismatch = run_medkit_fail(&[
        "cxr",
        "validate-cache",
        cache.to_str().unwrap(),
        "--split",
        "train",
        "--targets",
        "Pneumonia,No Finding",
        "--image-shape",
        "99,1,8,8",
        "--plan",
        root.join("different-plan.toml").to_str().unwrap(),
    ]);
    assert!(mismatch.stdout.contains("Status: failed"));
    assert!(mismatch.stdout.contains("target-list mismatch"));
    assert!(mismatch.stdout.contains("wrong image shape"));
    assert!(mismatch.stdout.contains("stale transform fingerprint"));
    assert!(mismatch
        .stderr
        .contains("CXR cache validation failed with 3 errors"));

    let mock_script = root.join("mock_benchmark.sh");
    fs::write(
        &mock_script,
        "args=\"$*\"\nout=\"\"\nwhile [ \"$#\" -gt 0 ]; do\n  if [ \"$1\" = \"--out\" ]; then\n    shift\n    out=\"$1\"\n  fi\n  shift || break\ndone\n[ -n \"$out\" ] || exit 3\nmkdir -p \"$(dirname \"$out\")\"\nprintf '{\"status\":\"ok\",\"args\":\"%s\"}\\n' \"$args\" > \"$out\"\n",
    )
    .unwrap();
    let benchmark = root.join("benchmark.json");
    run_medkit(&[
        "cxr",
        "benchmark",
        "--python",
        "/bin/sh",
        "--script",
        mock_script.to_str().unwrap(),
        "--manifest",
        manifest.to_str().unwrap(),
        "--splits",
        splits.to_str().unwrap(),
        "--plan",
        root.join("plan.toml").to_str().unwrap(),
        "--targets",
        "Pneumonia,No Finding",
        "--uncertain",
        "ignore",
        "--baselines",
        "pytorch_raw,monai_raw,medkit_cached_mmap",
        "--batch-sizes",
        "32,64",
        "--workers",
        "1,2",
        "--device",
        "cpu",
        "--work-dir",
        root.join("work").to_str().unwrap(),
        "--report-dir",
        root.join("reports").to_str().unwrap(),
        "--run-id",
        "fixture",
        "--max-samples",
        "4",
        "--out",
        benchmark.to_str().unwrap(),
    ]);
    let benchmark_json = fs::read_to_string(benchmark).unwrap();
    assert!(benchmark_json.contains("\"status\":\"ok\""));
    assert!(benchmark_json.contains("--batch-size 32"));
    assert!(benchmark_json.contains("--workers 1"));
}

#[test]
fn cxr_cli_index_alias_and_default_validation_report_paths_work() {
    let root = unique_test_dir();
    let images = root.join("images");
    fs::create_dir_all(&images).unwrap();
    write_png(&images.join("case_a.png"), 10);
    write_png(&images.join("case_b.png"), 20);

    let manifest = root.join("manifest.jsonl");
    let index_output = run_medkit_output(&[
        "cxr",
        "index",
        "--images",
        images.to_str().unwrap(),
        "--out",
        manifest.to_str().unwrap(),
    ]);
    assert!(index_output.contains("CXR records: 2"));
    assert!(index_output.contains("Wrote manifest:"));

    let output = Command::new(env!("CARGO_BIN_EXE_medkit"))
        .current_dir(&root)
        .args(["cxr", "validate", manifest.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8(output.stdout)
        .unwrap()
        .contains("Wrote report: validation.md"));
    assert!(root.join("validation.md").exists());
}

#[test]
fn cxr_cli_reports_parse_errors_without_running_products() {
    let missing_cxr_action = run_medkit_fail(&["cxr"]);
    assert!(missing_cxr_action.stderr.contains("Usage:"));

    let cxr_help = run_medkit_fail(&["cxr", "--help"]);
    assert!(cxr_help.stderr.contains("Usage:"));

    let missing_images = run_medkit_fail(&["cxr", "manifest", "--out", "manifest.jsonl"]);
    assert!(missing_images
        .stderr
        .contains("missing --images or --dicom-index"));
    assert!(missing_images.stderr.contains("Usage:"));

    let conflicting_sources = run_medkit_fail(&[
        "cxr",
        "manifest",
        "--images",
        ".",
        "--dicom-index",
        "dicom-index.jsonl",
        "--out",
        "manifest.jsonl",
    ]);
    assert!(conflicting_sources
        .stderr
        .contains("use either --images or --dicom-index"));

    let index_missing_out = run_medkit_fail(&["cxr", "index", "--images", "."]);
    assert!(index_missing_out.stderr.contains("missing --out"));

    let index_unknown_arg = run_medkit_fail(&["cxr", "index", "--unknown"]);
    assert!(index_unknown_arg
        .stderr
        .contains("unknown argument: --unknown"));

    let index_help = run_medkit_fail(&["cxr", "index", "--help"]);
    assert!(index_help.stderr.contains("Usage:"));

    let validate_missing_manifest = run_medkit_fail(&["cxr", "validate"]);
    assert!(validate_missing_manifest.stderr.contains("Usage:"));

    let validate_unknown_arg = run_medkit_fail(&["cxr", "validate", "manifest.jsonl", "--unknown"]);
    assert!(validate_unknown_arg
        .stderr
        .contains("unknown argument: --unknown"));

    let validate_help = run_medkit_fail(&["cxr", "validate", "manifest.jsonl", "--help"]);
    assert!(validate_help.stderr.contains("Usage:"));

    let validate_product_error = run_medkit_fail(&["cxr", "validate", "missing.jsonl"]);
    assert!(!validate_product_error.stderr.trim().is_empty());

    let split_missing_manifest = run_medkit_fail(&["cxr", "split"]);
    assert!(split_missing_manifest.stderr.contains("Usage:"));

    let split_bad_train = run_medkit_fail(&[
        "cxr",
        "split",
        "manifest.jsonl",
        "--train",
        "mostly",
        "--out",
        "splits.json",
    ]);
    assert!(split_bad_train
        .stderr
        .contains("invalid float for --train: mostly"));

    let split_bad_val = run_medkit_fail(&[
        "cxr",
        "split",
        "manifest.jsonl",
        "--val",
        "some",
        "--out",
        "splits.json",
    ]);
    assert!(split_bad_val
        .stderr
        .contains("invalid float for --val: some"));

    let split_bad_test = run_medkit_fail(&[
        "cxr",
        "split",
        "manifest.jsonl",
        "--test",
        "few",
        "--out",
        "splits.json",
    ]);
    assert!(split_bad_test
        .stderr
        .contains("invalid float for --test: few"));

    let split_bad_seed = run_medkit_fail(&[
        "cxr",
        "split",
        "manifest.jsonl",
        "--seed",
        "new",
        "--out",
        "splits.json",
    ]);
    assert!(split_bad_seed
        .stderr
        .contains("invalid integer for --seed: new"));

    let split_unknown_arg = run_medkit_fail(&["cxr", "split", "manifest.jsonl", "--unknown"]);
    assert!(split_unknown_arg
        .stderr
        .contains("unknown argument: --unknown"));

    let split_help = run_medkit_fail(&["cxr", "split", "manifest.jsonl", "--help"]);
    assert!(split_help.stderr.contains("Usage:"));

    let split_missing_out = run_medkit_fail(&["cxr", "split", "manifest.jsonl"]);
    assert!(split_missing_out.stderr.contains("missing --out"));

    let cache_missing_manifest = run_medkit_fail(&["cxr", "cache"]);
    assert!(cache_missing_manifest.stderr.contains("Usage:"));

    let cache_missing_splits = run_medkit_fail(&[
        "cxr",
        "cache",
        "manifest.jsonl",
        "--plan",
        "plan.toml",
        "--cache",
        "cache",
    ]);
    assert!(cache_missing_splits.stderr.contains("missing --splits"));

    let cache_missing_plan = run_medkit_fail(&[
        "cxr",
        "cache",
        "manifest.jsonl",
        "--splits",
        "splits.json",
        "--cache",
        "cache",
    ]);
    assert!(cache_missing_plan.stderr.contains("missing --plan"));

    let cache_missing_cache = run_medkit_fail(&[
        "cxr",
        "cache",
        "manifest.jsonl",
        "--splits",
        "splits.json",
        "--plan",
        "plan.toml",
    ]);
    assert!(cache_missing_cache.stderr.contains("missing --cache"));

    let cache_unknown_arg = run_medkit_fail(&["cxr", "cache", "manifest.jsonl", "--unknown"]);
    assert!(cache_unknown_arg
        .stderr
        .contains("unknown argument: --unknown"));

    let cache_help = run_medkit_fail(&["cxr", "cache", "manifest.jsonl", "--help"]);
    assert!(cache_help.stderr.contains("Usage:"));

    let bad_split_ratios = run_medkit_fail(&[
        "cxr",
        "split",
        "manifest.jsonl",
        "--train",
        "0.8",
        "--val",
        "0.8",
        "--test",
        "0.1",
        "--out",
        "splits.json",
    ]);
    assert!(bad_split_ratios
        .stderr
        .contains("train+val+test must equal 1.0"));

    let validate_cache_missing_cache = run_medkit_fail(&["cxr", "validate-cache"]);
    assert!(validate_cache_missing_cache.stderr.contains("Usage:"));

    let unknown_arg = run_medkit_fail(&["cxr", "validate-cache", "cache", "--bogus"]);
    assert!(unknown_arg.stderr.contains("unknown argument: --bogus"));
    assert!(unknown_arg.stderr.contains("Usage:"));

    let bad_shape = run_medkit_fail(&["cxr", "validate-cache", "cache", "--image-shape", "1,8,8"]);
    assert!(bad_shape
        .stderr
        .contains("--image-shape must be formatted as n,c,h,w"));

    let ingest_missing_root = run_medkit_fail(&["cxr", "ingest"]);
    assert!(ingest_missing_root.stderr.contains("Usage:"));

    let ingest_missing_recipe = run_medkit_fail(&[
        "cxr",
        "ingest",
        "raw",
        "--labels",
        "labels.csv",
        "--cache",
        "cache",
        "--workdir",
        "work",
        "--report",
        "report.md",
    ]);
    assert!(ingest_missing_recipe.stderr.contains("missing --recipe"));

    let ingest_bad_workers = run_medkit_fail(&[
        "cxr",
        "ingest",
        "raw",
        "--recipe",
        "recipe.toml",
        "--labels",
        "labels.csv",
        "--cache",
        "cache",
        "--workdir",
        "work",
        "--report",
        "report.md",
        "--workers",
        "many",
    ]);
    assert!(ingest_bad_workers
        .stderr
        .contains("invalid integer for --workers: many"));

    let ingest_unknown_arg = run_medkit_fail(&["cxr", "ingest", "raw", "--unknown"]);
    assert!(ingest_unknown_arg
        .stderr
        .contains("unknown argument: --unknown"));

    let root = unique_test_dir();
    fs::write(
        root.join("invalid-recipe.toml"),
        "name = \"invalid\"\n[image]\nsize = [0, 4]\n[labels]\ntargets = [\"Pneumonia\"]\n",
    )
    .unwrap();
    fs::write(
        root.join("labels.csv"),
        "patient_id,study_instance_uid,Pneumonia\n",
    )
    .unwrap();
    let invalid_recipe = run_medkit_fail(&[
        "cxr",
        "ingest",
        root.join("raw").to_str().unwrap(),
        "--recipe",
        root.join("invalid-recipe.toml").to_str().unwrap(),
        "--labels",
        root.join("labels.csv").to_str().unwrap(),
        "--cache",
        root.join("cache").to_str().unwrap(),
        "--workdir",
        root.join("work").to_str().unwrap(),
        "--report",
        root.join("report.md").to_str().unwrap(),
    ]);
    assert!(invalid_recipe.stderr.contains("image.size"));
    assert!(!root.join("cache/cache-metadata.json").exists());

    let unknown_cxr_command = run_medkit_fail(&["cxr", "unknown"]);
    assert!(unknown_cxr_command
        .stderr
        .contains("unknown cxr command: unknown"));
}

#[test]
fn cxr_benchmark_bridge_covers_flags_defaults_and_errors() {
    let root = unique_test_dir();
    let script = root.join("capture_benchmark.sh");
    fs::write(&script, "printf '%s\\n' \"$*\" > \"$0.args\"\nexit 0\n").unwrap();

    let report = root.join("nested").join("benchmark.json");
    let output = run_medkit_output(&[
        "cxr",
        "benchmark",
        "--python",
        "/bin/sh",
        "--script",
        script.to_str().unwrap(),
        "--manifest",
        "manifest.jsonl",
        "--splits",
        "splits.json",
        "--plan",
        "plan.toml",
        "--targets",
        " Pneumonia , No Finding ",
        "--uncertain",
        "ignore",
        "--baselines",
        "pytorch_raw",
        "--batch-sizes",
        " , 16,32",
        "--workers",
        " , 3,4",
        "--device",
        "cpu",
        "--work-dir",
        "work",
        "--report-dir",
        "reports",
        "--run-id",
        "run-1",
        "--max-samples",
        "10",
        "--max-train",
        "7",
        "--max-val",
        "2",
        "--max-test",
        "1",
        "--image-size",
        "224",
        "--epochs",
        "2",
        "--loader-batches",
        "5",
        "--warmup-batches",
        "1",
        "--smoke",
        "--force-cache",
        "--force-rematerialize",
        "--out",
        report.to_str().unwrap(),
    ]);
    assert!(output.contains("CXR benchmark command:"));
    assert!(output.contains("--batch-size 16"));
    assert!(output.contains("--workers 3"));
    assert!(output.contains("--smoke"));
    assert!(output.contains("--force-cache"));
    assert!(output.contains("--force-rematerialize"));

    let report_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&report).unwrap()).unwrap();
    assert_eq!(report_json["status"].as_str().unwrap(), "ok");
    assert_eq!(report_json["exit_code"].as_i64().unwrap(), 0);
    assert!(report_json["command"]
        .as_str()
        .unwrap()
        .contains("--max-train 7"));

    let no_out_output = run_medkit_output(&[
        "cxr",
        "benchmark",
        "--python",
        "/bin/sh",
        "--script",
        script.to_str().unwrap(),
        "--batch-sizes",
        ",",
        "--workers",
        ",",
    ]);
    assert!(no_out_output.contains("CXR benchmark command:"));
    assert!(!no_out_output.contains("--batch-size"));

    let failing_script = root.join("fail_benchmark.sh");
    fs::write(&failing_script, "exit 9\n").unwrap();
    let failed_report = root.join("failed.json");
    let failed = run_medkit_fail(&[
        "cxr",
        "benchmark",
        "--python",
        "/bin/sh",
        "--script",
        failing_script.to_str().unwrap(),
        "--out",
        failed_report.to_str().unwrap(),
    ]);
    assert!(failed
        .stderr
        .contains("cxr benchmark harness failed with status"));
    let failed_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(failed_report).unwrap()).unwrap();
    assert_eq!(failed_json["status"].as_str().unwrap(), "failed");
    assert_eq!(failed_json["exit_code"].as_i64().unwrap(), 9);

    let file_parent = root.join("file-parent");
    fs::write(&file_parent, "not a directory").unwrap();
    let create_dir_error = run_medkit_fail(&[
        "cxr",
        "benchmark",
        "--python",
        "/bin/sh",
        "--script",
        script.to_str().unwrap(),
        "--out",
        file_parent.join("benchmark.json").to_str().unwrap(),
    ]);
    assert!(!create_dir_error.stderr.trim().is_empty());

    let io_error = run_medkit_fail(&[
        "cxr",
        "benchmark",
        "--python",
        root.join("missing-python").to_str().unwrap(),
    ]);
    assert!(!io_error.stderr.trim().is_empty());

    let bad_max_samples = run_medkit_fail(&["cxr", "benchmark", "--max-samples", "many"]);
    assert!(bad_max_samples
        .stderr
        .contains("invalid integer for --max-samples: many"));

    let bad_max_train = run_medkit_fail(&["cxr", "benchmark", "--max-train", "many"]);
    assert!(bad_max_train
        .stderr
        .contains("invalid integer for --max-train: many"));

    let bad_max_val = run_medkit_fail(&["cxr", "benchmark", "--max-val", "many"]);
    assert!(bad_max_val
        .stderr
        .contains("invalid integer for --max-val: many"));

    let bad_max_test = run_medkit_fail(&["cxr", "benchmark", "--max-test", "many"]);
    assert!(bad_max_test
        .stderr
        .contains("invalid integer for --max-test: many"));

    let bad_image_size = run_medkit_fail(&["cxr", "benchmark", "--image-size", "large"]);
    assert!(bad_image_size
        .stderr
        .contains("invalid integer for --image-size: large"));

    let bad_epochs = run_medkit_fail(&["cxr", "benchmark", "--epochs", "many"]);
    assert!(bad_epochs
        .stderr
        .contains("invalid integer for --epochs: many"));

    let bad_loader_batches = run_medkit_fail(&["cxr", "benchmark", "--loader-batches", "many"]);
    assert!(bad_loader_batches
        .stderr
        .contains("invalid integer for --loader-batches: many"));

    let bad_warmup_batches = run_medkit_fail(&["cxr", "benchmark", "--warmup-batches", "many"]);
    assert!(bad_warmup_batches
        .stderr
        .contains("invalid integer for --warmup-batches: many"));

    let missing_value = run_medkit_fail(&["cxr", "benchmark", "--manifest"]);
    assert!(missing_value
        .stderr
        .contains("missing value for --manifest"));

    let unknown_arg = run_medkit_fail(&["cxr", "benchmark", "--unknown"]);
    assert!(unknown_arg.stderr.contains("unknown argument: --unknown"));

    let help = run_medkit_fail(&["cxr", "benchmark", "--help"]);
    assert!(help.stderr.contains("Usage:"));
}

fn run_medkit(args: &[&str]) {
    run_medkit_output(args);
}

fn run_medkit_output(args: &[&str]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_medkit"))
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "medkit {:?} failed\nstdout:\n{}\nstderr:\n{}",
        args,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

struct FailedCommand {
    stdout: String,
    stderr: String,
}

fn run_medkit_fail(args: &[&str]) -> FailedCommand {
    let output = Command::new(env!("CARGO_BIN_EXE_medkit"))
        .args(args)
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "medkit {:?} unexpectedly succeeded\nstdout:\n{}\nstderr:\n{}",
        args,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.status.code(), Some(2));
    FailedCommand {
        stdout: String::from_utf8(output.stdout).unwrap(),
        stderr: String::from_utf8(output.stderr).unwrap(),
    }
}

fn unique_test_dir() -> PathBuf {
    let counter = TEST_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "medkit-cxr-cli-test-{}-{}-{}",
        std::process::id(),
        counter,
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
