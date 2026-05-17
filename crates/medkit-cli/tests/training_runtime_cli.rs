use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

const HEADER_LEN: usize = 348;

#[test]
fn prepare_sample_and_bench_workflow_runs_end_to_end() {
    let root = temp_case_dir("runtime");
    let images = root.join("imagesTr");
    let labels = root.join("labelsTr");
    fs::create_dir_all(&images).unwrap();
    fs::create_dir_all(&labels).unwrap();

    write_runtime_case(&images, &labels, "case_a");
    write_runtime_case(&images, &labels, "case_b");

    let manifest_path = root.join("manifest.json");
    let report_path = root.join("report.txt");
    run_ok(Command::new(env!("CARGO_BIN_EXE_medkit")).args([
        "dataset",
        "validate",
        root.to_str().unwrap(),
        "--out",
        manifest_path.to_str().unwrap(),
        "--report",
        report_path.to_str().unwrap(),
    ]));

    let plan_path = root.join("ct-segmentation.toml");
    fs::write(&plan_path, transform_plan_toml()).unwrap();
    let cache_dir = root.join(".medkit").join("cache");
    let prepare = run_ok(Command::new(env!("CARGO_BIN_EXE_medkit")).args([
        "prepare",
        root.to_str().unwrap(),
        "--manifest",
        manifest_path.to_str().unwrap(),
        "--plan",
        plan_path.to_str().unwrap(),
        "--cache",
        cache_dir.to_str().unwrap(),
        "--chunk",
        "8,8,8",
    ]));
    assert!(prepare.contains("Cached cases: 2"));
    assert!(prepare.contains("Transform plan hash:"));

    let cache_manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(cache_dir.join("cache_manifest.json")).unwrap())
            .unwrap();
    assert_eq!(cache_manifest["summary"]["cached_cases"], 2);
    assert_eq!(
        cache_manifest["transform_plan"]["operations"][0]["op"],
        "resample"
    );
    assert_eq!(
        cache_manifest["cases"][0]["source_geometry"]["spacing"],
        serde_json::json!([2.0, 2.0, 2.0])
    );
    assert_eq!(
        cache_manifest["cases"][0]["output_geometry"]["spacing"],
        serde_json::json!([1.0, 1.0, 1.0])
    );
    assert_eq!(
        cache_manifest["cases"][0]["shape"],
        serde_json::json!([16, 16, 16])
    );
    assert_eq!(
        cache_manifest["cases"][0]["chunk_shape"],
        serde_json::json!([8, 8, 8])
    );
    assert_eq!(
        cache_manifest["cases"][0]["chunk_grid"],
        serde_json::json!([2, 2, 2])
    );
    for key in [
        "foreground_indices_path",
        "foreground_prefix_path",
        "image_chunk_cache_path",
        "label_chunk_cache_path",
    ] {
        let path = cache_manifest["cases"][0][key].as_str().unwrap();
        assert!(Path::new(path).exists(), "{key} should exist at {path}");
    }

    let inspect = run_ok(Command::new(env!("CARGO_BIN_EXE_medkit")).args([
        "cache",
        "inspect",
        cache_dir.to_str().unwrap(),
    ]));
    assert!(inspect.contains("Status: ok"));
    assert!(inspect.contains("Cases: 2"));
    assert!(inspect.contains("Chunked cases: 2"));

    let validate = run_ok(Command::new(env!("CARGO_BIN_EXE_medkit")).args([
        "cache",
        "validate",
        cache_dir.to_str().unwrap(),
    ]));
    assert!(validate.contains("Errors: 0"));

    let patches_path = root.join("patches.jsonl");
    let sample = run_ok(Command::new(env!("CARGO_BIN_EXE_medkit")).args([
        "sample",
        cache_dir.to_str().unwrap(),
        "--patch",
        "8,8,8",
        "--strategy",
        "foreground-balanced",
        "--count",
        "10",
        "--seed",
        "123",
        "--epoch",
        "4",
        "--worker",
        "2",
        "--out",
        patches_path.to_str().unwrap(),
    ]));
    assert!(sample.contains("Samples: 10"));
    assert!(sample.contains("Wrote samples:"));

    let patch_lines = fs::read_to_string(&patches_path).unwrap();
    let patches = patch_lines.lines().collect::<Vec<_>>();
    assert_eq!(patches.len(), 10);
    let first_patch: serde_json::Value = serde_json::from_str(patches[0]).unwrap();
    assert_eq!(first_patch["patch_size"], serde_json::json!([8, 8, 8]));
    assert_eq!(first_patch["epoch"], serde_json::json!(4));
    assert_eq!(first_patch["worker"], serde_json::json!(2));
    assert!(patches
        .iter()
        .any(|line| line.contains("\"has_foreground\":true")));

    let bench = run_ok(Command::new(env!("CARGO_BIN_EXE_medkit")).args([
        "bench",
        cache_dir.to_str().unwrap(),
        "--patch",
        "8,8,8",
        "--workers",
        "2",
        "--samples",
        "10",
    ]));
    assert!(bench.contains("Cold:"));
    assert!(bench.contains("Warm:"));
    assert!(bench.contains("Samples: 10"));
    assert!(bench.contains("Python/MONAI baseline: planned"));

    let plan_bench = run_ok(Command::new(env!("CARGO_BIN_EXE_medkit")).args([
        "bench-plan",
        cache_dir.to_str().unwrap(),
        "--patches",
        patches_path.to_str().unwrap(),
        "--workers",
        "2",
        "--samples",
        "10",
    ]));
    assert!(plan_bench.contains("Plan cold:"));
    assert!(plan_bench.contains("Plan warm:"));
    assert!(plan_bench.contains("Records: 10"));
}

#[test]
fn runtime_commands_report_parse_and_legacy_alias_errors() {
    let no_command = run_fail([]);
    assert!(no_command.stderr.contains("Usage:"));

    let unknown_command = run_fail(["unknown"]);
    assert!(unknown_command.stderr.contains("Usage:"));

    let prepare_missing_manifest = run_fail(["prepare", "."]);
    assert!(prepare_missing_manifest
        .stderr
        .contains("missing --manifest"));
    assert!(prepare_missing_manifest.stderr.contains("Usage:"));

    let prepare_no_args = run_fail(["prepare"]);
    assert!(prepare_no_args.stderr.contains("Usage:"));

    let prepare_bad_chunk = run_fail([
        "prepare",
        ".",
        "--manifest",
        "manifest.json",
        "--plan",
        "plan.toml",
        "--cache",
        "cache",
        "--chunk",
        "8,8",
    ]);
    assert!(prepare_bad_chunk
        .stderr
        .contains("patch must be formatted as x,y,z"));

    let prepare_unknown_arg = run_fail(["prepare", ".", "--unknown"]);
    assert!(prepare_unknown_arg
        .stderr
        .contains("unknown argument: --unknown"));

    let prepare_help = run_fail(["prepare", ".", "--help"]);
    assert!(prepare_help.stderr.contains("Usage:"));

    let prepare_missing_flag_value = run_fail(["prepare", ".", "--manifest"]);
    assert!(prepare_missing_flag_value
        .stderr
        .contains("missing value for --manifest"));

    let prepare_cache_error = run_fail([
        "prepare",
        ".",
        "--manifest",
        "missing-manifest.json",
        "--plan",
        "missing-plan.toml",
        "--cache",
        "cache",
    ]);
    assert!(!prepare_cache_error.stderr.trim().is_empty());

    let sample_no_args = run_fail(["sample"]);
    assert!(sample_no_args.stderr.contains("Usage:"));

    let sample_missing_patch = run_fail(["sample", "cache", "--count", "1", "--out", "patches"]);
    assert!(sample_missing_patch.stderr.contains("missing --patch"));

    let sample_missing_count = run_fail([
        "sample",
        "cache",
        "--patch",
        "8,8,8",
        "--out",
        "patches.jsonl",
    ]);
    assert!(sample_missing_count.stderr.contains("missing --count"));

    let sample_missing_out = run_fail([
        "sample",
        "cache",
        "--patch",
        "8,8,8",
        "--strategy",
        "foreground_balanced",
        "--count",
        "1",
    ]);
    assert!(sample_missing_out.stderr.contains("missing --out"));

    let sample_unknown_strategy = run_fail([
        "sample",
        "cache",
        "--patch",
        "8,8,8",
        "--strategy",
        "random",
        "--count",
        "1",
        "--out",
        "patches.jsonl",
    ]);
    assert!(sample_unknown_strategy
        .stderr
        .contains("unsupported sampling strategy: random"));

    let sample_unknown_arg = run_fail(["sample", "cache", "--unknown"]);
    assert!(sample_unknown_arg
        .stderr
        .contains("unknown argument: --unknown"));

    let sample_help = run_fail(["sample", "cache", "--help"]);
    assert!(sample_help.stderr.contains("Usage:"));

    let sample_bad_count = run_fail([
        "sample",
        "cache",
        "--patch",
        "8,8,8",
        "--count",
        "many",
        "--out",
        "patches.jsonl",
    ]);
    assert!(sample_bad_count
        .stderr
        .contains("invalid integer for --count: many"));

    let sample_bad_seed = run_fail([
        "sample",
        "cache",
        "--patch",
        "8,8,8",
        "--count",
        "1",
        "--seed",
        "many",
        "--out",
        "patches.jsonl",
    ]);
    assert!(sample_bad_seed
        .stderr
        .contains("invalid integer for --seed: many"));

    let sample_cache_error = run_fail([
        "sample",
        "missing-cache",
        "--patch",
        "8,8,8",
        "--count",
        "1",
        "--out",
        "patches.jsonl",
    ]);
    assert!(!sample_cache_error.stderr.trim().is_empty());

    let cache_no_args = run_fail(["cache"]);
    assert!(cache_no_args.stderr.contains("Usage:"));

    let cache_unknown = run_fail(["cache", "unknown", "cache-dir"]);
    assert!(cache_unknown
        .stderr
        .contains("unknown cache command: unknown"));

    let cache_inspect_missing = run_fail(["cache", "inspect", "missing-cache"]);
    assert!(!cache_inspect_missing.stderr.trim().is_empty());

    let bench_no_args = run_fail(["bench"]);
    assert!(bench_no_args.stderr.contains("Usage:"));

    let bench_missing_patch = run_fail(["bench", "cache"]);
    assert!(bench_missing_patch.stderr.contains("missing --patch"));

    let bench_bad_workers = run_fail(["bench", "cache", "--patch", "8,8,8", "--workers", "two"]);
    assert!(bench_bad_workers
        .stderr
        .contains("invalid integer for --workers: two"));

    let bench_bad_samples = run_fail(["bench", "cache", "--patch", "8,8,8", "--samples", "many"]);
    assert!(bench_bad_samples
        .stderr
        .contains("invalid integer for --samples: many"));

    let bench_unknown_arg = run_fail(["bench", "cache", "--unknown"]);
    assert!(bench_unknown_arg
        .stderr
        .contains("unknown argument: --unknown"));

    let bench_help = run_fail(["bench", "cache", "--help"]);
    assert!(bench_help.stderr.contains("Usage:"));

    let bench_cache_error = run_fail(["bench", "missing-cache", "--patch", "8,8,8"]);
    assert!(!bench_cache_error.stderr.trim().is_empty());

    let bench_plan_no_args = run_fail(["bench-plan"]);
    assert!(bench_plan_no_args.stderr.contains("Usage:"));

    let bench_plan_missing_patches = run_fail(["bench-plan", "cache"]);
    assert!(bench_plan_missing_patches
        .stderr
        .contains("missing --patches"));

    let bench_plan_bad_workers = run_fail([
        "bench-plan",
        "cache",
        "--patches",
        "patches.jsonl",
        "--workers",
        "two",
    ]);
    assert!(bench_plan_bad_workers
        .stderr
        .contains("invalid integer for --workers: two"));

    let bench_plan_bad_samples = run_fail([
        "bench-plan",
        "cache",
        "--patches",
        "patches.jsonl",
        "--samples",
        "many",
    ]);
    assert!(bench_plan_bad_samples
        .stderr
        .contains("invalid integer for --samples: many"));

    let bench_plan_unknown_arg = run_fail(["bench-plan", "cache", "--unknown"]);
    assert!(bench_plan_unknown_arg
        .stderr
        .contains("unknown argument: --unknown"));

    let bench_plan_help = run_fail(["bench-plan", "cache", "--help"]);
    assert!(bench_plan_help.stderr.contains("Usage:"));

    let bench_plan_error = run_fail(["bench-plan", "missing-cache", "--patches", "missing.jsonl"]);
    assert!(!bench_plan_error.stderr.trim().is_empty());
}

fn run_ok(command: &mut Command) -> String {
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

struct FailedCommand {
    stderr: String,
}

fn run_fail<const N: usize>(args: [&str; N]) -> FailedCommand {
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
        stderr: String::from_utf8(output.stderr).unwrap(),
    }
}

fn temp_case_dir(case: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "medkit-runtime-{case}-{}-{nanos}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_runtime_case(images: &Path, labels: &Path, case_id: &str) {
    let shape = [16_usize, 16, 16];
    let image = (0..shape[2])
        .flat_map(|z| {
            (0..shape[1]).flat_map(move |y| {
                (0..shape[0]).map(move |x| (x as i16 * 3 + y as i16 * 2 + z as i16) - 100)
            })
        })
        .collect::<Vec<_>>();
    let label = (0..shape[2])
        .flat_map(|z| {
            (0..shape[1]).flat_map(move |y| {
                (0..shape[0]).map(move |x| {
                    u8::from(
                        (5..=10).contains(&x) && (5..=10).contains(&y) && (5..=10).contains(&z),
                    )
                })
            })
        })
        .collect::<Vec<_>>();
    write_i16_nifti(&images.join(format!("{case_id}.nii")), shape, &image);
    write_u8_nifti(&labels.join(format!("{case_id}.nii")), shape, &label);
}

fn transform_plan_toml() -> &'static str {
    r#"
name = "ct-segmentation-test"
image_interpolation = "linear"
label_interpolation = "nearest"

[[operations]]
op = "resample"
spacing = [1.0, 1.0, 1.0]

[[operations]]
op = "ct_window"
min = -1000.0
max = 1000.0

[[operations]]
op = "min_max_normalize"

[[operations]]
op = "crop_foreground"
margin = 2

[[operations]]
op = "pad_crop"
size = [16, 16, 16]
"#
}

fn write_i16_nifti(path: &Path, shape: [usize; 3], values: &[i16]) {
    let mut bytes = header(shape, 4, 16, [2.0, 2.0, 2.0]);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).unwrap();
}

fn write_u8_nifti(path: &Path, shape: [usize; 3], values: &[u8]) {
    let mut bytes = header(shape, 2, 8, [2.0, 2.0, 2.0]);
    bytes.extend_from_slice(values);
    fs::write(path, bytes).unwrap();
}

fn header(shape: [usize; 3], datatype: i16, bitpix: i16, spacing: [f32; 3]) -> Vec<u8> {
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
