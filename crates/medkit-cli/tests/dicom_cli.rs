use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn dicom_cli_scans_inspects_explains_and_views_fixture() {
    let root = temp_case_dir("dicom-cli");
    let dicoms = root.join("dicoms");
    let image = dicoms.join("patient-1/image.dc");
    write_dicom_fixture(&image);

    let index = root.join("out/index.jsonl");
    let report = root.join("out/report.md");
    let scan = run_medkit([
        "dicom",
        "scan",
        dicoms.to_str().unwrap(),
        "--out",
        index.to_str().unwrap(),
        "--report",
        report.to_str().unwrap(),
    ]);
    assert!(scan.stdout.contains("Records: 1"));
    assert!(scan.stdout.contains("Errors: 0"));
    assert!(scan.stdout.contains("Wrote inventory:"));

    let record: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&index).unwrap()).unwrap();
    assert_eq!(record["patient_id"], "patient-1");
    assert_eq!(record["rows"], 2);
    assert_eq!(record["columns"], 2);
    assert!(fs::read_to_string(&report)
        .unwrap()
        .contains("DICOM QA Report"));

    let inspect = run_medkit(["dicom", "inspect", image.to_str().unwrap()]);
    let inspect_json: serde_json::Value = serde_json::from_str(&inspect.stdout).unwrap();
    assert_eq!(inspect_json["record"]["modality"], "DX");
    assert_eq!(inspect_json["elements"]["(0028,0010)"], "2");

    let pixels = run_medkit(["dicom", "pixels", "--explain", image.to_str().unwrap()]);
    let pixels_json: serde_json::Value = serde_json::from_str(&pixels.stdout).unwrap();
    assert_eq!(pixels_json["width"], 2);
    assert_eq!(pixels_json["height"], 2);
    assert!(pixels_json["presented_pixel_hash"].as_str().unwrap().len() >= 64);

    let view = run_medkit(["dicom", "view", image.to_str().unwrap(), "--width", "2"]);
    assert!(view.stdout.contains("DICOM 2x2 MONOCHROME2"));
    assert!(view.stdout.contains("transfer syntax: 1.2.840.10008.1.2.1"));

    fs::write(
        root.join("labels.csv"),
        "patient_id,study_instance_uid,Pneumonia\npatient-1,1.2.3,1\n",
    )
    .unwrap();
    let manifest = root.join("manifest.jsonl");
    let cxr_manifest = run_medkit([
        "cxr",
        "manifest",
        "--dicom-index",
        index.to_str().unwrap(),
        "--labels",
        root.join("labels.csv").to_str().unwrap(),
        "--out",
        manifest.to_str().unwrap(),
    ]);
    assert!(cxr_manifest.stdout.contains("CXR records: 1"));
    let cxr_record: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&manifest).unwrap()).unwrap();
    assert_eq!(cxr_record["source_format"], "dicom");
    assert_eq!(cxr_record["labels"]["Pneumonia"], 1);

    let splits = root.join("splits.json");
    run_medkit([
        "cxr",
        "split",
        manifest.to_str().unwrap(),
        "--by",
        "patient_id",
        "--train",
        "1.0",
        "--val",
        "0.0",
        "--test",
        "0.0",
        "--out",
        splits.to_str().unwrap(),
    ]);

    fs::write(
        root.join("plan.toml"),
        "name = \"cxr-dicom-cli-test\"\n[image]\nsize = [4, 4]\n",
    )
    .unwrap();
    let cache = root.join("cache");
    let cache_output = run_medkit([
        "cxr",
        "cache",
        manifest.to_str().unwrap(),
        "--splits",
        splits.to_str().unwrap(),
        "--plan",
        root.join("plan.toml").to_str().unwrap(),
        "--cache",
        cache.to_str().unwrap(),
    ]);
    assert!(cache_output.stdout.contains("Failed samples: 0"));

    let validation = run_medkit([
        "cxr",
        "validate-cache",
        cache.to_str().unwrap(),
        "--split",
        "train",
        "--targets",
        "Pneumonia",
        "--image-shape",
        "1,1,4,4",
        "--plan",
        root.join("plan.toml").to_str().unwrap(),
    ]);
    assert!(validation.stdout.contains("Status: ok"));
}

#[test]
fn dicom_cli_reports_usage_and_render_errors() {
    let missing_action = run_medkit_fail(["dicom"]);
    assert!(missing_action.stderr.contains("Usage:"));

    let unknown = run_medkit_fail(["dicom", "browse"]);
    assert!(unknown.stderr.contains("unknown dicom command"));

    let missing_scan_output = run_medkit_fail(["dicom", "scan", "."]);
    assert!(missing_scan_output.stderr.contains("missing --out"));

    let bad_pixels = run_medkit_fail(["dicom", "pixels", "explain", "image.dcm"]);
    assert!(bad_pixels.stderr.contains("unknown dicom pixels command"));

    let root = temp_case_dir("dicom-cli-render-error");
    let image = root.join("image.dcm");
    write_dicom_fixture(&image);
    let render = run_medkit_fail(["dicom", "view", image.to_str().unwrap(), "--width", "0"]);
    assert!(render
        .stderr
        .contains("render width must be greater than zero"));
}

struct CommandOutput {
    stdout: String,
    stderr: String,
}

fn run_medkit<const N: usize>(args: [&str; N]) -> CommandOutput {
    let output = Command::new(env!("CARGO_BIN_EXE_medkit"))
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    CommandOutput {
        stdout: String::from_utf8(output.stdout).unwrap(),
        stderr: String::from_utf8(output.stderr).unwrap(),
    }
}

fn run_medkit_fail<const N: usize>(args: [&str; N]) -> CommandOutput {
    let output = Command::new(env!("CARGO_BIN_EXE_medkit"))
        .args(args)
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.status.code(), Some(2));
    CommandOutput {
        stdout: String::from_utf8(output.stdout).unwrap(),
        stderr: String::from_utf8(output.stderr).unwrap(),
    }
}

fn temp_case_dir(case: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir =
        std::env::temp_dir().join(format!("medkit-cli-{case}-{}-{nanos}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_dicom_fixture(path: &Path) {
    let mut bytes = vec![0u8; 128];
    bytes.extend_from_slice(b"DICM");
    push_text(&mut bytes, (0x0002, 0x0010), "UI", "1.2.840.10008.1.2.1");
    push_text(&mut bytes, (0x0010, 0x0020), "LO", "patient-1");
    push_text(&mut bytes, (0x0020, 0x000D), "UI", "1.2.3");
    push_text(&mut bytes, (0x0020, 0x000E), "UI", "1.2.3.4");
    push_text(&mut bytes, (0x0008, 0x0018), "UI", "1.2.3.4.5");
    push_text(&mut bytes, (0x0008, 0x0060), "CS", "DX");
    push_text(&mut bytes, (0x0018, 0x5101), "CS", "PA");
    push_text(&mut bytes, (0x0028, 0x0030), "DS", "0.5\\0.5");
    push_text(&mut bytes, (0x0028, 0x0004), "CS", "MONOCHROME2");
    push_u16(&mut bytes, (0x0028, 0x0002), 1);
    push_u16(&mut bytes, (0x0028, 0x0010), 2);
    push_u16(&mut bytes, (0x0028, 0x0011), 2);
    push_u16(&mut bytes, (0x0028, 0x0100), 8);
    push_u16(&mut bytes, (0x0028, 0x0101), 8);
    push_u16(&mut bytes, (0x0028, 0x0102), 7);
    push_u16(&mut bytes, (0x0028, 0x0103), 0);
    push_element(&mut bytes, (0x7FE0, 0x0010), "OB", vec![0, 64, 128, 255]);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
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
