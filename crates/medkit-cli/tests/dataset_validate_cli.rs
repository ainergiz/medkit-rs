use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

const HEADER_LEN: usize = 348;

#[test]
fn dataset_validate_command_writes_manifest_and_report() {
    let root = temp_case_dir("cli-validation");
    let images = root.join("imagesTr");
    let labels = root.join("labelsTr");
    fs::create_dir_all(&images).unwrap();
    fs::create_dir_all(&labels).unwrap();

    write_nifti(
        &images.join("valid_0000.nii"),
        &[8, 8, 4],
        4,
        &[1.0, 1.0, 2.0],
    );
    write_nifti(&labels.join("valid.nii"), &[8, 8, 4], 2, &[1.0, 1.0, 2.0]);
    write_nifti(
        &images.join("bad_spacing_0000.nii"),
        &[8, 8, 4],
        4,
        &[1.0, 1.0, 2.0],
    );
    write_nifti(
        &labels.join("bad_spacing.nii"),
        &[8, 8, 4],
        2,
        &[1.0, 1.5, 2.0],
    );

    let manifest_path = root.join("artifacts").join("manifest.json");
    let report_path = root.join("artifacts").join("report.txt");
    let output = Command::new(env!("CARGO_BIN_EXE_medkit"))
        .args([
            "dataset",
            "validate",
            root.to_str().unwrap(),
            "--out",
            manifest_path.to_str().unwrap(),
            "--report",
            report_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Cases: 2"));
    assert!(stdout.contains("Valid: 1"));
    assert!(stdout.contains("Invalid: 1"));
    assert!(stdout.contains("Wrote manifest:"));

    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    assert_eq!(manifest["summary"]["total_cases"], 2);
    assert_eq!(manifest["summary"]["valid_cases"], 1);
    assert_eq!(manifest["summary"]["invalid_cases"], 1);

    let report = fs::read_to_string(&report_path).unwrap();
    assert!(report.contains("Problems:"));
    assert!(report.contains("bad_spacing"));
    assert!(report.contains("GeometryMismatch"));
}

#[test]
fn dataset_validate_command_reports_usage_for_bad_arguments() {
    let output = Command::new(env!("CARGO_BIN_EXE_medkit"))
        .arg("--help")
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8(output.stderr).unwrap().contains("Usage:"));
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

fn write_nifti(path: &Path, dims: &[i16], datatype: i16, spacing: &[f32]) {
    let mut bytes = [0_u8; HEADER_LEN];
    put_i32(&mut bytes, 0, 348);
    put_i16(&mut bytes, 40, i16::try_from(dims.len()).unwrap());
    for (index, dim) in dims.iter().enumerate() {
        put_i16(&mut bytes, 42 + index * 2, *dim);
    }
    put_i16(&mut bytes, 70, datatype);
    put_i16(&mut bytes, 72, bitpix_for(datatype));
    put_f32(&mut bytes, 76, 1.0);
    for (index, value) in spacing.iter().enumerate() {
        put_f32(&mut bytes, 80 + index * 4, *value);
    }
    put_f32(&mut bytes, 108, 352.0);
    bytes[344..348].copy_from_slice(b"n+1\0");

    let mut file = bytes.to_vec();
    file.extend_from_slice(&[0, 0, 0, 0]);
    fs::write(path, file).unwrap();
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

fn bitpix_for(datatype: i16) -> i16 {
    match datatype {
        1 => 1,
        2 | 256 => 8,
        4 | 512 => 16,
        8 | 16 | 768 => 32,
        64 => 64,
        _ => 0,
    }
}
