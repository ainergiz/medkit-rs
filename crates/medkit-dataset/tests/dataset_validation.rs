mod support;

use std::{fs, path::Path};

use medkit_dataset::{
    case_id_from_image_path, case_id_from_label_path, render_report, validate_dataset,
    write_manifest_json, write_report, CaseStatus, DatasetLayout, DatasetManifest, ProblemCode,
    ValidationConfig,
};
use support::{create_dataset_dirs, temp_case_dir, write_case, NiftiFixture};

fn fixture(dims: &[i16], dtype: i16, spacing: &[f32]) -> NiftiFixture {
    NiftiFixture::new(dims, dtype, spacing)
}

#[test]
fn derives_case_ids_without_implicit_channel_suffix_stripping() {
    assert_eq!(
        case_id_from_image_path(Path::new("imagesTr/liver_001_0000.nii.gz")).as_deref(),
        Some("liver_001_0000")
    );
    assert_eq!(
        case_id_from_image_path(Path::new("imagesTr/patient_2024.nii.gz")).as_deref(),
        Some("patient_2024")
    );
    assert_eq!(
        case_id_from_image_path(Path::new("imagesTr/liver_001.nii")).as_deref(),
        Some("liver_001")
    );
    assert_eq!(
        case_id_from_image_path(Path::new("imagesTr/liver.nii")).as_deref(),
        Some("liver")
    );
    assert_eq!(
        case_id_from_label_path(Path::new("labelsTr/liver_001.nii.gz")).as_deref(),
        Some("liver_001")
    );
    assert_eq!(case_id_from_image_path(Path::new("notes.txt")), None);
}

#[test]
fn nnunet_layout_strips_only_explicit_image_channel_suffix() {
    let root = temp_case_dir("nnunet-layout");
    let (images, labels) = create_dataset_dirs(&root);
    fixture(&[8, 8, 8], 4, &[1.0, 1.0, 1.0]).write_nii(&images.join("liver_001_0000.nii"));
    fixture(&[8, 8, 8], 2, &[1.0, 1.0, 1.0]).write_nii(&labels.join("liver_001.nii"));
    fixture(&[8, 8, 8], 4, &[1.0, 1.0, 1.0]).write_nii(&images.join("patient_2024.nii"));
    fixture(&[8, 8, 8], 2, &[1.0, 1.0, 1.0]).write_nii(&labels.join("patient_2024.nii"));

    let manifest =
        validate_dataset(&ValidationConfig::new(&root).layout(DatasetLayout::Nnunet)).unwrap();

    assert_eq!(manifest.summary.total_cases, 2);
    assert_eq!(manifest.summary.valid_cases, 2);
    assert!(manifest
        .cases
        .iter()
        .any(|case| case.case_id == "liver_001"));
    assert!(manifest
        .cases
        .iter()
        .any(|case| case.case_id == "patient_2024"));
}

#[test]
fn validates_dataset_end_to_end_and_writes_artifacts() {
    let root = temp_case_dir("validation");
    let (images, labels) = create_dataset_dirs(&root);

    write_case(
        &images,
        &labels,
        "valid",
        Some(fixture(&[16, 16, 8], 4, &[1.0, 1.0, 2.0])),
        Some(fixture(&[16, 16, 8], 2, &[1.0, 1.0, 2.0])),
    );
    write_case(
        &images,
        &labels,
        "shape_mismatch",
        Some(fixture(&[16, 16, 8], 4, &[1.0, 1.0, 2.0])),
        Some(fixture(&[16, 16, 7], 2, &[1.0, 1.0, 2.0])),
    );
    write_case(
        &images,
        &labels,
        "spacing_mismatch",
        Some(fixture(&[16, 16, 8], 4, &[1.0, 1.0, 2.0])),
        Some(fixture(&[16, 16, 8], 2, &[1.0, 1.5, 2.0])),
    );
    write_case(
        &images,
        &labels,
        "missing_label",
        Some(fixture(&[8, 8, 8], 4, &[1.0, 1.0, 1.0])),
        None,
    );
    write_case(
        &images,
        &labels,
        "missing_image",
        None,
        Some(fixture(&[8, 8, 8], 2, &[1.0, 1.0, 1.0])),
    );
    write_case(
        &images,
        &labels,
        "bad_dtype",
        Some(fixture(&[8, 8, 8], 128, &[1.0, 1.0, 1.0])),
        Some(fixture(&[8, 8, 8], 2, &[1.0, 1.0, 1.0])),
    );
    fixture(&[8, 8, 8], 4, &[1.0, 1.0, 1.0]).write_nii(&images.join("duplicate.nii"));
    fixture(&[8, 8, 8], 4, &[1.0, 1.0, 1.0]).write_nii(&images.join("duplicate.hdr"));
    fixture(&[8, 8, 8], 2, &[1.0, 1.0, 1.0]).write_nii(&labels.join("duplicate.nii"));

    let manifest = validate_dataset(&ValidationConfig::new(&root)).unwrap();

    assert_eq!(manifest.summary.total_cases, 7);
    assert_eq!(manifest.summary.valid_cases, 1);
    assert_eq!(manifest.summary.invalid_cases, 6);
    assert_eq!(manifest.summary.missing_images, 1);
    assert_eq!(manifest.summary.missing_labels, 1);
    assert_eq!(manifest.summary.geometry_mismatches, 2);
    assert_eq!(manifest.summary.read_errors, 1);

    let valid = manifest
        .cases
        .iter()
        .find(|case| case.case_id == "valid")
        .unwrap();
    assert_eq!(valid.status, CaseStatus::Valid);
    assert_eq!(valid.image.as_ref().unwrap().dtype, "I16");
    assert_eq!(valid.label.as_ref().unwrap().modality, "Segmentation");

    assert_problem(&manifest, "shape_mismatch", ProblemCode::GeometryMismatch);
    assert_problem(&manifest, "spacing_mismatch", ProblemCode::GeometryMismatch);
    assert_problem(&manifest, "missing_label", ProblemCode::MissingLabel);
    assert_problem(&manifest, "missing_image", ProblemCode::MissingImage);
    assert_problem(&manifest, "bad_dtype", ProblemCode::ImageReadError);
    assert_problem(&manifest, "duplicate", ProblemCode::DuplicateImage);

    let manifest_path = root.join("out").join("manifest.json");
    let report_path = root.join("out").join("report.txt");
    write_manifest_json(&manifest, &manifest_path).unwrap();
    write_report(&manifest, &report_path).unwrap();

    let loaded: DatasetManifest =
        serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    assert_eq!(loaded.summary, manifest.summary);

    let report = fs::read_to_string(&report_path).unwrap();
    assert!(report.contains("Cases: 7"));
    assert!(report.contains("Invalid: 6"));
    assert!(report.contains("shape_mismatch"));
    assert_eq!(report, render_report(&manifest));
}

#[test]
fn reports_invalid_input_directories() {
    let root = temp_case_dir("invalid-input");
    let error = validate_dataset(&ValidationConfig::new(&root)).unwrap_err();

    assert!(error.to_string().contains("filesystem error"));
}

fn assert_problem(manifest: &DatasetManifest, case_id: &str, code: ProblemCode) {
    let case = manifest
        .cases
        .iter()
        .find(|case| case.case_id == case_id)
        .unwrap();
    assert_eq!(case.status, CaseStatus::Invalid);
    assert!(
        case.problems.iter().any(|problem| problem.code == code),
        "{case_id} did not contain {code:?}: {:?}",
        case.problems
    );
}
