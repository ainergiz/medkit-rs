#[allow(dead_code)]
mod support;

use std::{error::Error, fs, io, path::PathBuf};

use medkit_core::GeometryMismatch;
use medkit_dataset::{
    render_report, validate_dataset, write_manifest_json, write_report, CaseManifest, CaseStatus,
    DatasetError, DatasetManifest, Problem, ProblemCode, ValidationConfig, ValidationSummary,
};
use support::{create_dataset_dirs, temp_case_dir, NiftiFixture};

fn fixture(dims: &[i16], dtype: i16, spacing: &[f32]) -> NiftiFixture {
    NiftiFixture::new(dims, dtype, spacing)
}

#[test]
fn dataset_errors_report_sources_and_messages() {
    let io_error = DatasetError::Io {
        path: PathBuf::from("imagesTr"),
        source: io::Error::new(io::ErrorKind::NotFound, "missing directory"),
    };
    assert_eq!(
        io_error.to_string(),
        "filesystem error at imagesTr: missing directory"
    );
    assert_eq!(io_error.source().unwrap().to_string(), "missing directory");

    let json_source = serde_json::from_str::<DatasetManifest>("{").unwrap_err();
    let json_error = DatasetError::Json {
        path: PathBuf::from("manifest.json"),
        source: json_source,
    };
    assert!(json_error
        .to_string()
        .starts_with("failed to write JSON manifest manifest.json:"));
    assert!(json_error.source().is_some());

    let invalid = DatasetError::InvalidInput {
        reason: "imagesTr is a file".to_string(),
    };
    assert_eq!(
        invalid.to_string(),
        "invalid dataset input: imagesTr is a file"
    );
    assert!(invalid.source().is_none());
}

#[test]
fn validates_absolute_and_relative_custom_dirs_with_no_cases() {
    let root = temp_case_dir("custom-empty");
    let images = root.join("custom_images");
    let labels = root.join("custom_labels");
    fs::create_dir_all(&images).unwrap();
    fs::create_dir_all(&labels).unwrap();

    let manifest = validate_dataset(
        &ValidationConfig::new(&root)
            .images_dir(&images)
            .labels_dir(PathBuf::from("custom_labels")),
    )
    .unwrap();

    assert_eq!(manifest.images_dir, images.to_string_lossy().into_owned());
    assert_eq!(manifest.labels_dir, labels.to_string_lossy().into_owned());
    assert_eq!(
        manifest.summary,
        ValidationSummary {
            total_cases: 0,
            valid_cases: 0,
            invalid_cases: 0,
            missing_images: 0,
            missing_labels: 0,
            geometry_mismatches: 0,
            read_errors: 0,
        }
    );
    assert_eq!(manifest.cases, Vec::<CaseManifest>::new());
    assert!(render_report(&manifest).contains("Problems: none"));
}

#[test]
fn config_accessors_return_original_paths() {
    let config = ValidationConfig::new("dataset")
        .images_dir("custom-images")
        .labels_dir("/tmp/custom-labels");

    assert_eq!(config.root(), PathBuf::from("dataset").as_path());
    assert_eq!(
        config.images_dir_path(),
        PathBuf::from("custom-images").as_path()
    );
    assert_eq!(
        config.labels_dir_path(),
        PathBuf::from("/tmp/custom-labels").as_path()
    );
}

#[test]
fn rejects_configured_image_directory_that_is_a_file() {
    let root = temp_case_dir("image-dir-file");
    fs::write(root.join("imagesTr"), b"not a directory").unwrap();
    fs::create_dir_all(root.join("labelsTr")).unwrap();

    let error = validate_dataset(&ValidationConfig::new(&root)).unwrap_err();

    match &error {
        DatasetError::InvalidInput { reason } => {
            assert!(reason.contains("image directory is not a directory"));
        }
        other => panic!("expected invalid input, got {other:?}"),
    }
    assert!(error.source().is_none());
}

#[test]
fn rejects_configured_label_directory_that_is_a_file() {
    let root = temp_case_dir("label-dir-file");
    fs::create_dir_all(root.join("imagesTr")).unwrap();
    fs::write(root.join("labelsTr"), b"not a directory").unwrap();

    let error = validate_dataset(&ValidationConfig::new(&root)).unwrap_err();

    match &error {
        DatasetError::InvalidInput { reason } => {
            assert!(reason.contains("label directory is not a directory"));
        }
        other => panic!("expected invalid input, got {other:?}"),
    }
}

#[test]
fn reports_label_read_error_and_recurses_nested_directories() {
    let root = temp_case_dir("nested-bad-label");
    let (images, labels) = create_dataset_dirs(&root);
    let nested_images = images.join("nested");
    let nested_labels = labels.join("nested");
    fs::create_dir_all(&nested_images).unwrap();
    fs::create_dir_all(&nested_labels).unwrap();
    fixture(&[8, 8, 8], 4, &[1.0, 1.0, 1.0]).write_nii(&nested_images.join("bad_0000.nii"));
    fs::write(nested_labels.join("bad.nii"), b"not a nifti header").unwrap();

    let manifest = validate_dataset(&ValidationConfig::new(&root)).unwrap();
    let case = manifest
        .cases
        .iter()
        .find(|case| case.case_id == "bad")
        .unwrap();

    assert_eq!(case.status, CaseStatus::Invalid);
    assert!(case.image.is_some());
    assert!(case.label.is_none());
    assert!(case
        .problems
        .iter()
        .any(|problem| problem.code == ProblemCode::LabelReadError));
    assert_eq!(manifest.summary.read_errors, 1);
}

#[test]
fn reports_duplicate_label_files_for_one_case() {
    let root = temp_case_dir("duplicate-label");
    let (images, labels) = create_dataset_dirs(&root);
    fixture(&[8, 8, 8], 4, &[1.0, 1.0, 1.0]).write_nii(&images.join("dupe_0000.nii"));
    fixture(&[8, 8, 8], 2, &[1.0, 1.0, 1.0]).write_nii(&labels.join("dupe.nii"));
    fixture(&[8, 8, 8], 2, &[1.0, 1.0, 1.0]).write_nii(&labels.join("dupe.hdr"));

    let manifest = validate_dataset(&ValidationConfig::new(&root)).unwrap();
    let case = manifest
        .cases
        .iter()
        .find(|case| case.case_id == "dupe")
        .unwrap();

    assert_eq!(case.status, CaseStatus::Invalid);
    assert!(case
        .problems
        .iter()
        .any(|problem| problem.code == ProblemCode::DuplicateLabel));
}

#[test]
fn geometry_problem_messages_cover_all_mismatch_kinds() {
    let cases = [
        (
            GeometryMismatch::CoordinateSystem {
                left: "LPS".to_string(),
                right: "RAS".to_string(),
            },
            "coordinate system differs: image=LPS, label=RAS",
        ),
        (
            GeometryMismatch::Origin {
                index: 1,
                left: 2.0,
                right: 3.0,
            },
            "origin[1] differs: image=2, label=3",
        ),
        (
            GeometryMismatch::Direction {
                index: 4,
                left: 0.0,
                right: 1.0,
            },
            "direction[4] differs: image=0, label=1",
        ),
    ];

    for (mismatch, expected) in cases {
        let problem = Problem::geometry(&mismatch);
        assert_eq!(problem.code, ProblemCode::GeometryMismatch);
        assert!(problem.message.contains(expected), "{}", problem.message);
    }
}

#[test]
fn write_manifest_json_reports_parent_creation_errors() {
    let root = temp_case_dir("manifest-parent-file");
    let parent = root.join("not-a-directory");
    fs::write(&parent, b"file blocks directory creation").unwrap();
    let manifest = DatasetManifest {
        dataset_root: root.to_string_lossy().into_owned(),
        images_dir: root.join("imagesTr").to_string_lossy().into_owned(),
        labels_dir: root.join("labelsTr").to_string_lossy().into_owned(),
        summary: ValidationSummary::default(),
        cases: Vec::new(),
    };

    let error = write_manifest_json(&manifest, parent.join("manifest.json")).unwrap_err();

    match &error {
        DatasetError::Io { path, .. } => assert_eq!(path, &parent),
        other => panic!("expected io error, got {other:?}"),
    }
    assert!(error.source().is_some());
}

#[test]
fn write_report_reports_parent_creation_errors() {
    let root = temp_case_dir("report-parent-file");
    let parent = root.join("not-a-directory");
    fs::write(&parent, b"file blocks directory creation").unwrap();
    let manifest = DatasetManifest {
        dataset_root: root.to_string_lossy().into_owned(),
        images_dir: root.join("imagesTr").to_string_lossy().into_owned(),
        labels_dir: root.join("labelsTr").to_string_lossy().into_owned(),
        summary: ValidationSummary::default(),
        cases: Vec::new(),
    };

    let error = write_report(&manifest, parent.join("report.txt")).unwrap_err();

    match &error {
        DatasetError::Io { path, .. } => assert_eq!(path, &parent),
        other => panic!("expected io error, got {other:?}"),
    }
}

#[test]
fn write_artifacts_support_paths_without_real_parents() {
    let root = temp_case_dir("artifact-current-dir");
    let manifest = DatasetManifest {
        dataset_root: root.to_string_lossy().into_owned(),
        images_dir: root.join("imagesTr").to_string_lossy().into_owned(),
        labels_dir: root.join("labelsTr").to_string_lossy().into_owned(),
        summary: ValidationSummary::default(),
        cases: Vec::new(),
    };
    let original_dir = std::env::current_dir().unwrap();
    std::env::set_current_dir(&root).unwrap();

    write_manifest_json(&manifest, "manifest.json").unwrap();
    write_report(&manifest, "report.txt").unwrap();

    std::env::set_current_dir(original_dir).unwrap();
    assert!(root.join("manifest.json").is_file());
    assert!(root.join("report.txt").is_file());
}
