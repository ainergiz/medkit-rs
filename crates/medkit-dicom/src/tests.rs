use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::{
    parser::DicomDataSet,
    pixel::{
        explain_pixels, present_dicom_pixels, present_dicom_pixels_with_backend,
        NativeDecoderBackend,
    },
    scan::write_scan_outputs,
    types::{
        DicomFileConfig, DicomScanConfig, DicomViewConfig, EXPLICIT_VR_BIG_ENDIAN,
        EXPLICIT_VR_LITTLE_ENDIAN, IMPLICIT_VR_LITTLE_ENDIAN,
    },
    view::{render_unicode, RenderOptions},
    *,
};

#[test]
fn scan_recurses_indexes_realistic_part10_files_and_reports_duplicates() {
    let root = unique_test_dir();
    fs::create_dir_all(root.join("nested")).unwrap();
    let first = root.join("nested/first.dcm");
    let second = root.join("second.dc");
    let extensionless = root.join("extensionless");
    write_fixture(
        &first,
        FixtureSpec {
            patient_id: "p1",
            sop_uid: "1.2.3.4",
            pixels: vec![0, 64, 128, 255],
            ..FixtureSpec::default()
        },
    );
    write_fixture(
        &second,
        FixtureSpec {
            patient_id: "p2",
            sop_uid: "1.2.3.4",
            pixels: vec![0, 64, 128, 255],
            ..FixtureSpec::default()
        },
    );
    write_fixture(
        &extensionless,
        FixtureSpec {
            patient_id: "p3",
            sop_uid: "9.8.7",
            pixels: vec![1, 2, 3, 4],
            ..FixtureSpec::default()
        },
    );
    fs::write(root.join("notes.txt"), "not dicom").unwrap();

    let config = DicomScanConfig {
        root: root.clone(),
        out_path: root.join("out/index.jsonl"),
        report_path: root.join("out/report.md"),
    };
    let (summary, records) = scan_dicom(&config).unwrap();
    assert_eq!(summary.records, 3);
    assert_eq!(summary.errors.len(), 0);
    assert_eq!(summary.duplicate_sop_instance_uids, 1);
    assert_eq!(summary.duplicate_pixel_hashes, 1);
    assert!(records
        .iter()
        .any(|record| record.path.ends_with("extensionless")));
    assert!(records.iter().any(|record| {
        record
            .warnings
            .iter()
            .any(|warning| warning.code == "duplicate_sop_instance_uid")
    }));

    write_scan_outputs(&summary, &records, &config.out_path, &config.report_path).unwrap();
    assert_eq!(
        fs::read_to_string(&config.out_path)
            .unwrap()
            .lines()
            .count(),
        3
    );
    let report = fs::read_to_string(&config.report_path).unwrap();
    assert!(report.contains("duplicate SOP Instance UIDs: 1"));
    assert!(report.contains("duplicate pixel hashes: 1"));
}

#[test]
fn inventory_extracts_metadata_phi_and_missing_tag_warnings() {
    let root = unique_test_dir();
    let path = root.join("with-phi.ima");
    write_fixture(
        &path,
        FixtureSpec {
            patient_id: "patient-x",
            patient_name: Some("Secret^Name"),
            include_spacing: false,
            view_position: None,
            modality: "CR",
            photometric: "MONOCHROME2",
            pixels: vec![5, 6, 7, 8],
            ..FixtureSpec::default()
        },
    );
    let report = inspect_dicom_file(&path).unwrap();
    assert_eq!(report.record.patient_id.as_deref(), Some("patient-x"));
    assert_eq!(report.record.modality.as_deref(), Some("CR"));
    assert_eq!(report.record.rows, Some(2));
    assert_eq!(report.record.columns, Some(2));
    assert_eq!(report.record.bits_allocated, Some(8));
    assert_eq!(
        report.record.pixel_representation.as_deref(),
        Some("unsigned")
    );
    assert_eq!(report.record.transfer_syntax_uid, EXPLICIT_VR_LITTLE_ENDIAN);
    assert!(report.record.pixel_hash.is_some());
    assert!(report
        .record
        .warnings
        .iter()
        .any(|warning| warning.code == "phi_patient_name"));
    assert!(report
        .record
        .warnings
        .iter()
        .any(|warning| warning.code == "missing_view_position"));
    assert!(report
        .record
        .warnings
        .iter()
        .any(|warning| warning.code == "missing_pixel_spacing"));
    assert_eq!(report.elements["(0010,0020)"], "patient-x");
}

#[test]
fn parser_handles_transfer_syntaxes_and_parse_errors() {
    let root = unique_test_dir();
    let implicit = root.join("implicit");
    write_fixture(
        &implicit,
        FixtureSpec {
            transfer_syntax: IMPLICIT_VR_LITTLE_ENDIAN,
            implicit_vr: true,
            pixels: vec![1, 2, 3, 4],
            ..FixtureSpec::default()
        },
    );
    let record = DicomDataSet::from_file(&implicit)
        .unwrap()
        .inventory_record();
    assert_eq!(record.transfer_syntax_uid, IMPLICIT_VR_LITTLE_ENDIAN);
    assert_eq!(record.patient_id.as_deref(), Some("patient-1"));

    let big = root.join("big.dcm");
    write_fixture(
        &big,
        FixtureSpec {
            transfer_syntax: EXPLICIT_VR_BIG_ENDIAN,
            big_endian: true,
            bits_allocated: 16,
            bits_stored: 12,
            pixels: u16_pixels(&[0x0FFF, 0x8001, 0x0002, 0x0003], true),
            ..FixtureSpec::default()
        },
    );
    let image = present_dicom_pixels(&big).unwrap();
    assert_eq!(image.width, 2);
    assert_eq!(image.height, 2);

    let truncated = root.join("truncated.dcm");
    fs::write(&truncated, b"not dicom").unwrap();
    assert!(DicomDataSet::from_file(&truncated)
        .unwrap_err()
        .to_string()
        .contains("missing DICOM Part 10 preamble"));
}

#[test]
fn presentation_applies_rescale_window_monochrome1_and_explains_steps() {
    let root = unique_test_dir();
    let path = root.join("mono1.dcm");
    write_fixture(
        &path,
        FixtureSpec {
            bits_allocated: 16,
            bits_stored: 16,
            pixel_representation: 1,
            photometric: "MONOCHROME1",
            rescale_slope: Some(2.0),
            rescale_intercept: Some(-10.0),
            window_center: Some(0.0),
            window_width: Some(20.0),
            pixels: u16_pixels(&[0, 5, 10, 20], false),
            ..FixtureSpec::default()
        },
    );
    let image = present_dicom_pixels(&path).unwrap();
    assert_eq!(image.pixels.len(), 4);
    assert_eq!(image.explanation.min_value, -10.0);
    assert_eq!(image.explanation.max_value, 30.0);
    assert!(image
        .explanation
        .steps
        .iter()
        .any(|step| step.contains("apply rescale")));
    assert!(image
        .explanation
        .steps
        .iter()
        .any(|step| step.contains("apply window")));
    assert!(image
        .explanation
        .steps
        .iter()
        .any(|step| step.contains("invert MONOCHROME1")));
    assert_eq!(explain_pixels(&path).unwrap(), image.explanation);
}

#[test]
fn pixel_errors_are_specific() {
    let root = unique_test_dir();
    let unsupported = root.join("jpeg2000.dcm");
    write_fixture(
        &unsupported,
        FixtureSpec {
            transfer_syntax: "1.2.840.10008.1.2.4.91",
            pixels: vec![1, 2, 3, 4],
            ..FixtureSpec::default()
        },
    );
    assert!(present_dicom_pixels(&unsupported)
        .unwrap_err()
        .to_string()
        .contains("unsupported transfer syntax"));

    let bad_length = root.join("bad-length.dcm");
    write_fixture(
        &bad_length,
        FixtureSpec {
            bits_allocated: 16,
            bits_stored: 16,
            pixels: vec![1, 2],
            ..FixtureSpec::default()
        },
    );
    assert!(present_dicom_pixels(&bad_length)
        .unwrap_err()
        .to_string()
        .contains("PixelData length mismatch"));

    let rgb = root.join("rgb.dcm");
    write_fixture(
        &rgb,
        FixtureSpec {
            samples_per_pixel: 3,
            pixels: vec![1; 12],
            ..FixtureSpec::default()
        },
    );
    assert!(present_dicom_pixels(&rgb)
        .unwrap_err()
        .to_string()
        .contains("single-sample grayscale"));
}

#[test]
fn rle_lossless_fixture_decodes_through_native_backend() {
    let root = unique_test_dir();
    let rle = root.join("rle.dcm");
    write_fixture(
        &rle,
        FixtureSpec {
            transfer_syntax: RLE_LOSSLESS,
            pixels: rle_single_segment_pixels(&[0, 64, 128, 255]),
            ..FixtureSpec::default()
        },
    );

    let image = present_dicom_pixels(&rle).unwrap();
    let image_with_backend =
        present_dicom_pixels_with_backend(&rle, &NativeDecoderBackend).unwrap();
    assert_eq!(image_with_backend.width, image.width);
    assert_eq!(image.width, 2);
    assert_eq!(image.height, 2);
    assert!(image.explanation.compressed);
    assert_eq!(image.explanation.decoder_backend, "medkit-native");
    assert!(image
        .explanation
        .steps
        .iter()
        .any(|step| step.contains("RLE Lossless")));
}

#[test]
fn parallel_scan_is_byte_stable_and_graph_summarizes_duplicates_and_warnings() {
    let root = unique_test_dir();
    write_fixture(
        &root.join("b/second.dcm"),
        FixtureSpec {
            patient_id: "p2",
            study_uid: "study-2",
            series_uid: "series-2",
            sop_uid: "duplicate-sop",
            modality: "CR",
            pixels: vec![0, 1, 2, 3],
            ..FixtureSpec::default()
        },
    );
    write_fixture(
        &root.join("a/first.dcm"),
        FixtureSpec {
            patient_id: "p1",
            study_uid: "study-1",
            series_uid: "series-1",
            sop_uid: "duplicate-sop",
            modality: "DX",
            rows: 2,
            columns: 2,
            pixels: vec![0, 1, 2, 3],
            ..FixtureSpec::default()
        },
    );
    write_fixture(
        &root.join("a/third.dcm"),
        FixtureSpec {
            patient_id: "p1",
            study_uid: "study-1",
            series_uid: "series-1",
            sop_uid: "third-sop",
            modality: "CR",
            rows: 2,
            columns: 3,
            pixels: vec![0, 1, 2, 3, 4, 5],
            ..FixtureSpec::default()
        },
    );
    fs::write(root.join("broken.dcm"), b"not a part10 file").unwrap();

    let config = DicomScanConfig {
        root: root.clone(),
        out_path: root.join("scan.jsonl"),
        report_path: root.join("scan.md"),
    };
    let (serial, serial_records) = scan_dicom(&config).unwrap();
    let (parallel, mut parallel_records) = scan_dicom_with_workers(&config, 3).unwrap();
    assert_eq!(parallel.records, serial.records);
    assert_eq!(parallel.errors.len(), serial.errors.len());
    assert_eq!(
        parallel_records
            .iter()
            .map(|record| record.path.clone())
            .collect::<Vec<_>>(),
        serial_records
            .iter()
            .map(|record| record.path.clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(parallel.duplicate_sop_instance_uids, 1);
    assert_eq!(parallel.duplicate_pixel_hashes, 1);

    parallel_records[2].series_instance_uid = None;
    let graph = construct_dicom_graph(&root, &parallel_records);
    assert_eq!(graph.patients, 2);
    assert_eq!(graph.instances, 3);
    assert_eq!(graph.duplicate_sop_instance_uids, 1);
    assert_eq!(graph.duplicate_pixel_hashes, 1);
    for code in [
        "duplicate_sop_instance_uid",
        "duplicate_pixel_hash",
        "missing_series_instance_uid",
        "mixed_modality",
        "mixed_dimensions",
    ] {
        assert!(
            graph.warnings.iter().any(|warning| warning.code == code),
            "missing graph warning {code}"
        );
    }
    let mut missing_sop_records = parallel_records.clone();
    missing_sop_records[0].sop_instance_uid = None;
    let missing_sop_graph = construct_dicom_graph(&root, &missing_sop_records);
    assert!(missing_sop_graph
        .warnings
        .iter()
        .any(|warning| warning.code == "missing_sop_instance_uid"));

    let graph_json = root.join("nested-output/graph.json");
    let graph_report = root.join("nested-output/graph.md");
    write_graph_outputs(&graph, &graph_json, &graph_report).unwrap();
    assert!(fs::read_to_string(graph_report)
        .unwrap()
        .contains("DICOM Graph Report"));
    let graph_value: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(graph_json).unwrap()).unwrap();
    assert_eq!(graph_value["instances"], 3);

    let blocked_graph_parent = root.join("blocked-graph-parent");
    fs::write(&blocked_graph_parent, b"file blocks graph json").unwrap();
    assert!(write_graph_outputs(
        &graph,
        blocked_graph_parent.join("graph.json"),
        root.join("ok-report.md")
    )
    .unwrap_err()
    .to_string()
    .contains("File exists"));
    let blocked_report_parent = root.join("blocked-report-parent");
    fs::write(&blocked_report_parent, b"file blocks graph report").unwrap();
    assert!(write_graph_outputs(
        &graph,
        root.join("ok-graph.json"),
        blocked_report_parent.join("graph.md")
    )
    .unwrap_err()
    .to_string()
    .contains("File exists"));

    let browsed = browse_dicom(&DicomBrowseConfig {
        root: root.clone(),
        group: vec![
            "patient".to_string(),
            "study".to_string(),
            "series".to_string(),
        ],
        out_path: unique_test_dir().join("browse/graph.json"),
        report_path: unique_test_dir().join("browse/graph.md"),
        workers: 2,
    })
    .unwrap();
    assert_eq!(browsed.instances, 3);

    let blocked = root.join("blocked-parent");
    fs::write(&blocked, b"file blocks output parent").unwrap();
    assert!(browse_dicom(&DicomBrowseConfig {
        root,
        group: vec!["patient".to_string()],
        out_path: blocked.join("graph.json"),
        report_path: blocked.join("graph.md"),
        workers: 1,
    })
    .unwrap_err()
    .to_string()
    .contains("File exists"));
}

#[test]
fn optional_pydicom_fixtures_scan_browse_and_decode_real_rle_when_available() {
    let root = Path::new("data/dicom-fixtures/pydicom");
    if !root.exists() {
        return;
    }

    let config = DicomScanConfig {
        root: root.to_path_buf(),
        out_path: unique_test_dir().join("pydicom-index.jsonl"),
        report_path: unique_test_dir().join("pydicom-report.md"),
    };
    let (summary, records) = scan_dicom_with_workers(&config, 2).unwrap();
    assert!(summary.records >= 6);
    assert!(records
        .iter()
        .any(|record| record.transfer_syntax_uid == RLE_LOSSLESS));

    let graph = construct_dicom_graph(root, &records);
    assert!(graph.patients >= 2);
    assert!(graph.duplicate_sop_instance_uids >= 1);

    let rle = root.join("MR_small_RLE.dcm");
    let image = present_dicom_pixels(&rle).unwrap();
    assert_eq!(image.width, 64);
    assert_eq!(image.height, 64);
    assert!(image.explanation.compressed);
}

#[test]
fn unicode_view_renders_metadata_and_validates_width() {
    let root = unique_test_dir();
    let path = root.join("view.dcm");
    write_fixture(
        &path,
        FixtureSpec {
            pixels: vec![0, 85, 170, 255],
            ..FixtureSpec::default()
        },
    );
    let rendered = render_unicode(
        &path,
        &RenderOptions {
            width: 2,
            include_metadata: true,
        },
    )
    .unwrap();
    assert!(rendered.contains("DICOM 2x2"));
    assert!(rendered.contains("transfer syntax"));
    assert!(rendered.lines().count() >= 3);
    assert!(render_unicode(
        &path,
        &RenderOptions {
            width: 0,
            include_metadata: false,
        },
    )
    .unwrap_err()
    .to_string()
    .contains("render width"));
}

#[test]
fn config_types_are_plain_data() {
    let root = PathBuf::from("root");
    let scan = DicomScanConfig {
        root: root.clone(),
        out_path: PathBuf::from("index.jsonl"),
        report_path: PathBuf::from("report.md"),
    };
    let file = DicomFileConfig { path: root.clone() };
    let view = DicomViewConfig {
        path: root,
        width: 80,
    };
    assert_eq!(scan.out_path, PathBuf::from("index.jsonl"));
    assert_eq!(file.path, PathBuf::from("root"));
    assert_eq!(view.width, 80);
}

#[derive(Clone)]
struct FixtureSpec<'a> {
    transfer_syntax: &'a str,
    implicit_vr: bool,
    big_endian: bool,
    patient_id: &'a str,
    patient_name: Option<&'a str>,
    study_uid: &'a str,
    series_uid: &'a str,
    sop_uid: &'a str,
    modality: &'a str,
    view_position: Option<&'a str>,
    photometric: &'a str,
    include_spacing: bool,
    rows: u16,
    columns: u16,
    samples_per_pixel: u16,
    bits_allocated: u16,
    bits_stored: u16,
    pixel_representation: u16,
    rescale_intercept: Option<f32>,
    rescale_slope: Option<f32>,
    window_center: Option<f32>,
    window_width: Option<f32>,
    pixels: Vec<u8>,
}

impl Default for FixtureSpec<'_> {
    fn default() -> Self {
        Self {
            transfer_syntax: EXPLICIT_VR_LITTLE_ENDIAN,
            implicit_vr: false,
            big_endian: false,
            patient_id: "patient-1",
            patient_name: None,
            study_uid: "1.2.3",
            series_uid: "1.2.3.4",
            sop_uid: "1.2.3.4.5",
            modality: "DX",
            view_position: Some("PA"),
            photometric: "MONOCHROME2",
            include_spacing: true,
            rows: 2,
            columns: 2,
            samples_per_pixel: 1,
            bits_allocated: 8,
            bits_stored: 8,
            pixel_representation: 0,
            rescale_intercept: None,
            rescale_slope: None,
            window_center: None,
            window_width: None,
            pixels: vec![0, 1, 2, 3],
        }
    }
}

fn write_fixture(path: &Path, spec: FixtureSpec<'_>) {
    let mut bytes = vec![0u8; 128];
    bytes.extend_from_slice(b"DICM");
    push_explicit(
        &mut bytes,
        (0x0002, 0x0010),
        "UI",
        spec.transfer_syntax.as_bytes(),
        false,
    );
    let implicit = spec.implicit_vr;
    let be = spec.big_endian;
    push_text(
        &mut bytes,
        (0x0010, 0x0020),
        "LO",
        spec.patient_id,
        implicit,
        be,
    );
    if let Some(name) = spec.patient_name {
        push_text(&mut bytes, (0x0010, 0x0010), "PN", name, implicit, be);
    }
    push_text(
        &mut bytes,
        (0x0020, 0x000D),
        "UI",
        spec.study_uid,
        implicit,
        be,
    );
    push_text(
        &mut bytes,
        (0x0020, 0x000E),
        "UI",
        spec.series_uid,
        implicit,
        be,
    );
    push_text(
        &mut bytes,
        (0x0008, 0x0018),
        "UI",
        spec.sop_uid,
        implicit,
        be,
    );
    push_text(
        &mut bytes,
        (0x0008, 0x0060),
        "CS",
        spec.modality,
        implicit,
        be,
    );
    if let Some(view) = spec.view_position {
        push_text(&mut bytes, (0x0018, 0x5101), "CS", view, implicit, be);
    }
    if spec.include_spacing {
        push_text(&mut bytes, (0x0028, 0x0030), "DS", "0.5\\0.6", implicit, be);
    }
    push_text(
        &mut bytes,
        (0x0028, 0x0004),
        "CS",
        spec.photometric,
        implicit,
        be,
    );
    push_u16(
        &mut bytes,
        (0x0028, 0x0002),
        spec.samples_per_pixel,
        implicit,
        be,
    );
    push_u16(&mut bytes, (0x0028, 0x0010), spec.rows, implicit, be);
    push_u16(&mut bytes, (0x0028, 0x0011), spec.columns, implicit, be);
    push_u16(
        &mut bytes,
        (0x0028, 0x0100),
        spec.bits_allocated,
        implicit,
        be,
    );
    push_u16(&mut bytes, (0x0028, 0x0101), spec.bits_stored, implicit, be);
    push_u16(
        &mut bytes,
        (0x0028, 0x0102),
        spec.bits_stored.saturating_sub(1),
        implicit,
        be,
    );
    push_u16(
        &mut bytes,
        (0x0028, 0x0103),
        spec.pixel_representation,
        implicit,
        be,
    );
    if let Some(value) = spec.rescale_intercept {
        push_text(
            &mut bytes,
            (0x0028, 0x1052),
            "DS",
            &value.to_string(),
            implicit,
            be,
        );
    }
    if let Some(value) = spec.rescale_slope {
        push_text(
            &mut bytes,
            (0x0028, 0x1053),
            "DS",
            &value.to_string(),
            implicit,
            be,
        );
    }
    if let Some(value) = spec.window_center {
        push_text(
            &mut bytes,
            (0x0028, 0x1050),
            "DS",
            &value.to_string(),
            implicit,
            be,
        );
    }
    if let Some(value) = spec.window_width {
        push_text(
            &mut bytes,
            (0x0028, 0x1051),
            "DS",
            &value.to_string(),
            implicit,
            be,
        );
    }
    push_pixel_data(&mut bytes, &spec.pixels, implicit, be);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
}

fn rle_single_segment_pixels(raw: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0u8; 64];
    bytes[0..4].copy_from_slice(&1u32.to_le_bytes());
    bytes[4..8].copy_from_slice(&64u32.to_le_bytes());
    bytes.push((raw.len() as u8).saturating_sub(1));
    bytes.extend_from_slice(raw);
    bytes
}

fn push_text(out: &mut Vec<u8>, tag: (u16, u16), vr: &str, value: &str, implicit: bool, be: bool) {
    push_element(out, tag, vr, value.as_bytes().to_vec(), implicit, be);
}

fn push_u16(out: &mut Vec<u8>, tag: (u16, u16), value: u16, implicit: bool, be: bool) {
    let bytes = if be {
        value.to_be_bytes()
    } else {
        value.to_le_bytes()
    };
    push_element(out, tag, "US", bytes.to_vec(), implicit, be);
}

fn push_pixel_data(out: &mut Vec<u8>, value: &[u8], implicit: bool, be: bool) {
    push_element(out, (0x7FE0, 0x0010), "OB", value.to_vec(), implicit, be);
}

fn push_element(
    out: &mut Vec<u8>,
    tag: (u16, u16),
    vr: &str,
    mut value: Vec<u8>,
    implicit: bool,
    be: bool,
) {
    if value.len() % 2 == 1 {
        value.push(if vr == "UI" { 0 } else { b' ' });
    }
    if implicit {
        if be {
            out.extend_from_slice(&tag.0.to_be_bytes());
            out.extend_from_slice(&tag.1.to_be_bytes());
        } else {
            out.extend_from_slice(&tag.0.to_le_bytes());
            out.extend_from_slice(&tag.1.to_le_bytes());
        }
        push_u32(out, value.len() as u32, be);
        out.extend_from_slice(&value);
    } else {
        push_explicit(out, tag, vr, &value, be);
    }
}

fn push_explicit(out: &mut Vec<u8>, tag: (u16, u16), vr: &str, value: &[u8], be: bool) {
    let mut value = value.to_vec();
    if value.len() % 2 == 1 {
        value.push(if vr == "UI" { 0 } else { b' ' });
    }
    if be {
        out.extend_from_slice(&tag.0.to_be_bytes());
        out.extend_from_slice(&tag.1.to_be_bytes());
    } else {
        out.extend_from_slice(&tag.0.to_le_bytes());
        out.extend_from_slice(&tag.1.to_le_bytes());
    }
    out.extend_from_slice(vr.as_bytes());
    if matches!(vr, "OB" | "OW" | "SQ" | "UN" | "UT") {
        out.extend_from_slice(&[0, 0]);
        push_u32(out, value.len() as u32, be);
    } else if be {
        out.extend_from_slice(&(value.len() as u16).to_be_bytes());
    } else {
        out.extend_from_slice(&(value.len() as u16).to_le_bytes());
    }
    out.extend_from_slice(&value);
}

fn push_u32(out: &mut Vec<u8>, value: u32, be: bool) {
    if be {
        out.extend_from_slice(&value.to_be_bytes());
    } else {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn u16_pixels(values: &[u16], be: bool) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| {
            if be {
                value.to_be_bytes()
            } else {
                value.to_le_bytes()
            }
        })
        .collect()
}

fn unique_test_dir() -> PathBuf {
    static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let sequence = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "medkit-dicom-test-{}-{}-{}",
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
