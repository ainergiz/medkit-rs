use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use crate::{
    parser::DicomDataSet,
    types::{
        DicomInventoryRecord, DicomScanConfig, DicomScanError, DicomScanSummary, DicomWarning,
    },
    DicomError, Result,
};

pub fn scan_dicom(
    config: &DicomScanConfig,
) -> Result<(DicomScanSummary, Vec<DicomInventoryRecord>)> {
    let mut paths = Vec::new();
    collect_candidate_paths(&config.root, &mut paths)?;
    paths.sort();

    let mut records = Vec::new();
    let mut errors = Vec::new();
    for path in paths {
        match DicomDataSet::from_file(&path) {
            Ok(dataset) => records.push(dataset.inventory_record()),
            Err(error) => errors.push(DicomScanError {
                path: path.display().to_string(),
                message: error.to_string(),
            }),
        }
    }
    add_duplicate_warnings(&mut records);
    let summary = scan_summary(config, &records, errors);
    Ok((summary, records))
}

pub fn write_scan_outputs(
    summary: &DicomScanSummary,
    records: &[DicomInventoryRecord],
    out_path: impl AsRef<Path>,
    report_path: impl AsRef<Path>,
) -> Result<()> {
    write_jsonl(out_path.as_ref(), records)?;
    write_report(report_path.as_ref(), summary, records)?;
    Ok(())
}

fn collect_candidate_paths(root: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
    if root.is_file() {
        if is_dicom_candidate(root) {
            paths.push(root.to_path_buf());
        }
        return Ok(());
    }
    for entry in fs::read_dir(root).map_err(|source| DicomError::io(root, source))? {
        let path = dir_entry_path(root, entry)?;
        if path.is_dir() {
            collect_candidate_paths(&path, paths)?;
        } else if is_dicom_candidate(&path) {
            paths.push(path);
        }
    }
    Ok(())
}

fn dir_entry_path(root: &Path, entry: std::io::Result<fs::DirEntry>) -> Result<PathBuf> {
    match entry {
        Ok(entry) => Ok(entry.path()),
        Err(source) => Err(DicomError::io(root, source)),
    }
}

fn is_dicom_candidate(path: &Path) -> bool {
    match path.extension().and_then(|value| value.to_str()) {
        Some(ext) => matches!(
            ext.to_ascii_lowercase().as_str(),
            "dc" | "dcm" | "dicom" | "ima"
        ),
        None => true,
    }
}

fn add_duplicate_warnings(records: &mut [DicomInventoryRecord]) {
    let mut sop_paths: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut pixel_paths: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for record in records.iter() {
        if let Some(uid) = &record.sop_instance_uid {
            sop_paths
                .entry(uid.clone())
                .or_default()
                .push(record.path.clone());
        }
        if let Some(hash) = &record.pixel_hash {
            pixel_paths
                .entry(hash.clone())
                .or_default()
                .push(record.path.clone());
        }
    }
    let duplicate_sops = sop_paths
        .into_iter()
        .filter(|(_uid, paths)| paths.len() > 1)
        .collect::<BTreeMap<_, _>>();
    let duplicate_pixels = pixel_paths
        .into_iter()
        .filter(|(_hash, paths)| paths.len() > 1)
        .collect::<BTreeMap<_, _>>();

    for record in records {
        if let Some(uid) = &record.sop_instance_uid {
            if let Some(paths) = duplicate_sops.get(uid) {
                record.warnings.push(DicomWarning::new(
                    "duplicate_sop_instance_uid",
                    format!("SOPInstanceUID {uid} appears in {}", paths.join(", ")),
                ));
            }
        }
        if let Some(hash) = &record.pixel_hash {
            if let Some(paths) = duplicate_pixels.get(hash) {
                record.warnings.push(DicomWarning::new(
                    "duplicate_pixel_hash",
                    format!("decoded pixel hash {hash} appears in {}", paths.join(", ")),
                ));
            }
        }
    }
}

fn scan_summary(
    config: &DicomScanConfig,
    records: &[DicomInventoryRecord],
    errors: Vec<DicomScanError>,
) -> DicomScanSummary {
    let warnings = records
        .iter()
        .map(|record| record.warnings.len())
        .sum::<usize>();
    let duplicate_sop_instance_uids = duplicate_count(
        records
            .iter()
            .filter_map(|record| record.sop_instance_uid.as_deref()),
    );
    let duplicate_pixel_hashes = duplicate_count(
        records
            .iter()
            .filter_map(|record| record.pixel_hash.as_deref()),
    );
    DicomScanSummary {
        root: config.root.display().to_string(),
        records: records.len(),
        errors,
        warnings,
        duplicate_sop_instance_uids,
        duplicate_pixel_hashes,
        out_path: config.out_path.display().to_string(),
        report_path: config.report_path.display().to_string(),
    }
}

fn duplicate_count<'a>(values: impl Iterator<Item = &'a str>) -> usize {
    let mut seen = BTreeSet::new();
    let mut duplicates = BTreeSet::new();
    for value in values {
        if !seen.insert(value.to_string()) {
            duplicates.insert(value.to_string());
        }
    }
    duplicates.len()
}

fn write_jsonl(path: &Path, records: &[DicomInventoryRecord]) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|source| DicomError::io(parent, source))?;
    }
    let mut body = String::new();
    for record in records {
        let line = serde_json::to_string(record).expect("DICOM inventory records serialize");
        body.push_str(&line);
        body.push('\n');
    }
    fs::write(path, body).map_err(|source| DicomError::io(path, source))
}

fn write_report(
    path: &Path,
    summary: &DicomScanSummary,
    records: &[DicomInventoryRecord],
) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|source| DicomError::io(parent, source))?;
    }
    let mut report = String::new();
    report.push_str("# DICOM QA Report\n\n");
    report.push_str(&format!("- records: {}\n", summary.records));
    report.push_str(&format!("- errors: {}\n", summary.errors.len()));
    report.push_str(&format!("- warnings: {}\n", summary.warnings));
    report.push_str(&format!(
        "- duplicate SOP Instance UIDs: {}\n",
        summary.duplicate_sop_instance_uids
    ));
    report.push_str(&format!(
        "- duplicate pixel hashes: {}\n",
        summary.duplicate_pixel_hashes
    ));
    report.push_str("\n## Errors\n\n");
    if summary.errors.is_empty() {
        report.push_str("- none\n");
    } else {
        for error in &summary.errors {
            report.push_str(&format!("- {}: {}\n", error.path, error.message));
        }
    }
    report.push_str("\n## Warnings\n\n");
    let mut any_warning = false;
    for record in records {
        for warning in &record.warnings {
            any_warning = true;
            report.push_str(&format!(
                "- {}: {} ({})\n",
                record.path, warning.message, warning.code
            ));
        }
    }
    if !any_warning {
        report.push_str("- none\n");
    }
    fs::write(path, report).map_err(|source| DicomError::io(path, source))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_file_root_records_parse_errors() {
        let root = unique_path("scan-file-root.dcm");
        fs::write(&root, b"not dicom").unwrap();
        let config = DicomScanConfig {
            root: root.clone(),
            out_path: root.with_extension("jsonl"),
            report_path: root.with_extension("md"),
        };

        let (summary, records) = scan_dicom(&config).unwrap();
        assert!(records.is_empty());
        assert_eq!(summary.errors.len(), 1);
        assert!(summary.errors[0].message.contains("missing DICOM"));
        let _ = fs::remove_file(root);
    }

    #[test]
    fn scan_missing_root_and_nondicom_file_roots_are_handled() {
        let missing = unique_path("missing-root");
        let config = DicomScanConfig {
            root: missing.clone(),
            out_path: missing.with_extension("jsonl"),
            report_path: missing.with_extension("md"),
        };
        assert!(scan_dicom(&config)
            .unwrap_err()
            .to_string()
            .contains("No such file"));

        let text = unique_path("not-dicom.txt");
        fs::write(&text, b"not dicom").unwrap();
        let config = DicomScanConfig {
            root: text.clone(),
            out_path: text.with_extension("jsonl"),
            report_path: text.with_extension("md"),
        };
        let (summary, records) = scan_dicom(&config).unwrap();
        assert_eq!(summary.records, 0);
        assert!(records.is_empty());
        let _ = fs::remove_file(text);
    }

    #[cfg(unix)]
    #[test]
    fn scan_propagates_recursive_read_errors() {
        use std::os::unix::fs::PermissionsExt;

        let root = unique_dir("recursive-read-error");
        let closed = root.join("closed");
        fs::create_dir_all(&closed).unwrap();
        let original_permissions = fs::metadata(&closed).unwrap().permissions();
        fs::set_permissions(&closed, fs::Permissions::from_mode(0)).unwrap();

        let config = DicomScanConfig {
            root: root.clone(),
            out_path: root.join("index.jsonl"),
            report_path: root.join("report.md"),
        };
        let error = scan_dicom(&config).unwrap_err().to_string();

        fs::set_permissions(&closed, original_permissions).unwrap();
        let _ = fs::remove_dir_all(root);
        assert!(error.contains("closed"));
    }

    #[test]
    fn directory_entry_errors_keep_root_context() {
        let error = dir_entry_path(
            Path::new("root-dir"),
            Err(std::io::Error::new(std::io::ErrorKind::Other, "entry lost")),
        )
        .unwrap_err();

        assert_eq!(error.to_string(), "root-dir: entry lost");
    }

    #[test]
    fn duplicate_warning_helper_handles_missing_ids_and_hashes() {
        let mut records = vec![empty_record("missing")];
        add_duplicate_warnings(&mut records);
        assert!(records[0].warnings.is_empty());
    }

    #[test]
    fn report_writers_cover_no_parent_errors_and_no_warnings() {
        let stem = unique_file_stem("report-no-parent");
        let out_path = PathBuf::from(format!("{stem}.jsonl"));
        let report_path = PathBuf::from(format!("{stem}.md"));
        let summary = DicomScanSummary {
            root: "root".to_string(),
            records: 0,
            errors: vec![DicomScanError {
                path: "broken.dcm".to_string(),
                message: "bad parse".to_string(),
            }],
            warnings: 0,
            duplicate_sop_instance_uids: 0,
            duplicate_pixel_hashes: 0,
            out_path: out_path.display().to_string(),
            report_path: report_path.display().to_string(),
        };

        write_scan_outputs(&summary, &[], &out_path, &report_path).unwrap();
        let report = fs::read_to_string(&report_path).unwrap();
        assert!(report.contains("broken.dcm: bad parse"));
        assert!(report.contains("- none"));

        let _ = fs::remove_file(out_path);
        let _ = fs::remove_file(report_path);
    }

    #[test]
    fn output_writers_propagate_jsonl_and_report_io_errors() {
        let root = unique_dir("writer-errors");
        let parent_file = root.join("not-a-directory");
        fs::write(&parent_file, b"file").unwrap();
        let summary = empty_summary(root.join("ok.jsonl"), root.join("ok.md"));

        assert!(write_scan_outputs(
            &summary,
            &[],
            parent_file.join("index.jsonl"),
            root.join("ok.md")
        )
        .unwrap_err()
        .to_string()
        .contains("not-a-directory"));

        assert!(write_scan_outputs(
            &summary,
            &[],
            root.join("ok.jsonl"),
            parent_file.join("report.md")
        )
        .unwrap_err()
        .to_string()
        .contains("not-a-directory"));

        let dir_path = root.join("existing-dir");
        fs::create_dir_all(&dir_path).unwrap();
        assert!(write_jsonl(&dir_path, &[])
            .unwrap_err()
            .to_string()
            .contains("existing-dir"));
        assert!(write_report(&dir_path, &summary, &[])
            .unwrap_err()
            .to_string()
            .contains("existing-dir"));

        let _ = fs::remove_dir_all(root);
    }

    fn empty_record(path: &str) -> DicomInventoryRecord {
        DicomInventoryRecord {
            path: path.to_string(),
            sha256: "hash".to_string(),
            patient_id: None,
            study_instance_uid: None,
            series_instance_uid: None,
            sop_instance_uid: None,
            modality: None,
            body_part_examined: None,
            view_position: None,
            laterality: None,
            rows: None,
            columns: None,
            samples_per_pixel: None,
            bits_allocated: None,
            bits_stored: None,
            high_bit: None,
            pixel_representation: None,
            photometric_interpretation: None,
            transfer_syntax_uid: "1.2".to_string(),
            pixel_spacing: None,
            imager_pixel_spacing: None,
            rescale_intercept: None,
            rescale_slope: None,
            window_center: None,
            window_width: None,
            pixel_hash: None,
            warnings: Vec::new(),
        }
    }

    fn empty_summary(out_path: PathBuf, report_path: PathBuf) -> DicomScanSummary {
        DicomScanSummary {
            root: "root".to_string(),
            records: 0,
            errors: Vec::new(),
            warnings: 0,
            duplicate_sop_instance_uids: 0,
            duplicate_pixel_hashes: 0,
            out_path: out_path.display().to_string(),
            report_path: report_path.display().to_string(),
        }
    }

    fn unique_path(file_name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "medkit-dicom-scan-{}-{}-{file_name}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn unique_dir(label: &str) -> PathBuf {
        let dir = unique_path(label);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn unique_file_stem(label: &str) -> String {
        format!(
            "medkit-dicom-scan-{}-{}-{label}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        )
    }
}
