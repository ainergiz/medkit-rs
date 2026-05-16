use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use flate2::read::MultiGzDecoder;

use crate::{
    error::CxrError,
    types::{CxrRecord, IndexConfig, IndexSummary, ValidateConfig, ValidationSummary},
    util::{add_label_count, hash_file, overlap_count},
};

pub fn index_cxr(config: &IndexConfig) -> Result<IndexSummary, CxrError> {
    let image_map = scan_images(&config.images_root)?;
    let label_map = match &config.labels_path {
        Some(path) => read_label_csv(path)?,
        None => HashMap::new(),
    };
    let mut records = match &config.metadata_path {
        Some(path) => records_from_metadata(path, &image_map, &label_map, config)?,
        None => records_from_images(&image_map, config)?,
    };
    records.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));
    write_manifest(&config.out_path, &records)?;
    let summary = index_summary(config, &records);
    Ok(summary)
}

pub fn validate_cxr(config: &ValidateConfig) -> Result<ValidationSummary, CxrError> {
    let records = read_manifest(&config.manifest_path)?;
    let mut readable_images = 0usize;
    let mut unreadable_images = 0usize;
    let mut filtered_non_frontal = 0usize;
    let mut split_counts = BTreeMap::new();
    let mut split_patients: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut split_hashes: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut target_counts = BTreeMap::new();

    for record in &records {
        if config.require_frontal && !is_frontal(record.view_position.as_deref()) {
            filtered_non_frontal += 1;
            continue;
        }
        if image::image_dimensions(&record.image_path).is_ok() {
            readable_images += 1;
        } else {
            unreadable_images += 1;
        }
        for (target, value) in &record.labels {
            add_label_count(target_counts.entry(target.clone()).or_default(), *value);
        }
        if let Some(split) = &record.split {
            *split_counts.entry(split.clone()).or_insert(0) += 1;
            split_patients
                .entry(split.clone())
                .or_default()
                .insert(record.patient_id.clone());
            if config.check_duplicates {
                let hash = match &record.sha256 {
                    Some(value) => value.clone(),
                    None => hash_file(Path::new(&record.image_path))?,
                };
                split_hashes.entry(split.clone()).or_default().insert(hash);
            }
        }
    }

    let patient_overlap_count = if config.check_patient_leakage {
        overlap_count(&split_patients)
    } else {
        0
    };
    let duplicate_hash_overlap_count = if config.check_duplicates {
        overlap_count(&split_hashes)
    } else {
        0
    };

    let summary = ValidationSummary {
        records: records.len(),
        readable_images,
        unreadable_images,
        filtered_non_frontal,
        patient_overlap_count,
        duplicate_hash_overlap_count,
        split_counts,
        target_counts,
        report_path: config.report_path.display().to_string(),
    };
    write_validation_report(&config.report_path, &summary)?;
    Ok(summary)
}
fn records_from_metadata(
    metadata_path: &Path,
    image_map: &HashMap<String, PathBuf>,
    label_map: &HashMap<(String, String), BTreeMap<String, Option<i8>>>,
    config: &IndexConfig,
) -> Result<Vec<CxrRecord>, CxrError> {
    let mut reader = csv_reader(metadata_path)?;
    let headers = HeaderIndex::new(reader.headers()?);
    let mut records = Vec::new();
    for row in reader.records() {
        let row = row?;
        let image_id = headers
            .get(&row, &["dicom_id", "image_id", "filename"])
            .ok_or_else(|| CxrError::Message("metadata row missing dicom_id".to_string()))?;
        let subject = headers
            .get(&row, &["subject_id", "patient_id"])
            .unwrap_or_else(|| patient_from_filename(&image_id));
        let study = headers
            .get(&row, &["study_id"])
            .unwrap_or_else(|| "unknown-study".to_string());
        let Some(path) = image_map.get(&image_id) else {
            continue;
        };
        let (width, height) = metadata_or_image_dimensions(&headers, &row, path)?;
        let labels = label_map
            .get(&(subject.clone(), study.clone()))
            .cloned()
            .unwrap_or_default();
        let report_path = config
            .reports_root
            .as_ref()
            .map(|root| root.join(format!("s{study}.txt")).display().to_string());
        records.push(CxrRecord {
            sample_id: format!("p{subject}/s{study}/{image_id}"),
            patient_id: format!("p{subject}"),
            study_id: format!("s{study}"),
            image_id: image_id.clone(),
            image_path: path.display().to_string(),
            source_format: path
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or("unknown")
                .to_ascii_lowercase(),
            modality: headers.get(&row, &["modality"]),
            view_position: headers.get(&row, &["ViewPosition", "view_position"]),
            laterality: headers.get(&row, &["laterality", "Laterality"]),
            width: Some(width),
            height: Some(height),
            photometric_interpretation: Some("MONOCHROME2".to_string()),
            labels,
            label_source: Some("chexpert_csv".to_string()),
            report_path,
            split: None,
            sha256: Some(hash_file(path)?),
        });
    }
    Ok(records)
}

fn records_from_images(
    image_map: &HashMap<String, PathBuf>,
    _config: &IndexConfig,
) -> Result<Vec<CxrRecord>, CxrError> {
    let mut records = Vec::new();
    for (image_id, path) in image_map {
        let (width, height) = image::image_dimensions(path)?;
        let patient = patient_from_filename(image_id);
        records.push(CxrRecord {
            sample_id: format!("{patient}/{image_id}"),
            patient_id: patient.clone(),
            study_id: patient,
            image_id: image_id.clone(),
            image_path: path.display().to_string(),
            source_format: path
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or("unknown")
                .to_ascii_lowercase(),
            modality: Some("CR".to_string()),
            view_position: None,
            laterality: None,
            width: Some(width),
            height: Some(height),
            photometric_interpretation: Some("MONOCHROME2".to_string()),
            labels: BTreeMap::new(),
            label_source: None,
            report_path: None,
            split: None,
            sha256: Some(hash_file(path)?),
        });
    }
    Ok(records)
}

pub(crate) fn read_label_csv(
    labels_path: &Path,
) -> Result<HashMap<(String, String), BTreeMap<String, Option<i8>>>, CxrError> {
    let mut reader = csv_reader(labels_path)?;
    let headers = reader.headers()?.clone();
    let index = HeaderIndex::new(&headers);
    let mut map = HashMap::new();
    for row in reader.records() {
        let row = row?;
        let subject = index
            .get(&row, &["subject_id", "patient_id"])
            .ok_or_else(|| CxrError::Message("labels row missing subject_id".to_string()))?;
        let study = index
            .get(&row, &["study_id"])
            .ok_or_else(|| CxrError::Message("labels row missing study_id".to_string()))?;
        let mut labels = BTreeMap::new();
        for (header_index, header) in headers.iter().enumerate() {
            if matches!(header, "subject_id" | "study_id" | "patient_id") {
                continue;
            }
            let value = row.get(header_index).unwrap_or("").trim();
            labels.insert(header.to_string(), parse_label_value(value));
        }
        map.insert((subject, study), labels);
    }
    Ok(map)
}
fn index_summary(config: &IndexConfig, records: &[CxrRecord]) -> IndexSummary {
    let mut labels = BTreeMap::new();
    for record in records {
        for (target, value) in &record.labels {
            add_label_count(labels.entry(target.clone()).or_default(), *value);
        }
    }
    IndexSummary {
        images_root: config.images_root.display().to_string(),
        metadata_path: config
            .metadata_path
            .as_ref()
            .map(|path| path.display().to_string()),
        labels_path: config
            .labels_path
            .as_ref()
            .map(|path| path.display().to_string()),
        records: records.len(),
        patients: records
            .iter()
            .map(|record| &record.patient_id)
            .collect::<BTreeSet<_>>()
            .len(),
        studies: records
            .iter()
            .map(|record| &record.study_id)
            .collect::<BTreeSet<_>>()
            .len(),
        labels,
        out_path: config.out_path.display().to_string(),
    }
}

fn write_validation_report(path: &Path, summary: &ValidationSummary) -> Result<(), CxrError> {
    let mut report = String::new();
    report.push_str("# CXR Validation Report\n\n");
    report.push_str(&format!("- records: {}\n", summary.records));
    report.push_str(&format!("- readable images: {}\n", summary.readable_images));
    report.push_str(&format!(
        "- unreadable images: {}\n",
        summary.unreadable_images
    ));
    report.push_str(&format!(
        "- filtered non-frontal: {}\n",
        summary.filtered_non_frontal
    ));
    report.push_str(&format!(
        "- patient overlap count: {}\n",
        summary.patient_overlap_count
    ));
    report.push_str(&format!(
        "- duplicate image hash overlap count: {}\n",
        summary.duplicate_hash_overlap_count
    ));
    report.push_str("\n## Splits\n\n");
    for (split, count) in &summary.split_counts {
        report.push_str(&format!("- {split}: {count}\n"));
    }
    report.push_str("\n## Labels\n\n");
    for (target, counts) in &summary.target_counts {
        report.push_str(&format!(
            "- {target}: positive {}, negative {}, uncertain {}, missing {}\n",
            counts.positive, counts.negative, counts.uncertain, counts.missing
        ));
    }
    fs::write(path, report)?;
    Ok(())
}
fn scan_images(root: &Path) -> Result<HashMap<String, PathBuf>, CxrError> {
    let mut map = HashMap::new();
    scan_images_inner(root, &mut map)?;
    Ok(map)
}

fn scan_images_inner(root: &Path, map: &mut HashMap<String, PathBuf>) -> Result<(), CxrError> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            scan_images_inner(&path, map)?;
        } else if is_image_path(&path) {
            if let Some(stem) = path.file_stem().and_then(|value| value.to_str()) {
                map.insert(stem.to_string(), path);
            }
        }
    }
    Ok(())
}

pub(crate) fn is_image_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase())
            .as_deref(),
        Some("jpg" | "jpeg" | "png")
    )
}

fn csv_reader(path: &Path) -> Result<csv::Reader<Box<dyn Read>>, CxrError> {
    let file = File::open(path)?;
    let reader: Box<dyn Read> = if path.extension().and_then(|value| value.to_str()) == Some("gz") {
        Box::new(MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };
    Ok(csv::ReaderBuilder::new().flexible(true).from_reader(reader))
}

struct HeaderIndex {
    headers: HashMap<String, usize>,
}

impl HeaderIndex {
    fn new(headers: &csv::StringRecord) -> Self {
        let headers = headers
            .iter()
            .enumerate()
            .map(|(index, header)| (header.to_ascii_lowercase(), index))
            .collect();
        Self { headers }
    }

    fn get(&self, row: &csv::StringRecord, names: &[&str]) -> Option<String> {
        names.iter().find_map(|name| {
            self.headers
                .get(&name.to_ascii_lowercase())
                .and_then(|index| row.get(*index))
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
        })
    }
}

fn metadata_or_image_dimensions(
    headers: &HeaderIndex,
    row: &csv::StringRecord,
    path: &Path,
) -> Result<(u32, u32), CxrError> {
    let rows = headers
        .get(row, &["Rows", "height"])
        .and_then(|value| value.parse::<u32>().ok());
    let columns = headers
        .get(row, &["Columns", "width"])
        .and_then(|value| value.parse::<u32>().ok());
    match (columns, rows) {
        (Some(width), Some(height)) => Ok((width, height)),
        _ => Ok(image::image_dimensions(path)?),
    }
}

pub(crate) fn parse_label_value(value: &str) -> Option<i8> {
    match value {
        "1" | "1.0" => Some(1),
        "0" | "0.0" => Some(0),
        "-1" | "-1.0" => Some(-1),
        "" => None,
        _ => None,
    }
}

pub(crate) fn patient_from_filename(value: &str) -> String {
    value
        .split('_')
        .next()
        .filter(|part| !part.is_empty())
        .unwrap_or(value)
        .to_string()
}

pub(crate) fn is_frontal(view_position: Option<&str>) -> bool {
    matches!(
        view_position
            .map(|value| value.to_ascii_uppercase())
            .as_deref(),
        Some("PA" | "AP")
    )
}

pub(crate) fn write_manifest(path: &Path, records: &[CxrRecord]) -> Result<(), CxrError> {
    let mut writer = BufWriter::new(File::create(path)?);
    for record in records {
        serde_json::to_writer(&mut writer, record)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

pub fn read_manifest(path: &Path) -> Result<Vec<CxrRecord>, CxrError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        records.push(serde_json::from_str(&line)?);
    }
    Ok(records)
}
