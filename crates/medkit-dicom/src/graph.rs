use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use crate::{
    scan::{scan_dicom_with_workers, write_scan_outputs},
    types::{
        DicomBrowseConfig, DicomGraphSummary, DicomGraphWarning, DicomInstanceNode,
        DicomInventoryRecord, DicomPatientNode, DicomSeriesNode, DicomStudyNode,
    },
    DicomError, Result,
};

#[derive(Default)]
struct SeriesAccumulator {
    modality: Option<String>,
    rows: Option<u16>,
    columns: Option<u16>,
    modalities: BTreeSet<String>,
    dimensions: BTreeSet<String>,
    instances: Vec<DicomInstanceNode>,
}

pub fn browse_dicom(config: &DicomBrowseConfig) -> Result<DicomGraphSummary> {
    let scan_config = crate::DicomScanConfig {
        root: config.root.clone(),
        out_path: config.out_path.with_extension("scan.jsonl"),
        report_path: config.report_path.with_extension("scan.md"),
    };
    let (scan_summary, records) = scan_dicom_with_workers(&scan_config, config.workers)?;
    write_scan_outputs(
        &scan_summary,
        &records,
        &scan_config.out_path,
        &scan_config.report_path,
    )?;
    let graph = construct_dicom_graph(&config.root, &records);
    write_graph_outputs(&graph, &config.out_path, &config.report_path)?;
    Ok(graph)
}

pub fn construct_dicom_graph(
    root: impl AsRef<Path>,
    records: &[DicomInventoryRecord],
) -> DicomGraphSummary {
    let mut warnings = Vec::new();
    let mut duplicate_sop_values = duplicate_values(
        records
            .iter()
            .filter_map(|record| record.sop_instance_uid.as_deref()),
    );
    let mut duplicate_pixel_values = duplicate_values(
        records
            .iter()
            .filter_map(|record| record.pixel_hash.as_deref()),
    );

    let mut patients: BTreeMap<String, BTreeMap<String, BTreeMap<String, SeriesAccumulator>>> =
        BTreeMap::new();

    for record in records {
        let patient_id = node_key(record.patient_id.as_deref(), "missing-patient");
        let study_uid = node_key(record.study_instance_uid.as_deref(), "missing-study");
        let series_uid = node_key(record.series_instance_uid.as_deref(), "missing-series");
        if record.series_instance_uid.is_none() {
            warnings.push(DicomGraphWarning {
                code: "missing_series_instance_uid".to_string(),
                message: "SeriesInstanceUID is missing".to_string(),
                path: Some(record.path.clone()),
            });
        }
        if record.sop_instance_uid.is_none() {
            warnings.push(DicomGraphWarning {
                code: "missing_sop_instance_uid".to_string(),
                message: "SOPInstanceUID is missing".to_string(),
                path: Some(record.path.clone()),
            });
        }

        let series = patients
            .entry(patient_id)
            .or_default()
            .entry(study_uid)
            .or_default()
            .entry(series_uid)
            .or_default();

        if series.modality.is_none() {
            series.modality = record.modality.clone();
        }
        if series.rows.is_none() {
            series.rows = record.rows;
        }
        if series.columns.is_none() {
            series.columns = record.columns;
        }
        if let Some(modality) = &record.modality {
            series.modalities.insert(modality.clone());
        }
        if let (Some(rows), Some(columns)) = (record.rows, record.columns) {
            series.dimensions.insert(format!("{rows}x{columns}"));
        }
        if let Some(uid) = &record.sop_instance_uid {
            if duplicate_sop_values.remove(uid) {
                warnings.push(DicomGraphWarning {
                    code: "duplicate_sop_instance_uid".to_string(),
                    message: format!("SOPInstanceUID {uid} appears multiple times"),
                    path: Some(record.path.clone()),
                });
            }
        }
        if let Some(hash) = &record.pixel_hash {
            if duplicate_pixel_values.remove(hash) {
                warnings.push(DicomGraphWarning {
                    code: "duplicate_pixel_hash".to_string(),
                    message: format!("pixel hash {hash} appears multiple times"),
                    path: Some(record.path.clone()),
                });
            }
        }
        series.instances.push(DicomInstanceNode {
            sop_instance_uid: record.sop_instance_uid.clone(),
            path: record.path.clone(),
            modality: record.modality.clone(),
            rows: record.rows,
            columns: record.columns,
            pixel_hash: record.pixel_hash.clone(),
        });
    }

    let mut patient_nodes = Vec::new();
    let mut study_count = 0usize;
    let mut series_count = 0usize;
    for (patient_id, studies) in patients {
        let mut study_nodes = Vec::new();
        for (study_instance_uid, series_map) in studies {
            study_count += 1;
            let mut series_nodes = Vec::new();
            for (series_instance_uid, mut series) in series_map {
                series_count += 1;
                if series.modalities.len() > 1 {
                    warnings.push(DicomGraphWarning {
                        code: "mixed_modality".to_string(),
                        message: format!(
                            "series {series_instance_uid} contains modalities {}",
                            series.modalities.into_iter().collect::<Vec<_>>().join(", ")
                        ),
                        path: None,
                    });
                }
                if series.dimensions.len() > 1 {
                    warnings.push(DicomGraphWarning {
                        code: "mixed_dimensions".to_string(),
                        message: format!(
                            "series {series_instance_uid} contains dimensions {}",
                            series.dimensions.into_iter().collect::<Vec<_>>().join(", ")
                        ),
                        path: None,
                    });
                }
                series.instances.sort_by(|left, right| {
                    left.sop_instance_uid
                        .cmp(&right.sop_instance_uid)
                        .then_with(|| left.path.cmp(&right.path))
                });
                series_nodes.push(DicomSeriesNode {
                    series_instance_uid,
                    modality: series.modality,
                    rows: series.rows,
                    columns: series.columns,
                    instances: series.instances,
                });
            }
            study_nodes.push(DicomStudyNode {
                study_instance_uid,
                series: series_nodes,
            });
        }
        patient_nodes.push(DicomPatientNode {
            patient_id,
            studies: study_nodes,
        });
    }
    warnings.sort_by(|left, right| {
        left.code
            .cmp(&right.code)
            .then_with(|| left.path.cmp(&right.path))
            .then_with(|| left.message.cmp(&right.message))
    });

    DicomGraphSummary {
        root: root.as_ref().display().to_string(),
        patients: patient_nodes.len(),
        studies: study_count,
        series: series_count,
        instances: records.len(),
        duplicate_sop_instance_uids: duplicate_count(
            records
                .iter()
                .filter_map(|record| record.sop_instance_uid.as_deref()),
        ),
        duplicate_pixel_hashes: duplicate_count(
            records
                .iter()
                .filter_map(|record| record.pixel_hash.as_deref()),
        ),
        warnings,
        patients_detail: patient_nodes,
    }
}

pub fn write_graph_outputs(
    graph: &DicomGraphSummary,
    out_path: impl AsRef<Path>,
    report_path: impl AsRef<Path>,
) -> Result<()> {
    write_graph_json(out_path.as_ref(), graph)?;
    write_graph_report(report_path.as_ref(), graph)?;
    Ok(())
}

fn write_graph_json(path: &Path, graph: &DicomGraphSummary) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|source| DicomError::io(parent, source))?;
    }
    let text = serde_json::to_string_pretty(graph).expect("DICOM graph serializes");
    fs::write(path, text).map_err(|source| DicomError::io(path, source))
}

fn write_graph_report(path: &Path, graph: &DicomGraphSummary) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|source| DicomError::io(parent, source))?;
    }
    let mut report = String::new();
    report.push_str("# DICOM Graph Report\n\n");
    report.push_str(&format!("- patients: {}\n", graph.patients));
    report.push_str(&format!("- studies: {}\n", graph.studies));
    report.push_str(&format!("- series: {}\n", graph.series));
    report.push_str(&format!("- instances: {}\n", graph.instances));
    report.push_str(&format!(
        "- duplicate SOP Instance UIDs: {}\n",
        graph.duplicate_sop_instance_uids
    ));
    report.push_str(&format!(
        "- duplicate pixel hashes: {}\n",
        graph.duplicate_pixel_hashes
    ));
    report.push_str("\n## Warnings\n\n");
    if graph.warnings.is_empty() {
        report.push_str("- none\n");
    } else {
        for warning in &graph.warnings {
            match &warning.path {
                Some(path) => {
                    report.push_str(&format!(
                        "- {}: {} ({})\n",
                        path, warning.message, warning.code
                    ));
                }
                None => report.push_str(&format!("- {} ({})\n", warning.message, warning.code)),
            }
        }
    }
    fs::write(path, report).map_err(|source| DicomError::io(path, source))
}

fn node_key(value: Option<&str>, fallback: &str) -> String {
    value
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(fallback)
        .to_string()
}

fn duplicate_values<'a>(values: impl Iterator<Item = &'a str>) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    let mut duplicates = BTreeSet::new();
    for value in values {
        if !seen.insert(value.to_string()) {
            duplicates.insert(value.to_string());
        }
    }
    duplicates
}

fn duplicate_count<'a>(values: impl Iterator<Item = &'a str>) -> usize {
    duplicate_values(values).len()
}
