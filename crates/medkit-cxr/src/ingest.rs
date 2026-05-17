use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use medkit_dicom::{DicomInventoryRecord, DicomScanConfig, DicomScanSummary};
use serde::{Deserialize, Serialize};

use crate::{
    cache::{cache_cxr_with_options, validate_cache_cxr, CxrCacheOptions},
    error::CxrError,
    manifest::{index_cxr, read_manifest, validate_cxr, write_manifest},
    recipe::{read_cxr_dicom_recipe, recipe_fingerprint, CxrDicomRecipe},
    split::split_cxr,
    types::{
        CacheConfig, CacheSummary, CacheValidationSummary, CxrRecord, IndexConfig, IngestConfig,
        IngestCounts, IngestPaths, IngestSampleIssue, IngestSummary, LabelCount, NumericSummary,
        PixelSpacingSummary, SplitConfig, SplitSummary, ValidateCacheConfig, ValidateConfig,
        ValidationSummary,
    },
    util::{add_label_count, hash_file, write_json},
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct IngestResumeState {
    raw_dicom_root: String,
    recipe_path: String,
    recipe_hash: String,
    labels_path: String,
    labels_hash: String,
    cache_dir: String,
}

pub fn ingest_cxr_dicom(config: &IngestConfig) -> Result<IngestSummary, CxrError> {
    let recipe = read_cxr_dicom_recipe(&config.recipe_path)?;
    let fingerprint = recipe_fingerprint(&config.recipe_path)?;
    let paths = ingest_paths(config);

    if config.dry_run {
        let summary = dry_run_summary(config, &recipe, &fingerprint.sha256, paths);
        write_ingest_outputs(
            &summary,
            &config.report_path,
            summary_json_path(&config.workdir),
        )?;
        return Ok(summary);
    }

    fs::create_dir_all(&config.workdir)?;
    let resume_state = IngestResumeState {
        raw_dicom_root: config.raw_root.display().to_string(),
        recipe_path: config.recipe_path.display().to_string(),
        recipe_hash: fingerprint.sha256.clone(),
        labels_path: config.labels_path.display().to_string(),
        labels_hash: hash_file(&config.labels_path)?,
        cache_dir: config.cache_dir.display().to_string(),
    };
    verify_or_write_resume_state(Path::new(&paths.resume_state), &resume_state)?;

    let scan_config = DicomScanConfig {
        root: config.raw_root.clone(),
        out_path: PathBuf::from(&paths.dicom_index),
        report_path: PathBuf::from(&paths.dicom_scan_report),
    };
    let (scan_summary, scan_records) =
        medkit_dicom::scan_dicom_with_workers(&scan_config, config.workers)?;
    medkit_dicom::write_scan_outputs(
        &scan_summary,
        &scan_records,
        &scan_config.out_path,
        &scan_config.report_path,
    )?;

    let (filtered_records, skipped_samples) = filter_dicom_records(&recipe, &scan_records)?;
    if filtered_records.is_empty() {
        return Err(CxrError::Message(
            "recipe filters removed all DICOM records; no CXR manifest can be built".to_string(),
        ));
    }
    write_dicom_index_jsonl(Path::new(&paths.recipe_dicom_index), &filtered_records)?;

    let manifest_path = PathBuf::from(&paths.manifest);
    let index_summary = index_cxr(&IndexConfig {
        images_root: config.raw_root.clone(),
        dicom_index_path: Some(PathBuf::from(&paths.recipe_dicom_index)),
        metadata_path: None,
        labels_path: Some(config.labels_path.clone()),
        reports_root: None,
        out_path: manifest_path.clone(),
    })?;
    let mut records = read_manifest(&manifest_path)?;
    apply_recipe_label_targets(&recipe, &mut records)?;
    write_manifest(&manifest_path, &records)?;

    let validation = validate_cxr(&ValidateConfig {
        manifest_path: manifest_path.clone(),
        require_frontal: !recipe.dicom.views.is_empty(),
        check_patient_leakage: true,
        check_duplicates: true,
        report_path: PathBuf::from(&paths.validation_report),
    })?;

    let split = split_cxr(&SplitConfig {
        manifest_path: manifest_path.clone(),
        by: recipe.split.by.clone(),
        train: recipe.split.train,
        val: recipe.split.val,
        test: recipe.split.test,
        stratify: recipe.split.stratify.clone(),
        out_path: PathBuf::from(&paths.splits),
        seed: recipe.split.seed,
    })?;
    let records = read_manifest(&manifest_path)?;

    if config.cache_dir.exists() {
        fs::remove_dir_all(&config.cache_dir)?;
    }
    let cache = cache_cxr_with_options(
        &CacheConfig {
            manifest_path: manifest_path.clone(),
            splits_path: PathBuf::from(&paths.splits),
            plan_path: config.recipe_path.clone(),
            cache_dir: config.cache_dir.clone(),
        },
        &CxrCacheOptions {
            targets: recipe.labels.targets.clone(),
            recipe_hash: fingerprint.sha256.clone(),
            recipe_path: config.recipe_path.display().to_string(),
            label_policy: recipe.label_policy(),
            image_size_policy: recipe.image_size_policy(),
            dicom_presentation_policy: recipe.presentation_policy(),
            transfer_syntax_policy: recipe.transfer_syntax_policy(),
            split_policy: recipe.split_policy(),
        },
    )?;

    let cache_validation = validate_cache_cxr(&ValidateCacheConfig {
        cache_dir: config.cache_dir.clone(),
        split: None,
        expected_targets: Some(recipe.labels.targets.clone()),
        expected_image_shape: None,
        plan_path: Some(config.recipe_path.clone()),
        report_path: Some(PathBuf::from(&paths.cache_validation_report)),
        json_path: Some(PathBuf::from(&paths.cache_validation_json)),
    })?;

    let mut summary = build_ingest_summary(
        false,
        "ok",
        &recipe,
        &fingerprint.sha256,
        paths,
        &scan_summary,
        &scan_records,
        &records,
        &validation,
        &split,
        &cache,
        &cache_validation,
        skipped_samples,
    );
    summary.counts.manifest_records = index_summary.records;
    write_ingest_outputs(
        &summary,
        &config.report_path,
        summary_json_path(&config.workdir),
    )?;
    Ok(summary)
}

fn filter_dicom_records(
    recipe: &CxrDicomRecipe,
    records: &[DicomInventoryRecord],
) -> Result<(Vec<DicomInventoryRecord>, Vec<IngestSampleIssue>), CxrError> {
    let allowed_transfer = recipe
        .dicom
        .allow_transfer_syntaxes
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let allowed_modalities = normalized_set(&recipe.dicom.modalities);
    let allowed_views = normalized_set(&recipe.dicom.views);
    let mut filtered = Vec::new();
    let mut skipped = Vec::new();

    for record in records {
        if !allowed_transfer.is_empty() && !allowed_transfer.contains(&record.transfer_syntax_uid) {
            let issue = IngestSampleIssue {
                path: record.path.clone(),
                reason: format!(
                    "transfer syntax {} is not allowed by the recipe",
                    record.transfer_syntax_uid
                ),
                code: "unsupported_transfer_syntax".to_string(),
            };
            if recipe.dicom.unsupported_transfer_syntax == "fail" {
                return Err(CxrError::Message(format!(
                    "{}: {}",
                    issue.path, issue.reason
                )));
            }
            skipped.push(issue);
            continue;
        }
        if !allowed_modalities.is_empty() {
            let modality = record
                .modality
                .as_deref()
                .map(str::to_ascii_uppercase)
                .unwrap_or_default();
            if !allowed_modalities.contains(&modality) {
                skipped.push(IngestSampleIssue {
                    path: record.path.clone(),
                    reason: format!("modality {modality:?} is not allowed by the recipe"),
                    code: "filtered_modality".to_string(),
                });
                continue;
            }
        }
        if !allowed_views.is_empty() {
            let view = record
                .view_position
                .as_deref()
                .map(str::to_ascii_uppercase)
                .unwrap_or_default();
            if !allowed_views.contains(&view) {
                skipped.push(IngestSampleIssue {
                    path: record.path.clone(),
                    reason: if view.is_empty() {
                        "ViewPosition is missing".to_string()
                    } else {
                        format!("ViewPosition {view:?} is not allowed by the recipe")
                    },
                    code: "filtered_view_position".to_string(),
                });
                continue;
            }
        }
        filtered.push(record.clone());
    }

    Ok((filtered, skipped))
}

fn apply_recipe_label_targets(
    recipe: &CxrDicomRecipe,
    records: &mut [CxrRecord],
) -> Result<(), CxrError> {
    let target_set = recipe
        .labels
        .targets
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    for record in records {
        for target in &target_set {
            record.labels.entry(target.clone()).or_insert(None);
        }
        record
            .labels
            .retain(|target, _| target_set.contains(target));
        if recipe.labels.missing == "fail" && record.labels.values().any(Option::is_none) {
            return Err(CxrError::Message(format!(
                "record {} is missing at least one required label",
                record.sample_id
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_ingest_summary(
    dry_run: bool,
    status: &str,
    recipe: &CxrDicomRecipe,
    recipe_hash: &str,
    paths: IngestPaths,
    scan_summary: &DicomScanSummary,
    scan_records: &[DicomInventoryRecord],
    manifest_records: &[CxrRecord],
    validation: &ValidationSummary,
    split: &SplitSummary,
    cache: &CacheSummary,
    cache_validation: &CacheValidationSummary,
    skipped_samples: Vec<IngestSampleIssue>,
) -> IngestSummary {
    let mut missing_identifier_counts = BTreeMap::new();
    for record in scan_records {
        count_missing(
            &mut missing_identifier_counts,
            "patient_id",
            record.patient_id.as_ref(),
        );
        count_missing(
            &mut missing_identifier_counts,
            "study_instance_uid",
            record.study_instance_uid.as_ref(),
        );
        count_missing(
            &mut missing_identifier_counts,
            "series_instance_uid",
            record.series_instance_uid.as_ref(),
        );
        count_missing(
            &mut missing_identifier_counts,
            "sop_instance_uid",
            record.sop_instance_uid.as_ref(),
        );
    }

    IngestSummary {
        dry_run,
        status: status.to_string(),
        recipe_name: recipe.name.clone(),
        recipe_hash: recipe_hash.to_string(),
        paths,
        planned_actions: planned_actions(),
        validation_rules: validation_rules(recipe),
        counts: IngestCounts {
            patients: manifest_records
                .iter()
                .map(|record| &record.patient_id)
                .collect::<BTreeSet<_>>()
                .len(),
            studies: manifest_records
                .iter()
                .map(|record| &record.study_id)
                .collect::<BTreeSet<_>>()
                .len(),
            series: manifest_records
                .iter()
                .filter_map(|record| record.series_instance_uid.as_ref())
                .collect::<BTreeSet<_>>()
                .len(),
            images: manifest_records.len(),
            dicom_records_scanned: scan_summary.records,
            manifest_records: manifest_records.len(),
            unsupported_or_skipped_images: skipped_samples.len(),
        },
        modality_distribution: option_counts(
            scan_records.iter().map(|record| record.modality.as_deref()),
        ),
        view_position_distribution: option_counts(
            scan_records
                .iter()
                .map(|record| record.view_position.as_deref()),
        ),
        transfer_syntax_distribution: string_counts(
            scan_records
                .iter()
                .map(|record| record.transfer_syntax_uid.as_str()),
        ),
        rows_summary: numeric_summary(
            scan_records
                .iter()
                .filter_map(|record| record.rows.map(f64::from)),
        ),
        columns_summary: numeric_summary(
            scan_records
                .iter()
                .filter_map(|record| record.columns.map(f64::from)),
        ),
        pixel_spacing_summary: pixel_spacing_summary(scan_records),
        missing_identifier_counts,
        missing_label_counts: missing_label_counts(manifest_records, &recipe.labels.targets),
        label_distribution: label_distribution(manifest_records, &recipe.labels.targets),
        label_distribution_by_split: label_distribution_by_split(
            manifest_records,
            &recipe.labels.targets,
        ),
        patient_counts_by_split: split.patient_counts.clone(),
        patient_overlap_count: validation.patient_overlap_count,
        skipped_samples,
        failed_preprocessing_samples: cache.failed_samples.clone(),
        scan_error_counts: scan_error_counts(scan_summary),
        warning_counts: warning_counts(scan_records),
        duplicate_sop_instance_uid_count: scan_summary.duplicate_sop_instance_uids,
        duplicate_pixel_hash_count: scan_summary.duplicate_pixel_hashes,
        cache_transform_fingerprint: cache.transform_fingerprint.clone(),
        cache_validation_status: cache_validation.status.clone(),
    }
}

fn dry_run_summary(
    config: &IngestConfig,
    recipe: &CxrDicomRecipe,
    recipe_hash: &str,
    paths: IngestPaths,
) -> IngestSummary {
    let empty_scan = DicomScanSummary {
        root: config.raw_root.display().to_string(),
        records: 0,
        errors: Vec::new(),
        warnings: 0,
        duplicate_sop_instance_uids: 0,
        duplicate_pixel_hashes: 0,
        out_path: paths.dicom_index.clone(),
        report_path: paths.dicom_scan_report.clone(),
    };
    let empty_validation = ValidationSummary {
        records: 0,
        readable_images: 0,
        unreadable_images: 0,
        filtered_non_frontal: 0,
        patient_overlap_count: 0,
        duplicate_hash_overlap_count: 0,
        split_counts: BTreeMap::new(),
        target_counts: BTreeMap::new(),
        report_path: paths.validation_report.clone(),
    };
    let empty_split = SplitSummary {
        counts: BTreeMap::new(),
        patient_counts: BTreeMap::new(),
        by: recipe.split_policy().by,
        ratios: BTreeMap::new(),
        stratify: recipe.split.stratify.clone(),
        seed: recipe.split.seed,
        patient_overlap_count: 0,
        out_path: paths.splits.clone(),
    };
    let empty_cache = CacheSummary {
        cache_schema_version: crate::types::CXR_CACHE_SCHEMA_VERSION,
        report_schema_version: crate::types::CXR_REPORT_SCHEMA_VERSION,
        cache_dir: config.cache_dir.display().to_string(),
        image_size: recipe.image_size(),
        channels: 1,
        dtype: "float32".to_string(),
        targets: recipe.labels.targets.clone(),
        label_policy: recipe.label_policy(),
        normalization: crate::types::Normalization {
            mean: 0.0,
            std: 1.0,
        },
        transform_plan_hash: recipe_hash.to_string(),
        transform_fingerprint: recipe_hash.to_string(),
        recipe_hash: recipe_hash.to_string(),
        recipe_path: config.recipe_path.display().to_string(),
        source_manifest_checksum: String::new(),
        split_names: Vec::new(),
        image_size_policy: recipe.image_size_policy(),
        dicom_presentation_policy: recipe.presentation_policy(),
        transfer_syntax_policy: recipe.transfer_syntax_policy(),
        split_policy: recipe.split_policy(),
        splits: BTreeMap::new(),
        failed_samples: Vec::new(),
        cache_size_bytes: 0,
    };
    let empty_cache_validation = CacheValidationSummary {
        cache_dir: config.cache_dir.display().to_string(),
        cache_schema_version: crate::types::CXR_CACHE_SCHEMA_VERSION,
        expected_cache_schema_version: crate::types::CXR_CACHE_SCHEMA_VERSION,
        report_schema_version: crate::types::CXR_REPORT_SCHEMA_VERSION,
        status: "planned".to_string(),
        errors: Vec::new(),
        warnings: Vec::new(),
        targets: recipe.labels.targets.clone(),
        label_policy: recipe.label_policy(),
        split_names: Vec::new(),
        checked_splits: Vec::new(),
        image_size_policy: recipe.image_size_policy(),
        transform_fingerprint: recipe_hash.to_string(),
        recipe_hash: recipe_hash.to_string(),
        source_manifest_checksum: String::new(),
        cache_size_bytes: 0,
    };
    build_ingest_summary(
        true,
        "planned",
        recipe,
        recipe_hash,
        paths,
        &empty_scan,
        &[],
        &[],
        &empty_validation,
        &empty_split,
        &empty_cache,
        &empty_cache_validation,
        Vec::new(),
    )
}

fn write_ingest_outputs(
    summary: &IngestSummary,
    report_path: &Path,
    summary_json_path: PathBuf,
) -> Result<(), CxrError> {
    if let Some(parent) = summary_json_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    write_json(&summary_json_path, summary)?;
    if let Some(parent) = report_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(report_path, render_ingest_report(summary))?;
    Ok(())
}

fn render_ingest_report(summary: &IngestSummary) -> String {
    let mut report = String::new();
    report.push_str("# CXR DICOM Ingest Report\n\n");
    report.push_str(&format!("- status: {}\n", summary.status));
    report.push_str(&format!("- dry run: {}\n", summary.dry_run));
    report.push_str(&format!("- recipe: {}\n", summary.paths.recipe));
    report.push_str(&format!("- recipe hash: {}\n", summary.recipe_hash));
    report.push_str(&format!(
        "- cache transform fingerprint: {}\n",
        summary.cache_transform_fingerprint
    ));
    report.push_str(&format!(
        "- cache validation status: {}\n",
        summary.cache_validation_status
    ));
    report.push_str("\n## Reproduce Paths\n\n");
    for (label, path) in [
        ("raw DICOM root", &summary.paths.raw_dicom_root),
        ("labels", &summary.paths.labels),
        ("DICOM index", &summary.paths.dicom_index),
        ("recipe DICOM index", &summary.paths.recipe_dicom_index),
        ("manifest", &summary.paths.manifest),
        ("splits", &summary.paths.splits),
        ("cache", &summary.paths.cache_dir),
        ("resume state", &summary.paths.resume_state),
        (
            "cache validation JSON",
            &summary.paths.cache_validation_json,
        ),
    ] {
        report.push_str(&format!("- {label}: {path}\n"));
    }
    report.push_str("\n## Counts\n\n");
    report.push_str(&format!("- patients: {}\n", summary.counts.patients));
    report.push_str(&format!("- studies: {}\n", summary.counts.studies));
    report.push_str(&format!("- series: {}\n", summary.counts.series));
    report.push_str(&format!("- images: {}\n", summary.counts.images));
    report.push_str(&format!(
        "- scanned DICOM records: {}\n",
        summary.counts.dicom_records_scanned
    ));
    report.push_str(&format!(
        "- unsupported or skipped images: {}\n",
        summary.counts.unsupported_or_skipped_images
    ));
    report.push_str(&format!(
        "- duplicate SOP Instance UIDs: {}\n",
        summary.duplicate_sop_instance_uid_count
    ));
    report.push_str(&format!(
        "- duplicate pixel hashes: {}\n",
        summary.duplicate_pixel_hash_count
    ));
    push_counts(
        &mut report,
        "Modality Distribution",
        &summary.modality_distribution,
    );
    push_counts(
        &mut report,
        "View Position Distribution",
        &summary.view_position_distribution,
    );
    push_counts(
        &mut report,
        "Transfer Syntax Distribution",
        &summary.transfer_syntax_distribution,
    );
    push_counts(
        &mut report,
        "Missing Identifiers",
        &summary.missing_identifier_counts,
    );
    push_counts(&mut report, "Missing Labels", &summary.missing_label_counts);
    report.push_str("\n## Labels By Split\n\n");
    for (split, labels) in &summary.label_distribution_by_split {
        report.push_str(&format!("### {split}\n\n"));
        for (target, count) in labels {
            report.push_str(&format!(
                "- {target}: positive {}, negative {}, uncertain {}, missing {}\n",
                count.positive, count.negative, count.uncertain, count.missing
            ));
        }
    }
    report.push_str("\n## Skipped And Failed Samples\n\n");
    if summary.skipped_samples.is_empty() {
        report.push_str("- skipped: none\n");
    } else {
        for issue in &summary.skipped_samples {
            report.push_str(&format!(
                "- skipped {}: {} ({})\n",
                issue.path, issue.reason, issue.code
            ));
        }
    }
    if summary.failed_preprocessing_samples.is_empty() {
        report.push_str("- failed preprocessing: none\n");
    } else {
        for failure in &summary.failed_preprocessing_samples {
            report.push_str(&format!("- failed preprocessing: {failure}\n"));
        }
    }
    push_counts(&mut report, "DICOM Warning Counts", &summary.warning_counts);
    push_counts(
        &mut report,
        "DICOM Parse Error Counts",
        &summary.scan_error_counts,
    );
    report
}

fn push_counts(report: &mut String, title: &str, counts: &BTreeMap<String, usize>) {
    report.push_str(&format!("\n## {title}\n\n"));
    if counts.is_empty() {
        report.push_str("- none\n");
    } else {
        for (key, count) in counts {
            report.push_str(&format!("- {key}: {count}\n"));
        }
    }
}

fn ingest_paths(config: &IngestConfig) -> IngestPaths {
    let workdir = &config.workdir;
    IngestPaths {
        recipe: config.recipe_path.display().to_string(),
        raw_dicom_root: config.raw_root.display().to_string(),
        labels: config.labels_path.display().to_string(),
        workdir: workdir.display().to_string(),
        dicom_index: workdir.join("dicom-index.jsonl").display().to_string(),
        dicom_scan_report: workdir.join("dicom-scan-report.md").display().to_string(),
        recipe_dicom_index: workdir
            .join("recipe-dicom-index.jsonl")
            .display()
            .to_string(),
        manifest: workdir.join("cxr-manifest.jsonl").display().to_string(),
        validation_report: workdir.join("cxr-validation.md").display().to_string(),
        splits: workdir.join("cxr-splits.json").display().to_string(),
        cache_dir: config.cache_dir.display().to_string(),
        cache_validation_report: workdir.join("cache-validation.md").display().to_string(),
        cache_validation_json: workdir.join("cache-validation.json").display().to_string(),
        ingest_report: config.report_path.display().to_string(),
        ingest_summary_json: summary_json_path(workdir).display().to_string(),
        resume_state: resume_state_path(workdir).display().to_string(),
    }
}

fn summary_json_path(workdir: &Path) -> PathBuf {
    workdir.join("ingestion-summary.json")
}

fn resume_state_path(workdir: &Path) -> PathBuf {
    workdir.join("ingest-run-state.json")
}

fn verify_or_write_resume_state(path: &Path, state: &IngestResumeState) -> Result<(), CxrError> {
    if path.exists() {
        let existing: IngestResumeState = serde_json::from_str(&fs::read_to_string(path)?)?;
        if &existing != state {
            return Err(CxrError::Message(format!(
                "workdir resume state does not match current ingest inputs: {}; use a new workdir or remove the existing one",
                path.display()
            )));
        }
    }
    write_json(path, state)
}

fn planned_actions() -> Vec<String> {
    vec![
        "medkit dicom scan".to_string(),
        "medkit cxr manifest --dicom-index".to_string(),
        "medkit cxr validate".to_string(),
        "medkit cxr split".to_string(),
        "medkit cxr cache".to_string(),
        "medkit cxr validate-cache".to_string(),
    ]
}

fn validation_rules(recipe: &CxrDicomRecipe) -> Vec<String> {
    let allowed_transfer = recipe.dicom.allow_transfer_syntaxes.join(", ");
    let unsupported_policy = recipe.dicom.unsupported_transfer_syntax.clone();
    let split_rule = format!(
        "split: by={} train={} val={} test={} seed={}",
        recipe.split.by, recipe.split.train, recipe.split.val, recipe.split.test, recipe.split.seed
    );
    vec![
        format!("modalities: {}", recipe.dicom.modalities.join(", ")),
        format!("views: {}", recipe.dicom.views.join(", ")),
        ["allowed transfer syntaxes: ", &allowed_transfer].concat(),
        ["unsupported transfer syntax policy: ", &unsupported_policy].concat(),
        format!("targets: {}", recipe.labels.targets.join(", ")),
        split_rule,
    ]
}

fn write_dicom_index_jsonl(path: &Path, records: &[DicomInventoryRecord]) -> Result<(), CxrError> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let mut body = String::new();
    for record in records {
        body.push_str(&serde_json::to_string(record)?);
        body.push('\n');
    }
    fs::write(path, body)?;
    Ok(())
}

fn normalized_set(values: &[String]) -> BTreeSet<String> {
    values
        .iter()
        .map(|value| value.to_ascii_uppercase())
        .collect()
}

fn option_counts<'a>(values: impl Iterator<Item = Option<&'a str>>) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for value in values {
        *counts
            .entry(value.unwrap_or("missing").to_ascii_uppercase())
            .or_insert(0) += 1;
    }
    counts
}

fn string_counts<'a>(values: impl Iterator<Item = &'a str>) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for value in values {
        *counts.entry(value.to_string()).or_insert(0) += 1;
    }
    counts
}

fn numeric_summary(values: impl Iterator<Item = f64>) -> NumericSummary {
    let mut count = 0usize;
    let mut sum = 0.0f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for value in values {
        count += 1;
        sum += value;
        min = min.min(value);
        max = max.max(value);
    }
    if count == 0 {
        return NumericSummary::default();
    }
    NumericSummary {
        count,
        min: Some(min),
        max: Some(max),
        mean: Some(sum / count as f64),
    }
}

fn pixel_spacing_summary(records: &[DicomInventoryRecord]) -> PixelSpacingSummary {
    let mut rows = Vec::new();
    let mut columns = Vec::new();
    for record in records {
        let spacing = record.pixel_spacing.or(record.imager_pixel_spacing);
        if let Some([row, column]) = spacing {
            rows.push(row as f64);
            columns.push(column as f64);
        }
    }
    PixelSpacingSummary {
        row_spacing: numeric_summary(rows.into_iter()),
        column_spacing: numeric_summary(columns.into_iter()),
    }
}

fn count_missing(counts: &mut BTreeMap<String, usize>, key: &str, value: Option<&String>) {
    if value.map(|value| value.trim().is_empty()).unwrap_or(true) {
        *counts.entry(key.to_string()).or_insert(0) += 1;
    }
}

fn missing_label_counts(records: &[CxrRecord], targets: &[String]) -> BTreeMap<String, usize> {
    let mut counts = targets
        .iter()
        .map(|target| (target.clone(), 0usize))
        .collect::<BTreeMap<_, _>>();
    for record in records {
        for target in targets {
            if record.labels.get(target).copied().flatten().is_none() {
                *counts.entry(target.clone()).or_insert(0) += 1;
            }
        }
    }
    counts
}

fn label_distribution(records: &[CxrRecord], targets: &[String]) -> BTreeMap<String, LabelCount> {
    let mut counts = targets
        .iter()
        .map(|target| (target.clone(), LabelCount::default()))
        .collect::<BTreeMap<_, _>>();
    for record in records {
        for target in targets {
            add_label_count(
                counts.entry(target.clone()).or_default(),
                record.labels.get(target).copied().flatten(),
            );
        }
    }
    counts
}

fn label_distribution_by_split(
    records: &[CxrRecord],
    targets: &[String],
) -> BTreeMap<String, BTreeMap<String, LabelCount>> {
    let mut by_split = BTreeMap::new();
    for split in ["train", "val", "test"] {
        let split_records = records
            .iter()
            .filter(|record| record.split.as_deref() == Some(split))
            .cloned()
            .collect::<Vec<_>>();
        by_split.insert(
            split.to_string(),
            label_distribution(&split_records, targets),
        );
    }
    by_split
}

fn scan_error_counts(summary: &DicomScanSummary) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for error in &summary.errors {
        let code = if error.message.contains("unsupported transfer syntax") {
            "unsupported_transfer_syntax"
        } else {
            "parse_failure"
        };
        *counts.entry(code.to_string()).or_insert(0) += 1;
    }
    counts
}

fn warning_counts(records: &[DicomInventoryRecord]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for record in records {
        for warning in &record.warnings {
            *counts.entry(warning.code.clone()).or_insert(0) += 1;
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use medkit_dicom::{DicomScanError, DicomWarning};

    use crate::recipe::{
        CxrDicomRecipe, RecipeDicomSection, RecipeImageSection, RecipeLabelsSection,
        RecipePresentationSection, RecipeSplitSection,
    };

    use super::*;

    #[test]
    fn ingest_private_helpers_cover_resume_filters_reports_and_counts() {
        let root = unique_test_dir();
        fs::create_dir_all(&root).unwrap();
        let state_path = root.join("nested").join("state.json");
        fs::create_dir_all(state_path.parent().unwrap()).unwrap();
        let state = IngestResumeState {
            raw_dicom_root: "raw".to_string(),
            recipe_path: "recipe.toml".to_string(),
            recipe_hash: "recipe-hash".to_string(),
            labels_path: "labels.csv".to_string(),
            labels_hash: "labels-hash".to_string(),
            cache_dir: "cache".to_string(),
        };
        verify_or_write_resume_state(&state_path, &state).unwrap();
        verify_or_write_resume_state(&state_path, &state).unwrap();
        let mut changed_state = state.clone();
        changed_state.labels_hash = "changed".to_string();
        assert!(verify_or_write_resume_state(&state_path, &changed_state)
            .unwrap_err()
            .to_string()
            .contains("resume state"));

        let recipe = recipe();
        let records = vec![
            dicom_record("ok", Some("CR"), Some("PA")),
            dicom_record("mr", Some("MR"), Some("PA")),
            dicom_record("lat", Some("CR"), Some("LAT")),
        ];
        let (filtered, skipped) = filter_dicom_records(&recipe, &records).unwrap();
        assert_eq!(filtered.len(), 1);
        assert!(skipped
            .iter()
            .any(|issue| issue.code == "filtered_modality"));
        assert!(skipped
            .iter()
            .any(|issue| issue.reason.contains("LAT") && issue.code == "filtered_view_position"));

        let index_path = root.join("index").join("dicom.jsonl");
        write_dicom_index_jsonl(&index_path, &filtered).unwrap();
        assert!(fs::read_to_string(index_path).unwrap().contains("\"path\""));

        let mut fail_labels = recipe.clone();
        fail_labels.labels.missing = "fail".to_string();
        let mut manifest_records = vec![cxr_record("train", None)];
        assert!(
            apply_recipe_label_targets(&fail_labels, &mut manifest_records)
                .unwrap_err()
                .to_string()
                .contains("missing at least one required label")
        );

        let mut complete_manifest = vec![
            cxr_record("train", Some(1)),
            cxr_record("val", Some(0)),
            cxr_record("test", None),
        ];
        apply_recipe_label_targets(&recipe, &mut complete_manifest).unwrap();
        let split_labels =
            label_distribution_by_split(&complete_manifest, &["Finding".to_string()]);
        assert_eq!(split_labels["train"]["Finding"].positive, 1);
        assert_eq!(
            missing_label_counts(&complete_manifest, &["Finding".to_string()])["Finding"],
            1
        );

        assert_eq!(numeric_summary(std::iter::empty()).count, 0);
        assert_eq!(numeric_summary([1.0, 3.0].into_iter()).mean, Some(2.0));
        let spacing = pixel_spacing_summary(&records);
        assert_eq!(spacing.row_spacing.count, 3);
        let mut missing = BTreeMap::new();
        count_missing(&mut missing, "patient_id", None);
        assert_eq!(missing["patient_id"], 1);

        let scan_summary = DicomScanSummary {
            root: "raw".to_string(),
            records: 3,
            errors: vec![
                DicomScanError {
                    path: "bad-transfer.dcm".to_string(),
                    message: "unsupported transfer syntax 1.2.3".to_string(),
                },
                DicomScanError {
                    path: "parse.dcm".to_string(),
                    message: "truncated header".to_string(),
                },
            ],
            warnings: 1,
            duplicate_sop_instance_uids: 0,
            duplicate_pixel_hashes: 0,
            out_path: "dicom-index.jsonl".to_string(),
            report_path: "dicom.md".to_string(),
        };
        let errors = scan_error_counts(&scan_summary);
        assert_eq!(errors["unsupported_transfer_syntax"], 1);
        assert_eq!(errors["parse_failure"], 1);

        let mut summary = dry_run_summary(
            &IngestConfig {
                raw_root: PathBuf::from("raw"),
                recipe_path: PathBuf::from("recipe.toml"),
                labels_path: PathBuf::from("labels.csv"),
                cache_dir: PathBuf::from("cache"),
                workdir: root.join("work"),
                report_path: root.join("report.md"),
                dry_run: true,
                workers: 1,
            },
            &recipe,
            "recipe-hash",
            ingest_paths(&IngestConfig {
                raw_root: PathBuf::from("raw"),
                recipe_path: PathBuf::from("recipe.toml"),
                labels_path: PathBuf::from("labels.csv"),
                cache_dir: PathBuf::from("cache"),
                workdir: root.join("work"),
                report_path: root.join("report.md"),
                dry_run: true,
                workers: 1,
            }),
        );
        summary.failed_preprocessing_samples = vec!["failed-sample".to_string()];
        summary.scan_error_counts = errors;
        summary.warning_counts = warning_counts(&[dicom_record("warned", Some("CR"), Some("PA"))]);
        let report = render_ingest_report(&summary);
        assert!(report.contains("failed preprocessing: failed-sample"));
        assert!(report.contains("unsupported_transfer_syntax: 1"));
        let rules = validation_rules(&recipe);
        assert_eq!(rules[2], "allowed transfer syntaxes: 1.2.840.10008.1.2.1");
        assert_eq!(rules[3], "unsupported transfer syntax policy: warn");
        assert!(rules
            .iter()
            .any(|rule| rule.contains("split: by=patient_id")));
    }

    fn recipe() -> CxrDicomRecipe {
        CxrDicomRecipe {
            name: "fixture".to_string(),
            dicom: RecipeDicomSection {
                modalities: vec!["CR".to_string()],
                views: vec!["PA".to_string()],
                require_single_frame: true,
                allow_transfer_syntaxes: vec![medkit_dicom::EXPLICIT_VR_LITTLE_ENDIAN.to_string()],
                unsupported_transfer_syntax: "warn".to_string(),
            },
            presentation: RecipePresentationSection::default(),
            image: RecipeImageSection {
                size: [4, 4],
                ..RecipeImageSection::default()
            },
            labels: RecipeLabelsSection {
                targets: vec!["Finding".to_string()],
                ..RecipeLabelsSection::default()
            },
            split: RecipeSplitSection {
                train: 1.0,
                val: 0.0,
                test: 0.0,
                ..RecipeSplitSection::default()
            },
        }
    }

    fn dicom_record(
        suffix: &str,
        modality: Option<&str>,
        view_position: Option<&str>,
    ) -> DicomInventoryRecord {
        DicomInventoryRecord {
            path: format!("{suffix}.dcm"),
            sha256: format!("sha-{suffix}"),
            patient_id: Some(format!("patient-{suffix}")),
            study_instance_uid: Some(format!("study-{suffix}")),
            series_instance_uid: Some(format!("series-{suffix}")),
            sop_instance_uid: Some(format!("sop-{suffix}")),
            modality: modality.map(str::to_string),
            body_part_examined: None,
            view_position: view_position.map(str::to_string),
            laterality: None,
            rows: Some(4),
            columns: Some(4),
            samples_per_pixel: Some(1),
            bits_allocated: Some(8),
            bits_stored: Some(8),
            high_bit: Some(7),
            pixel_representation: Some("unsigned".to_string()),
            photometric_interpretation: Some("MONOCHROME2".to_string()),
            transfer_syntax_uid: medkit_dicom::EXPLICIT_VR_LITTLE_ENDIAN.to_string(),
            pixel_spacing: Some([0.5, 0.6]),
            imager_pixel_spacing: None,
            rescale_intercept: None,
            rescale_slope: None,
            window_center: None,
            window_width: None,
            pixel_hash: Some(format!("pixel-{suffix}")),
            decoder_backend: Some("backend".to_string()),
            decoder_version: Some("version".to_string()),
            warnings: vec![DicomWarning::new("fixture_warning", "warning")],
        }
    }

    fn cxr_record(split: &str, label: Option<i8>) -> CxrRecord {
        let mut labels = BTreeMap::new();
        if let Some(label) = label {
            labels.insert("Finding".to_string(), Some(label));
        }
        CxrRecord {
            sample_id: format!("sample-{split}"),
            patient_id: format!("patient-{split}"),
            study_id: format!("study-{split}"),
            image_id: format!("image-{split}"),
            image_path: format!("{split}.png"),
            source_format: "png".to_string(),
            modality: Some("CR".to_string()),
            view_position: Some("PA".to_string()),
            laterality: None,
            width: Some(4),
            height: Some(4),
            photometric_interpretation: Some("MONOCHROME2".to_string()),
            series_instance_uid: Some(format!("series-{split}")),
            sop_instance_uid: Some(format!("sop-{split}")),
            transfer_syntax_uid: Some(medkit_dicom::EXPLICIT_VR_LITTLE_ENDIAN.to_string()),
            pixel_hash: Some(format!("pixel-{split}")),
            labels,
            label_source: Some("fixture".to_string()),
            report_path: None,
            split: Some(split.to_string()),
            sha256: Some(format!("sha-{split}")),
        }
    }

    fn unique_test_dir() -> PathBuf {
        std::env::temp_dir().join(format!(
            "medkit-cxr-ingest-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }
}
