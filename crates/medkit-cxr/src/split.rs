use std::collections::{BTreeMap, BTreeSet};

use crate::{
    error::CxrError,
    manifest::{read_manifest, write_manifest},
    types::{CxrRecord, SplitConfig, SplitFile, SplitSummary},
    util::{collect_targets, stable_bucket, write_json},
};

#[derive(Debug, Default)]
struct SplitAssignmentResult {
    train: Vec<String>,
    val: Vec<String>,
    test: Vec<String>,
    patient_counts: BTreeMap<String, usize>,
}

#[derive(Debug, Clone)]
struct StratifiedPatientGroup {
    patient_id: String,
    indices: Vec<usize>,
    strata_counts: BTreeMap<String, usize>,
    bucket: u64,
}

impl StratifiedPatientGroup {
    fn samples(&self) -> usize {
        self.indices.len()
    }

    fn stratified_samples(&self) -> usize {
        self.strata_counts.values().sum()
    }
}

#[derive(Debug, Clone)]
struct StratifiedSplitState {
    name: &'static str,
    ratio: f64,
    samples: usize,
    patient_count: usize,
    strata_counts: BTreeMap<String, usize>,
    sample_ids: Vec<String>,
}

impl StratifiedSplitState {
    fn new(name: &'static str, ratio: f64) -> Self {
        Self {
            name,
            ratio,
            samples: 0,
            patient_count: 0,
            strata_counts: BTreeMap::new(),
            sample_ids: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
enum StratifyKey {
    Label(String),
    Field(StratifyField),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StratifyField {
    LabelSource,
    Laterality,
    Modality,
    PhotometricInterpretation,
    SourceFormat,
    ViewPosition,
}

pub fn split_cxr(config: &SplitConfig) -> Result<SplitSummary, CxrError> {
    if config.by != "patient_id" && config.by != "patient" {
        return Err(CxrError::Message(format!(
            "only patient-level CXR splits are supported, got --by {}",
            config.by
        )));
    }
    let mut records = read_manifest(&config.manifest_path)?;
    let mut by_patient: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, record) in records.iter().enumerate() {
        by_patient
            .entry(record.patient_id.clone())
            .or_default()
            .push(index);
    }
    let groups = by_patient.into_iter().collect::<Vec<_>>();
    let assignments = if config.stratify.is_empty() {
        assign_patient_splits_by_hash(&mut records, groups, config)
    } else {
        assign_patient_splits_stratified(&mut records, groups, config)?
    };

    let counts = BTreeMap::from([
        ("train".to_string(), assignments.train.len()),
        ("val".to_string(), assignments.val.len()),
        ("test".to_string(), assignments.test.len()),
    ]);
    let ratios = BTreeMap::from([
        ("train".to_string(), config.train),
        ("val".to_string(), config.val),
        ("test".to_string(), config.test),
    ]);
    let summary = SplitSummary {
        counts,
        patient_counts: assignments.patient_counts,
        by: "patient_id".to_string(),
        ratios,
        stratify: config.stratify.clone(),
        patient_overlap_count: 0,
        out_path: config.out_path.display().to_string(),
    };
    let split_file = SplitFile {
        train: assignments.train,
        val: assignments.val,
        test: assignments.test,
        split_audit: summary.clone(),
    };
    write_json(&config.out_path, &split_file)?;
    write_manifest(&config.manifest_path, &records)?;
    Ok(summary)
}

fn assign_patient_splits_by_hash(
    records: &mut [CxrRecord],
    mut groups: Vec<(String, Vec<usize>)>,
    config: &SplitConfig,
) -> SplitAssignmentResult {
    groups.sort_by_key(|(patient, _)| stable_bucket(patient, config.seed));

    let total = records.len().max(1);
    let train_target = (config.train * total as f64).round() as usize;
    let val_target = (config.val * total as f64).round() as usize;
    let mut result = SplitAssignmentResult {
        patient_counts: initial_patient_counts(),
        ..SplitAssignmentResult::default()
    };

    for (_patient, indices) in groups {
        let target_split = if result.train.len() < train_target {
            "train"
        } else if result.val.len() < val_target {
            "val"
        } else {
            "test"
        };
        assign_group_to_result(records, indices, target_split, &mut result);
    }
    result
}

fn assign_patient_splits_stratified(
    records: &mut [CxrRecord],
    groups: Vec<(String, Vec<usize>)>,
    config: &SplitConfig,
) -> Result<SplitAssignmentResult, CxrError> {
    let stratify_keys = classify_stratify_keys(records, &config.stratify)?;
    let mut patient_groups = groups
        .into_iter()
        .map(|(patient_id, indices)| {
            let strata_counts = group_strata_counts(records, &indices, &stratify_keys);
            StratifiedPatientGroup {
                bucket: stable_bucket(&patient_id, config.seed),
                patient_id,
                indices,
                strata_counts,
            }
        })
        .collect::<Vec<_>>();
    let mut global_strata_counts = BTreeMap::new();
    for group in &patient_groups {
        for (stratum, count) in &group.strata_counts {
            *global_strata_counts.entry(stratum.clone()).or_insert(0) += count;
        }
    }

    patient_groups.sort_by(|left, right| {
        group_min_stratum_count(left, &global_strata_counts)
            .cmp(&group_min_stratum_count(right, &global_strata_counts))
            .then_with(|| right.stratified_samples().cmp(&left.stratified_samples()))
            .then_with(|| right.samples().cmp(&left.samples()))
            .then_with(|| left.bucket.cmp(&right.bucket))
            .then_with(|| left.patient_id.cmp(&right.patient_id))
    });

    let total_samples = records.len().max(1);
    let mut states = vec![
        StratifiedSplitState::new("train", config.train),
        StratifiedSplitState::new("val", config.val),
        StratifiedSplitState::new("test", config.test),
    ];

    for group in patient_groups {
        let split_index =
            choose_stratified_split(&states, &group, total_samples, &global_strata_counts);
        assign_group_to_state(records, group, &mut states[split_index]);
    }

    let mut result = SplitAssignmentResult {
        patient_counts: initial_patient_counts(),
        ..SplitAssignmentResult::default()
    };
    for state in states {
        result
            .patient_counts
            .insert(state.name.to_string(), state.patient_count);
        match state.name {
            "train" => result.train = state.sample_ids,
            "val" => result.val = state.sample_ids,
            _ => result.test = state.sample_ids,
        }
    }
    Ok(result)
}

fn choose_stratified_split(
    states: &[StratifiedSplitState],
    group: &StratifiedPatientGroup,
    total_samples: usize,
    global_strata_counts: &BTreeMap<String, usize>,
) -> usize {
    let mut best_index = 0usize;
    let mut best_score = f64::INFINITY;
    for index in 0..states.len() {
        let score =
            stratified_assignment_score(states, index, group, total_samples, global_strata_counts);
        let better = score + 1.0e-12 < best_score
            || ((score - best_score).abs() <= 1.0e-12
                && stratified_split_tie_break(states, index, best_index, group, total_samples)
                    == std::cmp::Ordering::Less);
        if better {
            best_score = score;
            best_index = index;
        }
    }
    best_index
}

fn stratified_assignment_score(
    states: &[StratifiedSplitState],
    candidate_index: usize,
    group: &StratifiedPatientGroup,
    total_samples: usize,
    global_strata_counts: &BTreeMap<String, usize>,
) -> f64 {
    let mut score = 0.0;
    let total_samples = total_samples.max(1) as f64;
    for (index, state) in states.iter().enumerate() {
        let candidate_samples =
            state.samples + usize::from(index == candidate_index) * group.samples();
        let sample_target = ratio_target(state.ratio, total_samples);
        let sample_delta = (candidate_samples as f64 - sample_target) / total_samples;
        score += sample_delta * sample_delta;

        for (stratum, global_count) in global_strata_counts {
            let group_count = if index == candidate_index {
                group.strata_counts.get(stratum).copied().unwrap_or(0)
            } else {
                0
            };
            let candidate_count =
                state.strata_counts.get(stratum).copied().unwrap_or(0) + group_count;
            let global_count = (*global_count).max(1) as f64;
            let stratum_target = ratio_target(state.ratio, global_count);
            let stratum_delta = (candidate_count as f64 - stratum_target) / global_count;
            let stratum_weight = if stratum.starts_with("label:") {
                16.0
            } else {
                4.0
            };
            score += stratum_weight * stratum_delta * stratum_delta;
        }
    }
    score
}

fn stratified_split_tie_break(
    states: &[StratifiedSplitState],
    left: usize,
    right: usize,
    group: &StratifiedPatientGroup,
    total_samples: usize,
) -> std::cmp::Ordering {
    let left_fill = split_fill_ratio(&states[left], group.samples(), total_samples);
    let right_fill = split_fill_ratio(&states[right], group.samples(), total_samples);
    left_fill
        .partial_cmp(&right_fill)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| states[left].samples.cmp(&states[right].samples))
        .then_with(|| left.cmp(&right))
}

fn split_fill_ratio(
    state: &StratifiedSplitState,
    added_samples: usize,
    total_samples: usize,
) -> f64 {
    let target = ratio_target(state.ratio, total_samples.max(1) as f64).max(1.0);
    (state.samples + added_samples) as f64 / target
}

fn ratio_target(ratio: f64, total: f64) -> f64 {
    if ratio.is_finite() && ratio > 0.0 {
        ratio * total
    } else {
        0.0
    }
}

fn assign_group_to_result(
    records: &mut [CxrRecord],
    indices: Vec<usize>,
    split: &str,
    result: &mut SplitAssignmentResult,
) {
    *result.patient_counts.entry(split.to_string()).or_insert(0) += 1;
    for index in indices {
        records[index].split = Some(split.to_string());
        match split {
            "train" => result.train.push(records[index].sample_id.clone()),
            "val" => result.val.push(records[index].sample_id.clone()),
            _ => result.test.push(records[index].sample_id.clone()),
        }
    }
}

fn assign_group_to_state(
    records: &mut [CxrRecord],
    group: StratifiedPatientGroup,
    state: &mut StratifiedSplitState,
) {
    state.patient_count += 1;
    state.samples += group.samples();
    for (stratum, count) in &group.strata_counts {
        *state.strata_counts.entry(stratum.clone()).or_insert(0) += count;
    }
    for index in group.indices {
        records[index].split = Some(state.name.to_string());
        state.sample_ids.push(records[index].sample_id.clone());
    }
}

pub(crate) fn initial_patient_counts() -> BTreeMap<String, usize> {
    BTreeMap::from([
        ("train".to_string(), 0usize),
        ("val".to_string(), 0usize),
        ("test".to_string(), 0usize),
    ])
}

fn classify_stratify_keys(
    records: &[CxrRecord],
    requested: &[String],
) -> Result<Vec<StratifyKey>, CxrError> {
    let available_targets = collect_targets(records)
        .into_iter()
        .collect::<BTreeSet<_>>();
    requested
        .iter()
        .map(|key| {
            if available_targets.contains(key) {
                Ok(StratifyKey::Label(key.clone()))
            } else if let Some(field) = StratifyField::from_name(key) {
                Ok(StratifyKey::Field(field))
            } else {
                Err(CxrError::Message(format!(
                    "unknown stratify target {key:?}; use a manifest label or one of: {}",
                    StratifyField::supported_names().join(", ")
                )))
            }
        })
        .collect()
}

fn group_strata_counts(
    records: &[CxrRecord],
    indices: &[usize],
    keys: &[StratifyKey],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for index in indices {
        for stratum in record_strata(&records[*index], keys) {
            *counts.entry(stratum).or_insert(0) += 1;
        }
    }
    counts
}

fn record_strata(record: &CxrRecord, keys: &[StratifyKey]) -> Vec<String> {
    let mut strata = Vec::new();
    for key in keys {
        match key {
            StratifyKey::Label(target) => {
                if matches!(record.labels.get(target).copied().flatten(), Some(1)) {
                    strata.push(format!("label:{target}=positive"));
                }
            }
            StratifyKey::Field(field) => {
                if let Some(value) = field.value(record).and_then(normalize_stratify_value) {
                    strata.push(format!("{}={value}", field.canonical_name()));
                }
            }
        }
    }
    strata
}

pub(crate) fn normalize_stratify_value(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_ascii_lowercase())
}

fn group_min_stratum_count(
    group: &StratifiedPatientGroup,
    global_strata_counts: &BTreeMap<String, usize>,
) -> usize {
    group
        .strata_counts
        .keys()
        .filter_map(|stratum| global_strata_counts.get(stratum).copied())
        .min()
        .unwrap_or(usize::MAX)
}

impl StratifyField {
    fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "label_source" | "labelsource" => Some(Self::LabelSource),
            "laterality" => Some(Self::Laterality),
            "modality" => Some(Self::Modality),
            "photometric_interpretation" | "photometricinterpretation" => {
                Some(Self::PhotometricInterpretation)
            }
            "source_format" | "sourceformat" => Some(Self::SourceFormat),
            "view" | "view_position" | "viewposition" | "viewpositioncode" => {
                Some(Self::ViewPosition)
            }
            _ => None,
        }
    }

    fn supported_names() -> Vec<&'static str> {
        vec![
            "label_source",
            "laterality",
            "modality",
            "photometric_interpretation",
            "source_format",
            "view_position",
        ]
    }

    fn canonical_name(self) -> &'static str {
        match self {
            Self::LabelSource => "label_source",
            Self::Laterality => "laterality",
            Self::Modality => "modality",
            Self::PhotometricInterpretation => "photometric_interpretation",
            Self::SourceFormat => "source_format",
            Self::ViewPosition => "view_position",
        }
    }

    fn value(self, record: &CxrRecord) -> Option<&str> {
        match self {
            Self::LabelSource => record.label_source.as_deref(),
            Self::Laterality => record.laterality.as_deref(),
            Self::Modality => record.modality.as_deref(),
            Self::PhotometricInterpretation => record.photometric_interpretation.as_deref(),
            Self::SourceFormat => Some(record.source_format.as_str()),
            Self::ViewPosition => record.view_position.as_deref(),
        }
    }
}
