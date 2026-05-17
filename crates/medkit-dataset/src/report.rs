use crate::{CaseStatus, DatasetManifest};

/// Renders a human-readable validation report.
pub fn render_report(manifest: &DatasetManifest) -> String {
    let mut report = String::new();
    report.push_str(&format!("Dataset: {}\n", manifest.dataset_root));
    report.push_str(&format!("Images: {}\n", manifest.images_dir));
    report.push_str(&format!("Labels: {}\n", manifest.labels_dir));
    report.push_str(&format!("Layout: {}\n", manifest.layout.as_str()));
    report.push('\n');
    report.push_str(&format!("Cases: {}\n", manifest.summary.total_cases));
    report.push_str(&format!("Valid: {}\n", manifest.summary.valid_cases));
    report.push_str(&format!("Invalid: {}\n", manifest.summary.invalid_cases));
    report.push_str(&format!(
        "Missing images: {}\n",
        manifest.summary.missing_images
    ));
    report.push_str(&format!(
        "Missing labels: {}\n",
        manifest.summary.missing_labels
    ));
    report.push_str(&format!(
        "Geometry mismatches: {}\n",
        manifest.summary.geometry_mismatches
    ));
    report.push_str(&format!("Read errors: {}\n", manifest.summary.read_errors));

    let invalid_cases = manifest
        .cases
        .iter()
        .filter(|case| case.status == CaseStatus::Invalid)
        .collect::<Vec<_>>();
    if invalid_cases.is_empty() {
        report.push_str("\nProblems: none\n");
        return report;
    }

    report.push_str("\nProblems:\n");
    for case in invalid_cases {
        report.push_str(&format!("  {}:\n", case.case_id));
        for problem in &case.problems {
            report.push_str(&format!("    - {:?}: {}\n", problem.code, problem.message));
        }
    }
    report
}
