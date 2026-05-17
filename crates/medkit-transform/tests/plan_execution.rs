use medkit_transform::{Interpolation, TransformPlan, Volume3D, VolumeGeometry};

#[test]
fn transform_plan_parses_lazy_graph_and_executes_pair() {
    let plan = TransformPlan::from_toml_str(
        r#"
name = "ct-segmentation-test"
image_interpolation = "linear"
label_interpolation = "nearest"

[[operations]]
op = "resample"
spacing = [1.0, 1.0, 1.0]

[[operations]]
op = "ct_window"
min = -100.0
max = 100.0

[[operations]]
op = "min_max_normalize"

[[operations]]
op = "crop_foreground"
margin = 0

[[operations]]
op = "pad_crop"
size = [4, 4, 4]
"#,
    )
    .unwrap();

    let graph = plan.lazy_graph();
    assert_eq!(graph.image_interpolation, Interpolation::Linear);
    assert_eq!(graph.label_interpolation, Interpolation::Nearest);
    assert_eq!(
        graph.operation_names(),
        vec![
            "resample",
            "ct_window",
            "min_max_normalize",
            "crop_foreground",
            "pad_crop",
        ]
    );
    assert_eq!(plan.plan_hash().unwrap().len(), 64);

    let image = Volume3D::new([4, 4, 4], (-32..32).map(|value| value as f32).collect()).unwrap();
    let mut label_values = vec![0_u16; 64];
    label_values[1 + 4 * (1 + 4)] = 1;
    label_values[2 + 4 * (2 + 4 * 2)] = 2;
    let label = Volume3D::new([4, 4, 4], label_values).unwrap();

    let prepared = plan.apply_pair(image, label).unwrap();

    assert_eq!(prepared.image.shape, [4, 4, 4]);
    assert_eq!(prepared.label.shape, [4, 4, 4]);
    assert_eq!(prepared.crop_origin, [1, 1, 1]);
    assert_eq!(
        prepared
            .label
            .data
            .iter()
            .filter(|value| **value != 0)
            .count(),
        2
    );
    assert!(prepared.image.data.iter().all(|value| value.is_finite()));
    assert!(prepared
        .applied_operations
        .contains(&"resample".to_string()));
    assert_eq!(prepared.geometry.shape, [4, 4, 4]);
    assert_eq!(prepared.geometry.spacing, [1.0, 1.0, 1.0]);
}

#[test]
fn intensity_transforms_have_explicit_semantics() {
    let percentile_clipped = apply_intensity_ops(
        r#"
[[operations]]
op = "percentile_clip"
lower = 25.0
upper = 75.0
"#,
    );
    assert_values_close(&percentile_clipped, &[7.5, 10.0, 20.0, 22.5]);

    let min_maxed = apply_intensity_ops(
        r#"
[[operations]]
op = "min_max_normalize"
output_min = -1.0
output_max = 1.0
"#,
    );
    assert_values_close(&min_maxed, &[-1.0, -0.33333334, 0.33333334, 1.0]);

    let dataset_normalized = apply_intensity_ops(
        r#"
[[operations]]
op = "dataset_mean_std_normalize"
mean = 10.0
std = 5.0
"#,
    );
    assert_values_close(&dataset_normalized, &[-2.0, 0.0, 2.0, 4.0]);

    let z_scored = apply_intensity_ops(
        r#"
[[operations]]
op = "z_score_normalize"
"#,
    );
    assert_values_close(&z_scored, &[-1.3416407, -0.4472136, 0.4472136, 1.3416407]);
}

fn apply_intensity_ops(operations: &str) -> Vec<f32> {
    let plan = TransformPlan::from_toml_str(&format!(
        r#"
name = "intensity-test"
image_interpolation = "nearest"
label_interpolation = "nearest"

{operations}
"#
    ))
    .unwrap();
    let image = Volume3D::new([4, 1, 1], vec![0.0, 10.0, 20.0, 30.0]).unwrap();
    let label = Volume3D::new([4, 1, 1], vec![0_u16; 4]).unwrap();

    let prepared = plan.apply_pair(image, label).unwrap();
    prepared.image.data
}

fn assert_values_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (*actual - *expected).abs() < 1.0e-5,
            "expected {expected}, got {actual}"
        );
    }
}

#[test]
fn resample_uses_physical_spacing_and_interpolation_policies() {
    let plan = TransformPlan::from_toml_str(
        r#"
name = "resample-test"
image_interpolation = "linear"
label_interpolation = "nearest"

[[operations]]
op = "resample"
spacing = [1.0, 1.0, 1.0]
"#,
    )
    .unwrap();
    let image = Volume3D::new([4, 1, 1], vec![0.0, 10.0, 20.0, 30.0]).unwrap();
    let label = Volume3D::new([4, 1, 1], vec![0, 1, 0, 2]).unwrap();
    let geometry = VolumeGeometry::identity([4, 1, 1], [2.0, 1.0, 1.0]).unwrap();

    let prepared = plan
        .apply_pair_with_geometry(image, label, geometry)
        .unwrap();

    assert_eq!(prepared.image.shape, [7, 1, 1]);
    assert_eq!(
        prepared.image.data,
        vec![0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    );
    assert_eq!(prepared.label.data, vec![0, 1, 1, 0, 0, 2, 2]);
    assert_eq!(prepared.geometry.shape, [7, 1, 1]);
    assert_eq!(prepared.geometry.spacing, [1.0, 1.0, 1.0]);
}
