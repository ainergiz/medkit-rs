use medkit_transform::{Interpolation, TransformPlan, Volume3D, VolumeGeometry};

#[test]
fn transform_plan_parses_lazy_graph_and_executes_pair() {
    let plan = TransformPlan::from_toml_str(
        r#"
name = "ct-segmentation-test"
image_interpolation = "linear"
label_interpolation = "nearest"

[[operations]]
op = "ct_window"
min = -100.0
max = 100.0

[[operations]]
op = "normalize"
mean = 0.0
std = 1.0

[[operations]]
op = "crop_foreground"
margin = 0

[[operations]]
op = "pad_crop"
size = [4, 4, 4]

[[operations]]
op = "resample"
spacing = [1.0, 1.0, 1.0]
"#,
    )
    .unwrap();

    let graph = plan.lazy_graph();
    assert_eq!(graph.image_interpolation, Interpolation::Linear);
    assert_eq!(graph.label_interpolation, Interpolation::Nearest);
    assert_eq!(
        graph.operation_names(),
        vec![
            "ct_window",
            "normalize",
            "crop_foreground",
            "pad_crop",
            "resample"
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
