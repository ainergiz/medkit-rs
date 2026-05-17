use medkit_transform::{
    BoundingBox3, Interpolation, TransformError, TransformOp, TransformPlan, Volume3D,
    VolumeGeometry,
};

#[test]
fn geometry_round_trips_rotated_coordinates_and_updates_crop_origin() {
    let geometry = VolumeGeometry::new(
        [4, 3, 2],
        [2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0],
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )
    .unwrap();

    let world = geometry.voxel_to_world([1.5, 2.0, 0.5]);
    assert_eq!(world, [4.0, 23.0, 32.0]);
    let voxel = geometry.world_to_voxel(world).unwrap();
    assert!(voxel
        .iter()
        .zip([1.5, 2.0, 0.5])
        .all(|(actual, expected)| (*actual - expected).abs() < 1e-12));

    let cropped = geometry
        .crop(BoundingBox3::new([1, 1, 0], [4, 3, 2]))
        .unwrap();

    assert_eq!(cropped.shape, [3, 2, 2]);
    assert_eq!(cropped.origin, [7.0, 22.0, 30.0]);
    assert!(cropped.approximately_eq(&cropped, 0.0));
    assert!(!cropped.approximately_eq(&geometry, 1e-12));
}

#[test]
fn geometry_resampling_preserves_singleton_axes_and_rounds_endpoint_shape() {
    let geometry = VolumeGeometry::identity([3, 1, 4], [2.0, 5.0, 3.0]).unwrap();

    let resampled = geometry.resampled_to_spacing([1.0, 2.0, 2.0]).unwrap();

    assert_eq!(resampled.shape, [5, 1, 6]);
    assert_eq!(resampled.spacing, [1.0, 2.0, 2.0]);
    assert_eq!(resampled.origin, geometry.origin);
}

#[test]
fn volume_crop_and_center_pad_crop_preserve_row_major_order() {
    let volume = Volume3D::new([4, 3, 2], (0_u16..24).collect()).unwrap();

    let cropped = volume
        .crop(BoundingBox3::new([1, 1, 0], [4, 3, 2]))
        .unwrap();
    assert_eq!(cropped.shape, [3, 2, 2]);
    assert_eq!(
        cropped.data,
        vec![5, 6, 7, 9, 10, 11, 17, 18, 19, 21, 22, 23]
    );

    let padded = Volume3D::new([2, 1, 1], vec![8_u16, 9])
        .unwrap()
        .pad_crop_center([4, 1, 1], 7)
        .unwrap();
    assert_eq!(padded.data, vec![7, 8, 9, 7]);

    let center_cropped = Volume3D::new([5, 1, 1], vec![0_u16, 1, 2, 3, 4])
        .unwrap()
        .pad_crop_center([3, 1, 1], 9)
        .unwrap();
    assert_eq!(center_cropped.data, vec![1, 2, 3]);
}

#[test]
fn geometry_center_crop_rejects_zero_target_and_offsets_origin() {
    let geometry = VolumeGeometry::identity([5, 1, 1], [2.0, 1.0, 1.0]).unwrap();

    assert_eq!(
        geometry.pad_crop_center([0, 1, 1]).unwrap_err(),
        TransformError::InvalidSize { size: [0, 1, 1] }
    );

    let cropped = geometry.pad_crop_center([3, 1, 1]).unwrap();
    assert_eq!(cropped.shape, [3, 1, 1]);
    assert_eq!(cropped.origin, [2.0, 0.0, 0.0]);
}

#[test]
fn plan_parse_reports_unknown_operations_bad_enums_and_missing_fields() {
    for input in [
        r#"
name = "missing-operations"
image_interpolation = "linear"
label_interpolation = "nearest"
"#,
        r#"
name = "bad-interpolation"
operations = []
image_interpolation = "cubic"
label_interpolation = "nearest"
"#,
        r#"
name = "unknown-op"
image_interpolation = "linear"
label_interpolation = "nearest"

[[operations]]
op = "flip"
"#,
    ] {
        assert!(matches!(
            TransformPlan::from_toml_str(input).unwrap_err(),
            TransformError::PlanParse { .. }
        ));
    }
}

#[test]
fn plan_validation_errors_surface_from_pad_crop_and_geometry_resample() {
    let image = Volume3D::new([2, 1, 1], vec![0.0, 1.0]).unwrap();
    let label = Volume3D::new([2, 1, 1], vec![0_u16, 1]).unwrap();
    let bad_pad = TransformPlan {
        name: "bad-pad".to_string(),
        operations: vec![TransformOp::PadCrop { size: [2, 0, 1] }],
        image_interpolation: Interpolation::Linear,
        label_interpolation: Interpolation::Nearest,
    };

    assert_eq!(
        bad_pad.apply_pair(image, label).unwrap_err(),
        TransformError::InvalidSize { size: [2, 0, 1] }
    );

    let image = Volume3D::new([2, 1, 1], vec![0.0, 1.0]).unwrap();
    let label = Volume3D::new([2, 1, 1], vec![0_u16, 1]).unwrap();
    let bad_resample = TransformPlan {
        name: "bad-resample".to_string(),
        operations: vec![TransformOp::Resample {
            spacing: [1.0, f64::NAN, 1.0],
        }],
        image_interpolation: Interpolation::Linear,
        label_interpolation: Interpolation::Nearest,
    };

    let error = bad_resample.apply_pair(image, label).unwrap_err();
    assert!(matches!(
        error,
        TransformError::InvalidSpacing { spacing }
            if spacing[0] == 1.0 && spacing[1].is_nan() && spacing[2] == 1.0
    ));
}

#[test]
fn default_plan_hash_lazy_graph_and_linear_label_resample_is_rejected() {
    let default_plan = TransformPlan::ct_segmentation_default();
    let graph = default_plan.lazy_graph();
    assert_eq!(graph.operations, default_plan.operations);
    assert_eq!(default_plan.plan_hash().unwrap().len(), 64);
    assert!(default_plan
        .canonical_json()
        .unwrap()
        .contains("ct-segmentation"));

    let plan = TransformPlan {
        name: "linear-label-resample".to_string(),
        operations: vec![TransformOp::Resample {
            spacing: [0.5, 1.0, 1.0],
        }],
        image_interpolation: Interpolation::Nearest,
        label_interpolation: Interpolation::Linear,
    };
    let image = Volume3D::new([2, 1, 1], vec![2.0, 6.0]).unwrap();
    let label = Volume3D::new([2, 1, 1], vec![0_u16, 4]).unwrap();

    let error = plan.apply_pair(image, label).unwrap_err();

    assert!(matches!(
        error,
        TransformError::InvalidLabelInterpolation { .. }
    ));
    assert!(error.to_string().contains("nearest-neighbor"));
}
