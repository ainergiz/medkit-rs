use std::error::Error;

use medkit_transform::{
    Interpolation, TransformError, TransformOp, TransformPlan, Volume3D, VolumeGeometry,
};

#[test]
fn transform_errors_have_stable_messages_and_no_sources() {
    let errors = [
        (
            TransformError::PlanParse {
                message: "bad toml".to_string(),
            },
            "failed to parse transform plan: bad toml",
        ),
        (
            TransformError::PlanSerialize {
                message: "bad json".to_string(),
            },
            "failed to serialize transform plan: bad json",
        ),
        (
            TransformError::InvalidVolume {
                shape: [2, 2, 2],
                len: 7,
            },
            "invalid volume shape [2, 2, 2] for 7 values",
        ),
        (
            TransformError::InvalidSize { size: [1, 0, 1] },
            "invalid 3D size [1, 0, 1]",
        ),
        (
            TransformError::ShapeMismatch {
                image: [2, 1, 1],
                label: [1, 1, 1],
            },
            "image shape [2, 1, 1] does not match label shape [1, 1, 1]",
        ),
        (
            TransformError::GeometryShapeMismatch {
                volume: [2, 1, 1],
                geometry: [3, 1, 1],
            },
            "volume shape [2, 1, 1] does not match geometry shape [3, 1, 1]",
        ),
        (
            TransformError::InvalidSpacing {
                spacing: [1.0, 0.0, 1.0],
            },
            "invalid spacing [1.0, 0.0, 1.0]",
        ),
        (
            TransformError::InvalidOrigin {
                origin: [f64::NAN, 0.0, 0.0],
            },
            "invalid origin [NaN, 0.0, 0.0]",
        ),
        (
            TransformError::InvalidDirection { determinant: 0.0 },
            "invalid direction matrix with determinant 0",
        ),
        (
            TransformError::InvalidLabelInterpolation {
                reason: "labels require nearest".to_string(),
            },
            "invalid label interpolation: labels require nearest",
        ),
        (
            TransformError::InvalidIntensityTransform {
                reason: "bad intensity params".to_string(),
            },
            "invalid intensity transform: bad intensity params",
        ),
    ];

    for (error, expected) in errors {
        assert_eq!(error.to_string(), expected);
        assert!(error.source().is_none());
    }
}

#[test]
fn volume_and_geometry_constructors_return_specific_errors() {
    assert_eq!(
        Volume3D::<u8>::new([0, 1, 1], Vec::new()).unwrap_err(),
        TransformError::InvalidSize { size: [0, 1, 1] }
    );
    assert_eq!(
        Volume3D::new([2, 1, 1], vec![1_u8]).unwrap_err(),
        TransformError::InvalidVolume {
            shape: [2, 1, 1],
            len: 1,
        }
    );
    assert_eq!(
        Volume3D::filled([1, 0, 1], 0_u8).unwrap_err(),
        TransformError::InvalidSize { size: [1, 0, 1] }
    );
    assert_eq!(
        VolumeGeometry::identity([1, 1, 1], [1.0, 0.0, 1.0]).unwrap_err(),
        TransformError::InvalidSpacing {
            spacing: [1.0, 0.0, 1.0],
        }
    );
    assert_eq!(
        VolumeGeometry::identity([0, 1, 1], [1.0, 1.0, 1.0]).unwrap_err(),
        TransformError::InvalidSize { size: [0, 1, 1] }
    );
    let invalid_origin = VolumeGeometry::new(
        [1, 1, 1],
        [1.0, 1.0, 1.0],
        [f64::NAN, 0.0, 0.0],
        identity_direction(),
    )
    .unwrap_err();
    assert!(matches!(
        invalid_origin,
        TransformError::InvalidOrigin { origin } if origin[0].is_nan() && origin[1] == 0.0 && origin[2] == 0.0
    ));
    assert_eq!(
        VolumeGeometry::new(
            [1, 1, 1],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        )
        .unwrap_err(),
        TransformError::InvalidDirection { determinant: 0.0 }
    );
    let mut mutated_geometry = VolumeGeometry::identity([1, 1, 1], [1.0, 1.0, 1.0]).unwrap();
    mutated_geometry.direction = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    assert_eq!(
        mutated_geometry
            .world_to_voxel([0.0, 0.0, 0.0])
            .unwrap_err(),
        TransformError::InvalidDirection { determinant: 0.0 }
    );

    let non_finite_direction = VolumeGeometry::new(
        [1, 1, 1],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [[f64::NAN, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    .unwrap_err();
    assert!(matches!(
        non_finite_direction,
        TransformError::InvalidDirection { determinant } if determinant.is_nan()
    ));
}

#[test]
fn apply_pair_rejects_volume_and_geometry_shape_mismatches() {
    let plan = empty_plan();
    let image = Volume3D::new([2, 1, 1], vec![0.0, 1.0]).unwrap();
    let label = Volume3D::new([1, 1, 1], vec![0_u16]).unwrap();

    assert_eq!(
        plan.apply_pair(image, label).unwrap_err(),
        TransformError::ShapeMismatch {
            image: [2, 1, 1],
            label: [1, 1, 1],
        }
    );

    let image = Volume3D::new([2, 1, 1], vec![0.0, 1.0]).unwrap();
    let label = Volume3D::new([2, 1, 1], vec![0_u16, 0]).unwrap();
    let geometry = VolumeGeometry::identity([3, 1, 1], [1.0, 1.0, 1.0]).unwrap();

    assert_eq!(
        plan.apply_pair_with_geometry(image, label, geometry)
            .unwrap_err(),
        TransformError::GeometryShapeMismatch {
            volume: [2, 1, 1],
            geometry: [3, 1, 1],
        }
    );
}

#[test]
fn apply_pair_with_geometry_rejects_label_shape_mismatch() {
    let plan = empty_plan();
    let image = Volume3D::new([2, 1, 1], vec![0.0, 1.0]).unwrap();
    let label = Volume3D::new([1, 1, 1], vec![0_u16]).unwrap();
    let geometry = VolumeGeometry::identity([2, 1, 1], [1.0, 1.0, 1.0]).unwrap();

    assert_eq!(
        plan.apply_pair_with_geometry(image, label, geometry)
            .unwrap_err(),
        TransformError::ShapeMismatch {
            image: [2, 1, 1],
            label: [1, 1, 1],
        }
    );
}

#[test]
fn foreground_crop_noops_when_label_has_no_foreground_then_pad_crop_updates_geometry() {
    let plan = TransformPlan {
        name: "no-foreground".to_string(),
        operations: vec![
            TransformOp::CropForeground { margin: 1 },
            TransformOp::PadCrop { size: [4, 4, 1] },
        ],
        image_interpolation: Interpolation::Linear,
        label_interpolation: Interpolation::Nearest,
    };
    let image = Volume3D::new([2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let label = Volume3D::new([2, 2, 1], vec![0_u16; 4]).unwrap();
    let geometry = VolumeGeometry::identity([2, 2, 1], [2.0, 3.0, 1.0]).unwrap();

    let prepared = plan
        .apply_pair_with_geometry(image, label, geometry)
        .unwrap();

    assert_eq!(prepared.crop_origin, [0, 0, 0]);
    assert_eq!(
        prepared.applied_operations,
        vec!["crop_foreground".to_string(), "pad_crop".to_string()]
    );
    assert_eq!(prepared.image.shape, [4, 4, 1]);
    assert_eq!(*prepared.image.get(0, 0, 0), 0.0);
    assert_eq!(*prepared.image.get(1, 1, 0), 1.0);
    assert_eq!(*prepared.image.get(2, 2, 0), 4.0);
    assert!(prepared.label.data.iter().all(|value| *value == 0));
    assert_eq!(prepared.geometry.shape, [4, 4, 1]);
    assert_eq!(prepared.geometry.origin, [-2.0, -3.0, 0.0]);
}

#[test]
fn parsing_and_resampling_errors_surface_as_transform_errors() {
    let parse_error = TransformPlan::from_toml_str("not = [valid").unwrap_err();
    assert!(matches!(parse_error, TransformError::PlanParse { .. }));

    let plan = TransformPlan {
        name: "bad-resample".to_string(),
        operations: vec![TransformOp::Resample {
            spacing: [1.0, 0.0, 1.0],
        }],
        image_interpolation: Interpolation::Linear,
        label_interpolation: Interpolation::Nearest,
    };
    let image = Volume3D::new([2, 1, 1], vec![0.0, 1.0]).unwrap();
    let label = Volume3D::new([2, 1, 1], vec![0_u16, 1]).unwrap();

    assert_eq!(
        plan.apply_pair(image, label).unwrap_err(),
        TransformError::InvalidSpacing {
            spacing: [1.0, 0.0, 1.0],
        }
    );
}

fn empty_plan() -> TransformPlan {
    TransformPlan {
        name: "empty".to_string(),
        operations: Vec::new(),
        image_interpolation: Interpolation::Linear,
        label_interpolation: Interpolation::Nearest,
    }
}

fn identity_direction() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}
