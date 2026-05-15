use medkit_core::{
    Axis, AxisKind, CoordinateSystem, DType, GeometryCompatibility, GeometryMismatch,
    GeometryTolerance, ImageModality, ImageSpec, MedkitCoreError, Provenance, Shape, SourceKind,
    SourceRef, SpatialGeometry,
};

fn source(kind: SourceKind, uri: &str) -> SourceRef {
    SourceRef::new(kind, uri).unwrap()
}

fn provenance(kind: SourceKind, uri: &str, operations: &[&str]) -> Provenance {
    let mut provenance = Provenance::new(source(kind, uri));
    for operation in operations {
        provenance.add_operation(*operation).unwrap();
    }
    provenance
}

fn ct_geometry() -> SpatialGeometry {
    SpatialGeometry::identity(
        Shape::new(vec![512, 512, 128]).unwrap(),
        vec![0.742, 0.742, 1.25],
        CoordinateSystem::LPS,
    )
    .unwrap()
}

fn mr_geometry_with_direction() -> SpatialGeometry {
    SpatialGeometry::new(
        Shape::new(vec![240, 240, 155]).unwrap(),
        vec![1.0, 1.0, 1.0],
        vec![-119.5, -119.5, -77.0],
        vec![1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        CoordinateSystem::RAS,
    )
    .unwrap()
}

#[test]
fn dtype_reports_size_and_numeric_categories() {
    let cases = [
        (DType::Bool, 1, false, false, false),
        (DType::U8, 1, false, true, false),
        (DType::I8, 1, false, true, true),
        (DType::U16, 2, false, true, false),
        (DType::I16, 2, false, true, true),
        (DType::U32, 4, false, true, false),
        (DType::I32, 4, false, true, true),
        (DType::F16, 2, true, false, true),
        (DType::F32, 4, true, false, true),
        (DType::F64, 8, true, false, true),
    ];

    for (dtype, size, is_float, is_integer, is_signed) in cases {
        assert_eq!(dtype.size_bytes(), size);
        assert_eq!(dtype.is_float(), is_float);
        assert_eq!(dtype.is_integer(), is_integer);
        assert_eq!(dtype.is_signed(), is_signed);
    }
}

#[test]
fn axis_construction_covers_defaults_and_custom_axes() {
    let x = Axis::x();
    let y = Axis::y();
    let z = Axis::z();
    let channel = Axis::channel();
    let time = Axis::new(AxisKind::Time, "time").unwrap();
    let other = Axis::new(AxisKind::Other("phase".to_string()), "phase").unwrap();

    assert_eq!(x.kind(), &AxisKind::LeftRight);
    assert_eq!(x.label(), "x");
    assert_eq!(y.kind(), &AxisKind::PosteriorAnterior);
    assert_eq!(z.kind(), &AxisKind::InferiorSuperior);
    assert_eq!(channel.kind(), &AxisKind::Channel);
    assert_eq!(time.label(), "time");
    assert_eq!(other.kind(), &AxisKind::Other("phase".to_string()));

    assert_eq!(
        Axis::new(AxisKind::Channel, "").unwrap_err(),
        MedkitCoreError::EmptyAxisLabel
    );
}

#[test]
fn shape_rejects_invalid_dimensions_and_reports_rank_elements_and_dims() {
    assert_eq!(
        Shape::new(Vec::<usize>::new()).unwrap_err(),
        MedkitCoreError::EmptyShape
    );
    assert_eq!(
        Shape::new(vec![64, 0, 32]).unwrap_err(),
        MedkitCoreError::ZeroDimension { index: 1 }
    );

    let shape = Shape::new(vec![64, 48, 32]).unwrap();
    assert_eq!(shape.rank(), 3);
    assert_eq!(shape.dim(0), Some(64));
    assert_eq!(shape.dim(3), None);
    assert_eq!(shape.as_slice(), &[64, 48, 32]);
    assert_eq!(shape.num_elements(), 64 * 48 * 32);
}

#[test]
fn source_and_provenance_track_different_source_kinds() {
    let source_kinds = [
        SourceKind::Dicom,
        SourceKind::Nifti,
        SourceKind::WholeSlide,
        SourceKind::Zarr,
        SourceKind::Memory,
        SourceKind::Other("custom".to_string()),
    ];

    for (index, kind) in source_kinds.into_iter().enumerate() {
        let source = source(kind.clone(), &format!("source-{index}"));
        assert_eq!(source.kind(), &kind);
        assert_eq!(source.uri(), format!("source-{index}"));
    }

    assert_eq!(
        SourceRef::new(SourceKind::Nifti, "").unwrap_err(),
        MedkitCoreError::EmptySourceUri
    );

    let mut provenance = provenance(SourceKind::Dicom, "dicom://study/series", &["scan"]);
    provenance.add_operation("dcm2niix").unwrap();
    assert_eq!(provenance.source().uri(), "dicom://study/series");
    assert_eq!(
        provenance.operations(),
        &["scan".to_string(), "dcm2niix".to_string()]
    );
    assert_eq!(
        provenance.add_operation("").unwrap_err(),
        MedkitCoreError::EmptyProvenanceOperation
    );
}

#[test]
fn spatial_geometry_builds_identity_and_affine_for_ct() {
    let geometry = ct_geometry();
    assert_eq!(geometry.shape().as_slice(), &[512, 512, 128]);
    assert_eq!(geometry.spacing(), &[0.742, 0.742, 1.25]);
    assert_eq!(geometry.origin(), &[0.0, 0.0, 0.0]);
    assert_eq!(
        geometry.direction(),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,]
    );
    assert_eq!(geometry.coordinate_system(), &CoordinateSystem::LPS);
    assert_eq!(
        geometry.affine(),
        vec![0.742, 0.0, 0.0, 0.0, 0.0, 0.742, 0.0, 0.0, 0.0, 0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 1.0,]
    );
}

#[test]
fn spatial_geometry_builds_oriented_affine_for_mri() {
    let geometry = mr_geometry_with_direction();

    assert_eq!(geometry.coordinate_system(), &CoordinateSystem::RAS);
    assert_eq!(
        geometry.affine(),
        vec![
            1.0, 0.0, 0.0, -119.5, 0.0, 0.0, -1.0, -119.5, 0.0, 1.0, 0.0, -77.0, 0.0, 0.0, 0.0,
            1.0,
        ]
    );
}

#[test]
fn spatial_geometry_rejects_invalid_rank_and_spacing_data() {
    let shape = Shape::new(vec![8, 8, 8]).unwrap();
    assert_eq!(
        SpatialGeometry::new(
            shape.clone(),
            vec![1.0, 1.0],
            vec![0.0, 0.0, 0.0],
            vec![1.0; 9],
            CoordinateSystem::RAS,
        )
        .unwrap_err(),
        MedkitCoreError::SpacingRankMismatch {
            spacing: 2,
            rank: 3
        }
    );
    assert_eq!(
        SpatialGeometry::new(
            shape.clone(),
            vec![1.0, 1.0, 1.0],
            vec![0.0],
            vec![1.0; 9],
            CoordinateSystem::RAS,
        )
        .unwrap_err(),
        MedkitCoreError::OriginRankMismatch { origin: 1, rank: 3 }
    );
    assert_eq!(
        SpatialGeometry::new(
            shape.clone(),
            vec![1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0],
            vec![1.0; 8],
            CoordinateSystem::RAS,
        )
        .unwrap_err(),
        MedkitCoreError::DirectionSizeMismatch {
            values: 8,
            expected: 9
        }
    );
    assert_eq!(
        SpatialGeometry::new(
            shape,
            vec![1.0, f64::INFINITY, 1.0],
            vec![0.0, 0.0, 0.0],
            vec![1.0; 9],
            CoordinateSystem::RAS,
        )
        .unwrap_err(),
        MedkitCoreError::InvalidSpacing {
            index: 1,
            value: f64::INFINITY
        }
    );
}

#[test]
fn image_spec_builds_ct_image_and_segmentation_mask_specs() {
    let image_spec = ImageSpec::builder(
        "ct-image",
        DType::I16,
        ct_geometry(),
        ImageModality::CT,
        provenance(SourceKind::Dicom, "dicom://ct", &["indexed"]),
    )
    .build()
    .unwrap();
    let mask_spec = ImageSpec::builder(
        "ct-mask",
        DType::U8,
        ct_geometry(),
        ImageModality::Segmentation,
        provenance(
            SourceKind::Nifti,
            "file:///ct-mask.nii.gz",
            &["manual-label"],
        ),
    )
    .axes(vec![Axis::x(), Axis::y(), Axis::z()])
    .build()
    .unwrap();

    assert_eq!(image_spec.id(), "ct-image");
    assert_eq!(image_spec.dtype(), DType::I16);
    assert_eq!(image_spec.axes().len(), 3);
    assert_eq!(image_spec.modality(), &ImageModality::CT);
    assert_eq!(image_spec.provenance().source().kind(), &SourceKind::Dicom);
    assert!(image_spec
        .geometry()
        .is_compatible_with(mask_spec.geometry()));
    assert_eq!(mask_spec.dtype(), DType::U8);
    assert_eq!(mask_spec.modality(), &ImageModality::Segmentation);
}

#[test]
fn image_spec_supports_rank_four_and_rejects_bad_builder_inputs() {
    let geometry = SpatialGeometry::identity(
        Shape::new(vec![4, 128, 128, 64]).unwrap(),
        vec![1.0, 0.8, 0.8, 2.0],
        CoordinateSystem::Other("scanner-native".to_string()),
    )
    .unwrap();
    let spec = ImageSpec::builder(
        "multi-channel-mr",
        DType::F32,
        geometry.clone(),
        ImageModality::MR,
        provenance(
            SourceKind::Memory,
            "memory://synthetic",
            &["stacked-modalities"],
        ),
    )
    .build()
    .unwrap();

    assert_eq!(spec.axes()[0], Axis::x());
    assert_eq!(spec.axes()[3].label(), "dim3");
    assert_eq!(spec.geometry().shape().rank(), 4);

    assert_eq!(
        ImageSpec::builder(
            "",
            DType::F32,
            geometry.clone(),
            ImageModality::MR,
            provenance(SourceKind::Memory, "memory://empty-id", &[]),
        )
        .build()
        .unwrap_err(),
        MedkitCoreError::EmptyImageId
    );
    assert_eq!(
        ImageSpec::builder(
            "bad-axes",
            DType::F32,
            geometry,
            ImageModality::MR,
            provenance(SourceKind::Memory, "memory://bad-axes", &[]),
        )
        .axes(vec![Axis::channel()])
        .build()
        .unwrap_err(),
        MedkitCoreError::AxisRankMismatch { axes: 1, rank: 4 }
    );
}

#[test]
fn image_spec_default_axes_cover_lower_rank_images() {
    let one_dimensional = SpatialGeometry::identity(
        Shape::new(vec![128]).unwrap(),
        vec![0.5],
        CoordinateSystem::Other("line-profile".to_string()),
    )
    .unwrap();
    let two_dimensional = SpatialGeometry::identity(
        Shape::new(vec![1024, 768]).unwrap(),
        vec![0.2, 0.2],
        CoordinateSystem::LPS,
    )
    .unwrap();

    let one_dimensional_spec = ImageSpec::builder(
        "line-profile",
        DType::F32,
        one_dimensional,
        ImageModality::Other("spectral-profile".to_string()),
        provenance(SourceKind::Memory, "memory://line-profile", &[]),
    )
    .build()
    .unwrap();
    let two_dimensional_spec = ImageSpec::builder(
        "xray",
        DType::U16,
        two_dimensional,
        ImageModality::XR,
        provenance(SourceKind::Dicom, "dicom://projection", &[]),
    )
    .build()
    .unwrap();

    assert_eq!(one_dimensional_spec.axes(), &[Axis::x()]);
    assert_eq!(two_dimensional_spec.axes(), &[Axis::x(), Axis::y()]);
}

#[test]
fn geometry_compatibility_accepts_small_mri_drift_with_tolerance() {
    let left = mr_geometry_with_direction();
    let right = SpatialGeometry::new(
        Shape::new(vec![240, 240, 155]).unwrap(),
        vec![1.0 + 5e-6, 1.0, 1.0],
        vec![-119.5, -119.5 + 5e-5, -77.0],
        vec![1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0 + 5e-7, 0.0],
        CoordinateSystem::RAS,
    )
    .unwrap();

    let report = left.compatibility_with(&right, GeometryTolerance::default());
    assert!(report.is_compatible());
    assert!(report.mismatches().is_empty());
    assert!(left.is_compatible_with(&right));
}

#[test]
fn geometry_compatibility_reports_multiple_medical_mismatches() {
    let left = ct_geometry();
    let right = SpatialGeometry::new(
        Shape::new(vec![512, 512, 64]).unwrap(),
        vec![0.8, 0.742, 1.25],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        CoordinateSystem::RAS,
    )
    .unwrap();

    let report = left.compatibility_with(&right, GeometryTolerance::new(1e-6, 1e-6, 1e-6).unwrap());

    assert!(!report.is_compatible());
    assert_eq!(
        report.mismatches(),
        &[
            GeometryMismatch::Shape {
                left: vec![512, 512, 128],
                right: vec![512, 512, 64],
            },
            GeometryMismatch::CoordinateSystem {
                left: "LPS".to_string(),
                right: "RAS".to_string(),
            },
            GeometryMismatch::Spacing {
                index: 0,
                left: 0.742,
                right: 0.8,
            },
            GeometryMismatch::Origin {
                index: 1,
                left: 0.0,
                right: 1.0,
            },
            GeometryMismatch::Direction {
                index: 4,
                left: 1.0,
                right: -1.0,
            },
        ]
    );
}

#[test]
fn geometry_tolerance_rejects_invalid_values_for_each_field() {
    assert_eq!(
        GeometryTolerance::new(-1.0, 0.0, 0.0).unwrap_err(),
        MedkitCoreError::InvalidTolerance {
            field: "spacing",
            value: -1.0
        }
    );
    match GeometryTolerance::new(0.0, f64::NAN, 0.0).unwrap_err() {
        MedkitCoreError::InvalidTolerance { field, value } => {
            assert_eq!(field, "origin");
            assert!(value.is_nan());
        }
        other => panic!("expected invalid origin tolerance, got {other:?}"),
    }
    assert_eq!(
        GeometryTolerance::new(0.0, 0.0, f64::INFINITY).unwrap_err(),
        MedkitCoreError::InvalidTolerance {
            field: "direction",
            value: f64::INFINITY
        }
    );
}

#[test]
fn modalities_cover_radiology_pathology_and_custom_cases() {
    let modalities = [
        ImageModality::CT,
        ImageModality::MR,
        ImageModality::PET,
        ImageModality::US,
        ImageModality::XR,
        ImageModality::Pathology,
        ImageModality::Segmentation,
        ImageModality::Other("fundus".to_string()),
    ];

    assert!(modalities.contains(&ImageModality::CT));
    assert!(modalities.contains(&ImageModality::Other("fundus".to_string())));
}

#[test]
fn error_display_messages_are_specific() {
    let errors = [
        (
            MedkitCoreError::EmptyShape,
            "shape must contain at least one dimension",
        ),
        (
            MedkitCoreError::ZeroDimension { index: 2 },
            "shape dimension 2 must be non-zero",
        ),
        (
            MedkitCoreError::EmptyAxisLabel,
            "axis label must not be empty",
        ),
        (
            MedkitCoreError::AxisRankMismatch { axes: 2, rank: 3 },
            "axis count 2 does not match rank 3",
        ),
        (
            MedkitCoreError::SpacingRankMismatch {
                spacing: 2,
                rank: 3,
            },
            "spacing count 2 does not match rank 3",
        ),
        (
            MedkitCoreError::OriginRankMismatch { origin: 2, rank: 3 },
            "origin count 2 does not match rank 3",
        ),
        (
            MedkitCoreError::DirectionSizeMismatch {
                values: 8,
                expected: 9,
            },
            "direction matrix has 8 values, expected 9",
        ),
        (
            MedkitCoreError::InvalidSpacing {
                index: 1,
                value: -1.0,
            },
            "spacing value -1 at index 1 must be finite and positive",
        ),
        (
            MedkitCoreError::InvalidTolerance {
                field: "spacing",
                value: -1.0,
            },
            "tolerance spacing=-1 must be finite and non-negative",
        ),
        (MedkitCoreError::EmptyImageId, "image id must not be empty"),
        (
            MedkitCoreError::EmptySourceUri,
            "source URI must not be empty",
        ),
        (
            MedkitCoreError::EmptyProvenanceOperation,
            "provenance operation must not be empty",
        ),
    ];

    for (error, message) in errors {
        assert_eq!(error.to_string(), message);
    }
}
