use std::{
    error::Error,
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use flate2::{write::GzEncoder, Compression};
use medkit_core::{CoordinateSystem, DType, ImageModality, SourceKind};
use medkit_io::{ImageMetadataReader, MedkitIoError, NiftiMetadataReader};

const HEADER_LEN: usize = 348;

#[derive(Debug, Clone, Copy)]
enum Endian {
    Little,
    Big,
}

#[derive(Debug, Clone)]
struct NiftiFixture {
    bytes: [u8; HEADER_LEN],
    endian: Endian,
}

impl NiftiFixture {
    fn new(endian: Endian, dims: &[i16], datatype: i16, pixdim: &[f32]) -> Self {
        let mut fixture = Self {
            bytes: [0; HEADER_LEN],
            endian,
        };
        fixture.put_i32(0, 348);
        fixture.put_i16(40, i16::try_from(dims.len()).unwrap());
        for (index, dim) in dims.iter().enumerate() {
            fixture.put_i16(42 + index * 2, *dim);
        }
        fixture.put_i16(70, datatype);
        fixture.put_i16(72, bitpix_for(datatype));
        fixture.put_f32(76, 1.0);
        for (index, spacing) in pixdim.iter().enumerate() {
            fixture.put_f32(80 + index * 4, *spacing);
        }
        fixture.put_f32(108, 352.0);
        fixture.bytes[344..348].copy_from_slice(b"n+1\0");
        fixture
    }

    fn with_magic(mut self, magic: [u8; 4]) -> Self {
        self.bytes[344..348].copy_from_slice(&magic);
        self
    }

    fn with_sizeof_hdr(mut self, sizeof_hdr: i32) -> Self {
        self.put_i32(0, sizeof_hdr);
        self
    }

    fn with_sform(mut self, rows: [[f32; 4]; 3]) -> Self {
        self.put_i16(254, 1);
        for (offset, row) in [(280, rows[0]), (296, rows[1]), (312, rows[2])] {
            for (index, value) in row.into_iter().enumerate() {
                self.put_f32(offset + index * 4, value);
            }
        }
        self
    }

    fn with_qform(mut self, quatern: [f32; 3], offsets: [f32; 3], qfac: f32) -> Self {
        self.put_i16(252, 1);
        self.put_f32(76, qfac);
        self.put_f32(256, quatern[0]);
        self.put_f32(260, quatern[1]);
        self.put_f32(264, quatern[2]);
        self.put_f32(268, offsets[0]);
        self.put_f32(272, offsets[1]);
        self.put_f32(276, offsets[2]);
        self
    }

    fn with_rank(mut self, rank: i16) -> Self {
        self.put_i16(40, rank);
        self
    }

    fn with_dim(mut self, index: usize, value: i16) -> Self {
        self.put_i16(40 + index * 2, value);
        self
    }

    fn with_pixdim(mut self, index: usize, value: f32) -> Self {
        self.put_f32(76 + index * 4, value);
        self
    }

    fn put_i32(&mut self, offset: usize, value: i32) {
        let bytes = match self.endian {
            Endian::Little => value.to_le_bytes(),
            Endian::Big => value.to_be_bytes(),
        };
        self.bytes[offset..offset + 4].copy_from_slice(&bytes);
    }

    fn put_i16(&mut self, offset: usize, value: i16) {
        let bytes = match self.endian {
            Endian::Little => value.to_le_bytes(),
            Endian::Big => value.to_be_bytes(),
        };
        self.bytes[offset..offset + 2].copy_from_slice(&bytes);
    }

    fn put_f32(&mut self, offset: usize, value: f32) {
        let bytes = match self.endian {
            Endian::Little => value.to_le_bytes(),
            Endian::Big => value.to_be_bytes(),
        };
        self.bytes[offset..offset + 4].copy_from_slice(&bytes);
    }
}

fn bitpix_for(datatype: i16) -> i16 {
    match datatype {
        1 => 1,
        2 | 256 => 8,
        4 | 512 => 16,
        8 | 16 | 768 => 32,
        64 => 64,
        _ => 0,
    }
}

fn temp_case_dir(case: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("medkit-io-{case}-{}-{nanos}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_nii(path: &Path, fixture: NiftiFixture) {
    let mut bytes = fixture.bytes.to_vec();
    bytes.extend_from_slice(&[0, 0, 0, 0]);
    fs::write(path, bytes).unwrap();
}

fn write_nii_gz(path: &Path, fixture: NiftiFixture) {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
    encoder.write_all(&fixture.bytes).unwrap();
    encoder.write_all(&[0, 0, 0, 0]).unwrap();
    fs::write(path, encoder.finish().unwrap()).unwrap();
}

fn assert_close(left: f64, right: f64) {
    assert!((left - right).abs() < 1e-6, "{left} != {right}");
}

#[test]
fn default_reader_assigns_unknown_modality() {
    let dir = temp_case_dir("default-reader");
    let path = dir.join("unknown.nii");
    write_nii(
        &path,
        NiftiFixture::new(Endian::Little, &[8, 8], 2, &[1.0, 1.0]),
    );

    let spec = NiftiMetadataReader::default().read_spec(&path).unwrap();

    assert_eq!(
        spec.modality(),
        &ImageModality::Other("unknown".to_string())
    );
}

#[test]
fn reads_little_endian_sform_ct_metadata_from_nii() {
    let dir = temp_case_dir("ct-sform");
    let path = dir.join("case_ct.nii");
    write_nii(
        &path,
        NiftiFixture::new(Endian::Little, &[512, 512, 64], 4, &[0.5, 0.5, 2.0]).with_sform([
            [0.5, 0.0, 0.0, -100.0],
            [0.0, 0.5, 0.0, -120.0],
            [0.0, 0.0, 2.0, -80.0],
        ]),
    );

    let spec = NiftiMetadataReader::with_default_modality(ImageModality::CT)
        .read_spec(&path)
        .unwrap();

    assert_eq!(spec.id(), "case_ct.nii");
    assert_eq!(spec.dtype(), DType::I16);
    assert_eq!(spec.modality(), &ImageModality::CT);
    assert_eq!(spec.provenance().source().kind(), &SourceKind::Nifti);
    assert_eq!(spec.geometry().shape().as_slice(), &[512, 512, 64]);
    assert_eq!(spec.geometry().coordinate_system(), &CoordinateSystem::RAS);
    assert_eq!(spec.geometry().spacing(), &[0.5, 0.5, 2.0]);
    assert_eq!(spec.geometry().origin(), &[-100.0, -120.0, -80.0]);
    assert_eq!(
        spec.geometry().direction(),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,]
    );
}

#[test]
fn reads_gzipped_segmentation_qform_with_negative_qfac() {
    let dir = temp_case_dir("seg-qform-gz");
    let path = dir.join("mask.nii.gz");
    write_nii_gz(
        &path,
        NiftiFixture::new(Endian::Little, &[32, 24, 16], 2, &[1.0, 1.5, 2.0]).with_qform(
            [0.0, 0.0, 0.0],
            [10.0, 20.0, -5.0],
            -1.0,
        ),
    );

    let spec = NiftiMetadataReader::with_default_modality(ImageModality::Segmentation)
        .read_spec(&path)
        .unwrap();

    assert_eq!(spec.id(), "mask.nii.gz");
    assert_eq!(spec.dtype(), DType::U8);
    assert_eq!(spec.modality(), &ImageModality::Segmentation);
    assert_eq!(spec.geometry().shape().as_slice(), &[32, 24, 16]);
    assert_eq!(spec.geometry().spacing(), &[1.0, 1.5, 2.0]);
    assert_eq!(spec.geometry().origin(), &[10.0, 20.0, -5.0]);
    assert_eq!(
        spec.geometry().direction(),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0,]
    );
}

#[test]
fn reads_big_endian_rank_four_mr_header_without_affine_codes() {
    let dir = temp_case_dir("big-rank4");
    let path = dir.join("multi_echo.hdr");
    write_nii(
        &path,
        NiftiFixture::new(Endian::Big, &[16, 16, 8, 2], 16, &[1.0, 1.25, 2.5, 4.0])
            .with_magic(*b"ni1\0"),
    );

    let spec = NiftiMetadataReader::with_default_modality(ImageModality::MR)
        .read_spec(&path)
        .unwrap();

    assert_eq!(spec.dtype(), DType::F32);
    assert_eq!(spec.modality(), &ImageModality::MR);
    assert_eq!(spec.geometry().shape().as_slice(), &[16, 16, 8, 2]);
    assert_eq!(spec.axes()[3].label(), "dim3");
    for (left, right) in spec.geometry().spacing().iter().zip([1.0, 1.25, 2.5, 4.0]) {
        assert_close(*left, right);
    }
}

#[test]
fn maps_supported_nifti_datatypes() {
    let cases = [
        (1, DType::Bool),
        (8, DType::I32),
        (64, DType::F64),
        (256, DType::I8),
        (512, DType::U16),
        (768, DType::U32),
    ];

    for (datatype, expected) in cases {
        let dir = temp_case_dir(&format!("dtype-{datatype}"));
        let path = dir.join(format!("dtype_{datatype}.nii"));
        write_nii(
            &path,
            NiftiFixture::new(Endian::Little, &[4, 4, 4], datatype, &[1.0, 1.0, 1.0]),
        );

        let spec = NiftiMetadataReader::new().read_spec(&path).unwrap();

        assert_eq!(spec.dtype(), expected);
    }
}

#[test]
fn rejects_unsupported_extension_before_opening_file() {
    let err = NiftiMetadataReader::new()
        .read_spec(Path::new("not-a-nifti.png"))
        .unwrap_err();

    match err {
        MedkitIoError::UnsupportedFormat { path } => {
            assert_eq!(path, PathBuf::from("not-a-nifti.png"));
        }
        other => panic!("expected unsupported format, got {other:?}"),
    }
}

#[test]
fn rejects_paths_without_file_names_as_unsupported_format() {
    let err = NiftiMetadataReader::new()
        .read_spec(Path::new(""))
        .unwrap_err();

    match err {
        MedkitIoError::UnsupportedFormat { path } => assert_eq!(path, PathBuf::from("")),
        other => panic!("expected unsupported format, got {other:?}"),
    }
}

#[test]
fn reports_io_error_for_short_header() {
    let dir = temp_case_dir("short");
    let path = dir.join("short.nii");
    fs::write(&path, [1, 2, 3]).unwrap();

    let err = NiftiMetadataReader::new().read_spec(&path).unwrap_err();

    match err {
        MedkitIoError::Io { path: err_path, .. } => assert_eq!(err_path, path),
        other => panic!("expected io error, got {other:?}"),
    }
}

#[test]
fn reports_io_error_for_missing_nifti_file() {
    let path = temp_case_dir("missing").join("missing.nii");

    let err = NiftiMetadataReader::new().read_spec(&path).unwrap_err();

    match &err {
        MedkitIoError::Io { path: err_path, .. } => assert_eq!(err_path, &path),
        other => panic!("expected io error, got {other:?}"),
    }
    assert!(err.source().is_some());
    assert!(err.to_string().contains("failed to read"));
}

#[test]
fn rejects_invalid_size_rank_dimension_magic_and_pixdim() {
    let cases = [
        (
            "bad-sizeof.nii",
            NiftiFixture::new(Endian::Little, &[8, 8, 8], 4, &[1.0, 1.0, 1.0]).with_sizeof_hdr(100),
            "sizeof_hdr must be 348",
        ),
        (
            "bad-rank.nii",
            NiftiFixture::new(Endian::Little, &[8, 8, 8], 4, &[1.0, 1.0, 1.0]).with_rank(0),
            "rank must be between 1 and 7",
        ),
        (
            "bad-dim.nii",
            NiftiFixture::new(Endian::Little, &[8, 8, 8], 4, &[1.0, 1.0, 1.0]).with_dim(2, 0),
            "dimension 2 must be positive",
        ),
        (
            "bad-magic.nii",
            NiftiFixture::new(Endian::Little, &[8, 8, 8], 4, &[1.0, 1.0, 1.0]).with_magic(*b"bad!"),
            "unsupported NIfTI magic",
        ),
        (
            "bad-pixdim.nii",
            NiftiFixture::new(Endian::Little, &[8, 8, 8], 4, &[1.0, 1.0, 1.0]).with_pixdim(2, -1.0),
            "pixdim[2] must be finite and positive",
        ),
    ];

    for (file_name, fixture, expected) in cases {
        let dir = temp_case_dir(file_name);
        let path = dir.join(file_name);
        write_nii(&path, fixture);

        let err = NiftiMetadataReader::new().read_spec(&path).unwrap_err();

        match err {
            MedkitIoError::InvalidHeader { reason } => assert!(
                reason.contains(expected),
                "expected {reason:?} to contain {expected:?}"
            ),
            other => panic!("expected invalid header, got {other:?}"),
        }
    }
}

#[test]
fn io_error_display_and_source_cover_all_variants() {
    let unsupported = MedkitIoError::UnsupportedFormat {
        path: PathBuf::from("image.png"),
    };
    let invalid = MedkitIoError::InvalidHeader {
        reason: "bad rank".to_string(),
    };
    let unsupported_dtype = MedkitIoError::UnsupportedDatatype { code: 2048 };
    let core: MedkitIoError = medkit_core::MedkitCoreError::EmptyShape.into();

    assert_eq!(
        unsupported.to_string(),
        "unsupported image metadata format: image.png"
    );
    assert_eq!(invalid.to_string(), "invalid image header: bad rank");
    assert_eq!(
        unsupported_dtype.to_string(),
        "unsupported NIfTI datatype code 2048"
    );
    assert_eq!(
        core.to_string(),
        "shape must contain at least one dimension"
    );

    assert!(unsupported.source().is_none());
    assert!(invalid.source().is_none());
    assert!(unsupported_dtype.source().is_none());
    assert!(core.source().is_some());
}

#[test]
fn rejects_unsupported_nifti_datatype() {
    let dir = temp_case_dir("unsupported-dtype");
    let path = dir.join("rgb.nii");
    write_nii(
        &path,
        NiftiFixture::new(Endian::Little, &[128, 128, 3], 128, &[1.0, 1.0, 1.0]),
    );

    let err = NiftiMetadataReader::new().read_spec(&path).unwrap_err();

    match err {
        MedkitIoError::UnsupportedDatatype { code } => assert_eq!(code, 128),
        other => panic!("expected unsupported datatype, got {other:?}"),
    }
}

#[test]
fn rejects_qform_quaternion_with_invalid_magnitude() {
    let dir = temp_case_dir("bad-qform");
    let path = dir.join("bad_qform.nii");
    write_nii(
        &path,
        NiftiFixture::new(Endian::Little, &[8, 8, 8], 64, &[1.0, 1.0, 1.0]).with_qform(
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            1.0,
        ),
    );

    let err = NiftiMetadataReader::new().read_spec(&path).unwrap_err();

    match err {
        MedkitIoError::InvalidHeader { reason } => {
            assert!(reason.contains("qform quaternion has magnitude greater than one"));
        }
        other => panic!("expected invalid header, got {other:?}"),
    }
}

#[test]
fn rejects_sform_with_degenerate_affine_column() {
    let dir = temp_case_dir("bad-sform");
    let path = dir.join("bad_sform.nii");
    write_nii(
        &path,
        NiftiFixture::new(Endian::Little, &[8, 8, 8], 512, &[1.0, 1.0, 1.0]).with_sform([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]),
    );

    let err = NiftiMetadataReader::new().read_spec(&path).unwrap_err();

    match err {
        MedkitIoError::InvalidHeader { reason } => {
            assert!(reason.contains("affine column 1 has invalid norm"));
        }
        other => panic!("expected invalid header, got {other:?}"),
    }
}
