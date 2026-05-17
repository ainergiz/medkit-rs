use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use flate2::read::GzDecoder;
use medkit_core::{
    CoordinateSystem, DType, ImageModality, ImageSpec, Provenance, Shape, SourceKind, SourceRef,
    SpatialGeometry,
};

use crate::{ImageMetadataReader, MedkitIoError, Result};

const NIFTI1_HEADER_LEN: usize = 348;
const NIFTI1_SIZEOF_HDR: i32 = 348;

/// Metadata-only reader for NIfTI-1 `.nii`, `.nii.gz`, `.hdr`, and `.hdr.gz` files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NiftiMetadataReader {
    default_modality: ImageModality,
}

impl NiftiMetadataReader {
    /// Creates a NIfTI reader with unknown image modality.
    pub fn new() -> Self {
        Self {
            default_modality: ImageModality::Other("unknown".to_string()),
        }
    }

    /// Creates a NIfTI reader that assigns `modality` to produced specs.
    pub fn with_default_modality(default_modality: ImageModality) -> Self {
        Self { default_modality }
    }
}

impl Default for NiftiMetadataReader {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageMetadataReader for NiftiMetadataReader {
    fn read_spec(&self, path: &Path) -> Result<ImageSpec> {
        if !looks_like_nifti(path) {
            return Err(MedkitIoError::UnsupportedFormat {
                path: path.to_path_buf(),
            });
        }

        let bytes = read_header_bytes(path)?;
        let header = Nifti1Header::parse(bytes)?;
        let shape = Shape::new(header.shape())?;
        let geometry = header.geometry(shape)?;
        let dtype = header.dtype()?;
        let source = SourceRef::new(SourceKind::Nifti, path_to_uriish_string(path))
            .expect("NIfTI source path is non-empty after format detection");
        let provenance = Provenance::new(source);

        ImageSpec::builder(
            image_id_from_path(path),
            dtype,
            geometry,
            self.default_modality.clone(),
            provenance,
        )
        .build()
        .map_err(Into::into)
    }
}

fn looks_like_nifti(path: &Path) -> bool {
    let Some(file_name) = path.file_name().and_then(|file_name| file_name.to_str()) else {
        return false;
    };
    let file_name = file_name.to_ascii_lowercase();
    file_name.ends_with(".nii")
        || file_name.ends_with(".nii.gz")
        || file_name.ends_with(".hdr")
        || file_name.ends_with(".hdr.gz")
}

fn read_header_bytes(path: &Path) -> Result<[u8; NIFTI1_HEADER_LEN]> {
    let file = File::open(path).map_err(|source| MedkitIoError::io(path, source))?;
    let mut bytes = [0_u8; NIFTI1_HEADER_LEN];
    if path
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .is_some_and(|file_name| file_name.to_ascii_lowercase().ends_with(".gz"))
    {
        let mut decoder = GzDecoder::new(file);
        decoder
            .read_exact(&mut bytes)
            .map_err(|source| MedkitIoError::io(path, source))?;
    } else {
        let mut file = file;
        file.seek(SeekFrom::Start(0))
            .map_err(|source| MedkitIoError::io(path, source))?;
        file.read_exact(&mut bytes)
            .map_err(|source| MedkitIoError::io(path, source))?;
    }
    Ok(bytes)
}

fn path_to_uriish_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn image_id_from_path(path: &Path) -> String {
    path.file_name()
        .and_then(|file_name| file_name.to_str())
        .unwrap_or("nifti-image")
        .to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endian {
    Little,
    Big,
}

#[derive(Debug, Clone, PartialEq)]
struct Nifti1Header {
    dim: [i16; 8],
    datatype: i16,
    pixdim: [f32; 8],
    qform_code: i16,
    sform_code: i16,
    quatern_b: f32,
    quatern_c: f32,
    quatern_d: f32,
    qoffset_x: f32,
    qoffset_y: f32,
    qoffset_z: f32,
    srow_x: [f32; 4],
    srow_y: [f32; 4],
    srow_z: [f32; 4],
    magic: [u8; 4],
}

impl Nifti1Header {
    fn parse(bytes: [u8; NIFTI1_HEADER_LEN]) -> Result<Self> {
        let endian = match i32::from_le_bytes(bytes[0..4].try_into().expect("slice len")) {
            NIFTI1_SIZEOF_HDR => Endian::Little,
            _ if i32::from_be_bytes(bytes[0..4].try_into().expect("slice len"))
                == NIFTI1_SIZEOF_HDR =>
            {
                Endian::Big
            }
            value => {
                return Err(MedkitIoError::invalid_header(format!(
                    "sizeof_hdr must be 348, got {value}"
                )));
            }
        };

        let mut dim = [0_i16; 8];
        for (index, value) in dim.iter_mut().enumerate() {
            *value = read_i16(&bytes, 40 + index * 2, endian);
        }

        let mut pixdim = [0_f32; 8];
        for (index, value) in pixdim.iter_mut().enumerate() {
            *value = read_f32(&bytes, 76 + index * 4, endian);
        }

        let mut srow_x = [0_f32; 4];
        let mut srow_y = [0_f32; 4];
        let mut srow_z = [0_f32; 4];
        for index in 0..4 {
            srow_x[index] = read_f32(&bytes, 280 + index * 4, endian);
            srow_y[index] = read_f32(&bytes, 296 + index * 4, endian);
            srow_z[index] = read_f32(&bytes, 312 + index * 4, endian);
        }

        let header = Self {
            dim,
            datatype: read_i16(&bytes, 70, endian),
            pixdim,
            qform_code: read_i16(&bytes, 252, endian),
            sform_code: read_i16(&bytes, 254, endian),
            quatern_b: read_f32(&bytes, 256, endian),
            quatern_c: read_f32(&bytes, 260, endian),
            quatern_d: read_f32(&bytes, 264, endian),
            qoffset_x: read_f32(&bytes, 268, endian),
            qoffset_y: read_f32(&bytes, 272, endian),
            qoffset_z: read_f32(&bytes, 276, endian),
            srow_x,
            srow_y,
            srow_z,
            magic: bytes[344..348].try_into().expect("slice len"),
        };
        header.validate()?;
        Ok(header)
    }

    fn validate(&self) -> Result<()> {
        if !matches!(&self.magic, b"n+1\0" | b"ni1\0") {
            return Err(MedkitIoError::invalid_header(format!(
                "unsupported NIfTI magic {:?}",
                String::from_utf8_lossy(&self.magic)
            )));
        }
        let rank = self.dim[0];
        if !(1..=7).contains(&rank) {
            return Err(MedkitIoError::invalid_header(format!(
                "rank must be between 1 and 7, got {rank}"
            )));
        }
        for axis in 0..rank as usize {
            let dim = self.dim[axis + 1];
            if dim <= 0 {
                return Err(MedkitIoError::invalid_header(format!(
                    "dimension {} must be positive, got {dim}",
                    axis + 1
                )));
            }
        }
        for axis in 0..rank as usize {
            let spacing = self.pixdim[axis + 1];
            if !spacing.is_finite() || spacing <= 0.0 {
                return Err(MedkitIoError::invalid_header(format!(
                    "pixdim[{}] must be finite and positive, got {spacing}",
                    axis + 1
                )));
            }
        }
        Ok(())
    }

    fn shape(&self) -> Vec<usize> {
        let rank = self.dim[0] as usize;
        let mut shape = Vec::with_capacity(rank);
        for axis in 0..rank {
            let dim = self.dim[axis + 1];
            shape.push(dim as usize);
        }
        shape
    }

    fn dtype(&self) -> Result<DType> {
        match self.datatype {
            1 => Ok(DType::Bool),
            2 => Ok(DType::U8),
            4 => Ok(DType::I16),
            8 => Ok(DType::I32),
            16 => Ok(DType::F32),
            64 => Ok(DType::F64),
            256 => Ok(DType::I8),
            512 => Ok(DType::U16),
            768 => Ok(DType::U32),
            code => Err(MedkitIoError::UnsupportedDatatype { code }),
        }
    }

    fn geometry(&self, shape: Shape) -> Result<SpatialGeometry> {
        let affine = if self.sform_code > 0 {
            Some(self.sform_affine())
        } else if self.qform_code > 0 {
            Some(self.qform_affine()?)
        } else {
            None
        };

        match affine {
            Some(affine) => geometry_from_affine(shape, &self.pixdim, affine),
            None => {
                SpatialGeometry::identity(shape, self.spacing_for_rank(), CoordinateSystem::RAS)
                    .map_err(Into::into)
            }
        }
    }

    fn spacing_for_rank(&self) -> Vec<f64> {
        let rank = self.dim[0] as usize;
        (0..rank)
            .map(|axis| f64::from(self.pixdim[axis + 1]))
            .collect()
    }

    fn sform_affine(&self) -> [[f64; 4]; 3] {
        [
            self.srow_x.map(f64::from),
            self.srow_y.map(f64::from),
            self.srow_z.map(f64::from),
        ]
    }

    fn qform_affine(&self) -> Result<[[f64; 4]; 3]> {
        let b = f64::from(self.quatern_b);
        let c = f64::from(self.quatern_c);
        let d = f64::from(self.quatern_d);
        let a_squared = 1.0 - (b * b + c * c + d * d);
        if a_squared < -1e-5 {
            return Err(MedkitIoError::invalid_header(
                "qform quaternion has magnitude greater than one",
            ));
        }
        let a = a_squared.max(0.0).sqrt();
        let qfac = if self.pixdim[0] < 0.0 { -1.0 } else { 1.0 };
        let dx = f64::from(self.pixdim[1]);
        let dy = f64::from(self.pixdim[2]);
        let dz = f64::from(self.pixdim[3]) * qfac;

        let r11 = a * a + b * b - c * c - d * d;
        let r12 = 2.0 * b * c - 2.0 * a * d;
        let r13 = 2.0 * b * d + 2.0 * a * c;
        let r21 = 2.0 * b * c + 2.0 * a * d;
        let r22 = a * a + c * c - b * b - d * d;
        let r23 = 2.0 * c * d - 2.0 * a * b;
        let r31 = 2.0 * b * d - 2.0 * a * c;
        let r32 = 2.0 * c * d + 2.0 * a * b;
        let r33 = a * a + d * d - c * c - b * b;

        Ok([
            [r11 * dx, r12 * dy, r13 * dz, f64::from(self.qoffset_x)],
            [r21 * dx, r22 * dy, r23 * dz, f64::from(self.qoffset_y)],
            [r31 * dx, r32 * dy, r33 * dz, f64::from(self.qoffset_z)],
        ])
    }
}

fn geometry_from_affine(
    shape: Shape,
    pixdim: &[f32; 8],
    affine: [[f64; 4]; 3],
) -> Result<SpatialGeometry> {
    let rank = shape.rank();
    let mut spacing = (0..rank)
        .map(|axis| f64::from(pixdim[axis + 1]))
        .collect::<Vec<_>>();
    let mut origin = vec![0.0; rank];
    let mut direction = vec![0.0; rank * rank];

    for axis in 0..rank {
        direction[axis * rank + axis] = 1.0;
    }

    let spatial_rank = rank.min(3);
    for row in 0..spatial_rank {
        origin[row] = affine[row][3];
    }
    for col in 0..spatial_rank {
        let norm = (0..3)
            .map(|row| affine[row][col] * affine[row][col])
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return Err(MedkitIoError::invalid_header(format!(
                "affine column {col} has invalid norm {norm}"
            )));
        }
        spacing[col] = norm;
        for row in 0..spatial_rank {
            direction[row * rank + col] = affine[row][col] / norm;
        }
    }

    SpatialGeometry::new(shape, spacing, origin, direction, CoordinateSystem::RAS)
        .map_err(Into::into)
}

fn read_i16(bytes: &[u8; NIFTI1_HEADER_LEN], offset: usize, endian: Endian) -> i16 {
    let value = [bytes[offset], bytes[offset + 1]];
    match endian {
        Endian::Little => i16::from_le_bytes(value),
        Endian::Big => i16::from_be_bytes(value),
    }
}

fn read_f32(bytes: &[u8; NIFTI1_HEADER_LEN], offset: usize, endian: Endian) -> f32 {
    let value = [
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ];
    match endian {
        Endian::Little => f32::from_le_bytes(value),
        Endian::Big => f32::from_be_bytes(value),
    }
}
