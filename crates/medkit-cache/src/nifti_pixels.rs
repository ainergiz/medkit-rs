use std::{fs::File, io::Read, path::Path};

use flate2::read::GzDecoder;
use medkit_transform::{Volume3D, VolumeGeometry};

use crate::{CacheError, Result};

const HEADER_LEN: usize = 348;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endian {
    Little,
    Big,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PixelKind {
    U8,
    I16,
    I32,
    F32,
    F64,
    I8,
    U16,
    U32,
}

#[derive(Debug, Clone, PartialEq)]
struct Header {
    endian: Endian,
    shape: [usize; 3],
    datatype: PixelKind,
    vox_offset: usize,
    geometry: VolumeGeometry,
}

pub(crate) struct LoadedVolume<T> {
    pub(crate) volume: Volume3D<T>,
    pub(crate) geometry: VolumeGeometry,
}

pub(crate) fn load_image_f32(path: &Path) -> Result<LoadedVolume<f32>> {
    let bytes = read_all(path)?;
    let header = parse_header(path, &bytes)?;
    let values = read_pixels(path, &bytes, &header, |value| value as f32)?;
    Ok(LoadedVolume {
        volume: Volume3D::new(header.shape, values)?,
        geometry: header.geometry,
    })
}

pub(crate) fn load_label_u16(path: &Path) -> Result<LoadedVolume<u16>> {
    let bytes = read_all(path)?;
    let header = parse_header(path, &bytes)?;
    let values = read_pixels(path, &bytes, &header, |value| value.max(0.0).round() as u16)?;
    Ok(LoadedVolume {
        volume: Volume3D::new(header.shape, values)?,
        geometry: header.geometry,
    })
}

fn read_all(path: &Path) -> Result<Vec<u8>> {
    let file = File::open(path).map_err(|source| CacheError::io(path, source))?;
    let mut bytes = Vec::new();
    if path
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .is_some_and(|file_name| file_name.to_ascii_lowercase().ends_with(".gz"))
    {
        let mut decoder = GzDecoder::new(file);
        decoder
            .read_to_end(&mut bytes)
            .map_err(|source| CacheError::io(path, source))?;
    } else {
        let mut file = file;
        file.read_to_end(&mut bytes)
            .map_err(|source| CacheError::io(path, source))?;
    }
    Ok(bytes)
}

fn parse_header(path: &Path, bytes: &[u8]) -> Result<Header> {
    if bytes.len() < HEADER_LEN {
        return Err(CacheError::nifti(
            path,
            "file is shorter than a NIfTI-1 header",
        ));
    }
    let endian = match i32_from(&bytes[0..4], Endian::Little) {
        348 => Endian::Little,
        _ if i32_from(&bytes[0..4], Endian::Big) == 348 => Endian::Big,
        value => {
            return Err(CacheError::nifti(
                path,
                format!("sizeof_hdr must be 348, got {value}"),
            ))
        }
    };
    let rank = i16_from(&bytes[40..42], endian);
    if !(1..=7).contains(&rank) {
        return Err(CacheError::nifti(
            path,
            format!("rank must be between 1 and 7, got {rank}"),
        ));
    }
    let mut shape = [1_usize; 3];
    for axis in 0..rank.min(3) as usize {
        let dim = i16_from(&bytes[42 + axis * 2..44 + axis * 2], endian);
        if dim <= 0 {
            return Err(CacheError::nifti(
                path,
                format!("dimension {} must be positive", axis + 1),
            ));
        }
        shape[axis] = dim as usize;
    }
    let mut pixdim = [0.0_f32; 8];
    for (index, value) in pixdim.iter_mut().enumerate() {
        *value = f32_from(&bytes[76 + index * 4..80 + index * 4], endian);
    }
    let spacing = spacing_from_pixdim(path, rank as usize, &pixdim)?;
    let datatype_code = i16_from(&bytes[70..72], endian);
    let datatype = match datatype_code {
        2 => PixelKind::U8,
        4 => PixelKind::I16,
        8 => PixelKind::I32,
        16 => PixelKind::F32,
        64 => PixelKind::F64,
        256 => PixelKind::I8,
        512 => PixelKind::U16,
        768 => PixelKind::U32,
        code => {
            return Err(CacheError::nifti(
                path,
                format!("unsupported datatype code {code}"),
            ))
        }
    };
    let vox_offset = f32_from(&bytes[108..112], endian).max(352.0) as usize;
    let qform_code = i16_from(&bytes[252..254], endian);
    let sform_code = i16_from(&bytes[254..256], endian);
    let geometry = if sform_code > 0 {
        geometry_from_affine(path, shape, sform_affine(bytes, endian))?
    } else if qform_code > 0 {
        geometry_from_affine(path, shape, qform_affine(path, bytes, endian, &pixdim)?)?
    } else {
        VolumeGeometry::identity(shape, spacing)?
    };
    Ok(Header {
        endian,
        shape,
        datatype,
        vox_offset,
        geometry,
    })
}

fn spacing_from_pixdim(path: &Path, rank: usize, pixdim: &[f32; 8]) -> Result<[f64; 3]> {
    let mut spacing = [1.0_f64; 3];
    for axis in 0..rank.min(3) {
        let value = f64::from(pixdim[axis + 1]);
        if !value.is_finite() || value <= 0.0 {
            return Err(CacheError::nifti(
                path,
                format!("pixdim[{}] must be finite and positive", axis + 1),
            ));
        }
        spacing[axis] = value;
    }
    Ok(spacing)
}

fn sform_affine(bytes: &[u8], endian: Endian) -> [[f64; 4]; 3] {
    let mut affine = [[0.0_f64; 4]; 3];
    for column in 0..4 {
        affine[0][column] = f64::from(f32_from(&bytes[280 + column * 4..284 + column * 4], endian));
        affine[1][column] = f64::from(f32_from(&bytes[296 + column * 4..300 + column * 4], endian));
        affine[2][column] = f64::from(f32_from(&bytes[312 + column * 4..316 + column * 4], endian));
    }
    affine
}

fn qform_affine(
    path: &Path,
    bytes: &[u8],
    endian: Endian,
    pixdim: &[f32; 8],
) -> Result<[[f64; 4]; 3]> {
    let b = f64::from(f32_from(&bytes[256..260], endian));
    let c = f64::from(f32_from(&bytes[260..264], endian));
    let d = f64::from(f32_from(&bytes[264..268], endian));
    let a_squared = 1.0 - (b * b + c * c + d * d);
    if a_squared < -1e-5 {
        return Err(CacheError::nifti(
            path,
            "qform quaternion has magnitude greater than one",
        ));
    }
    let a = a_squared.max(0.0).sqrt();
    let qfac = if pixdim[0] < 0.0 { -1.0 } else { 1.0 };
    let dx = f64::from(pixdim[1]);
    let dy = f64::from(pixdim[2]);
    let dz = f64::from(pixdim[3]) * qfac;

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
        [
            r11 * dx,
            r12 * dy,
            r13 * dz,
            f64::from(f32_from(&bytes[268..272], endian)),
        ],
        [
            r21 * dx,
            r22 * dy,
            r23 * dz,
            f64::from(f32_from(&bytes[272..276], endian)),
        ],
        [
            r31 * dx,
            r32 * dy,
            r33 * dz,
            f64::from(f32_from(&bytes[276..280], endian)),
        ],
    ])
}

fn geometry_from_affine(
    path: &Path,
    shape: [usize; 3],
    affine: [[f64; 4]; 3],
) -> Result<VolumeGeometry> {
    let mut spacing = [1.0_f64; 3];
    let mut direction = [[0.0_f64; 3]; 3];
    for axis in 0..3 {
        let norm = (0..3)
            .map(|row| affine[row][axis] * affine[row][axis])
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return Err(CacheError::nifti(
                path,
                format!("affine column {axis} has invalid norm {norm}"),
            ));
        }
        spacing[axis] = norm;
        for row in 0..3 {
            direction[row][axis] = affine[row][axis] / norm;
        }
    }
    VolumeGeometry::new(
        shape,
        spacing,
        [affine[0][3], affine[1][3], affine[2][3]],
        direction,
    )
    .map_err(Into::into)
}

fn read_pixels<T>(
    path: &Path,
    bytes: &[u8],
    header: &Header,
    convert: impl Fn(f64) -> T,
) -> Result<Vec<T>> {
    let count = header.shape[0] * header.shape[1] * header.shape[2];
    let bytes_per_value = bytes_per_value(header.datatype);
    let end = header.vox_offset + count * bytes_per_value;
    if bytes.len() < end {
        return Err(CacheError::nifti(
            path,
            format!(
                "pixel data ends at byte {end}, but file has {} bytes",
                bytes.len()
            ),
        ));
    }
    let mut out = Vec::with_capacity(count);
    let pixel_bytes = &bytes[header.vox_offset..end];
    for chunk in pixel_bytes.chunks_exact(bytes_per_value) {
        out.push(convert(read_value(chunk, header.datatype, header.endian)));
    }
    Ok(out)
}

fn bytes_per_value(kind: PixelKind) -> usize {
    match kind {
        PixelKind::U8 | PixelKind::I8 => 1,
        PixelKind::I16 | PixelKind::U16 => 2,
        PixelKind::I32 | PixelKind::F32 | PixelKind::U32 => 4,
        PixelKind::F64 => 8,
    }
}

fn read_value(bytes: &[u8], kind: PixelKind, endian: Endian) -> f64 {
    match kind {
        PixelKind::U8 => bytes[0] as f64,
        PixelKind::I8 => i8::from_ne_bytes([bytes[0]]) as f64,
        PixelKind::I16 => i16_from(bytes, endian) as f64,
        PixelKind::U16 => u16_from(bytes, endian) as f64,
        PixelKind::I32 => i32_from(bytes, endian) as f64,
        PixelKind::U32 => u32_from(bytes, endian) as f64,
        PixelKind::F32 => f32_from(bytes, endian) as f64,
        PixelKind::F64 => f64_from(bytes, endian),
    }
}

fn i16_from(bytes: &[u8], endian: Endian) -> i16 {
    let value = [bytes[0], bytes[1]];
    match endian {
        Endian::Little => i16::from_le_bytes(value),
        Endian::Big => i16::from_be_bytes(value),
    }
}

fn u16_from(bytes: &[u8], endian: Endian) -> u16 {
    let value = [bytes[0], bytes[1]];
    match endian {
        Endian::Little => u16::from_le_bytes(value),
        Endian::Big => u16::from_be_bytes(value),
    }
}

fn i32_from(bytes: &[u8], endian: Endian) -> i32 {
    let value = [bytes[0], bytes[1], bytes[2], bytes[3]];
    match endian {
        Endian::Little => i32::from_le_bytes(value),
        Endian::Big => i32::from_be_bytes(value),
    }
}

fn u32_from(bytes: &[u8], endian: Endian) -> u32 {
    let value = [bytes[0], bytes[1], bytes[2], bytes[3]];
    match endian {
        Endian::Little => u32::from_le_bytes(value),
        Endian::Big => u32::from_be_bytes(value),
    }
}

fn f32_from(bytes: &[u8], endian: Endian) -> f32 {
    let value = [bytes[0], bytes[1], bytes[2], bytes[3]];
    match endian {
        Endian::Little => f32::from_le_bytes(value),
        Endian::Big => f32::from_be_bytes(value),
    }
}

fn f64_from(bytes: &[u8], endian: Endian) -> f64 {
    let value = [
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ];
    match endian {
        Endian::Little => f64::from_le_bytes(value),
        Endian::Big => f64::from_be_bytes(value),
    }
}
