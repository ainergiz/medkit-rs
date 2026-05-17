use std::{
    fs::File,
    io::{self, Read},
    path::{Path, PathBuf},
};

use flate2::read::GzDecoder;
use medkit_transform::{Volume3D, VolumeGeometry};
use sha2::{Digest, Sha256};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NiftiStorage {
    SingleFile,
    Detached,
}

#[derive(Debug, Clone, PartialEq)]
struct Header {
    endian: Endian,
    shape: [usize; 3],
    datatype: PixelKind,
    vox_offset: usize,
    scl_slope: f64,
    scl_inter: f64,
    geometry: VolumeGeometry,
}

#[derive(Debug, Clone)]
struct NiftiSource {
    storage: NiftiStorage,
    header_path: PathBuf,
    header_bytes: Vec<u8>,
    pixel_path: PathBuf,
    pixel_bytes: Vec<u8>,
}

#[derive(Debug)]
pub(crate) struct LoadedVolume<T> {
    pub(crate) volume: Volume3D<T>,
    pub(crate) geometry: VolumeGeometry,
    pub(crate) source_content_hash: String,
}

pub(crate) fn load_image_f32(path: &Path) -> Result<LoadedVolume<f32>> {
    let source = read_source(path)?;
    let source_content_hash = source_content_hash(&source);
    let header = parse_header(path, &source.header_bytes, source.storage)?;
    let values = read_pixels(path, &source.pixel_bytes, &header, |value| value as f32)?;
    Ok(LoadedVolume {
        volume: Volume3D::new(header.shape, values)?,
        geometry: header.geometry,
        source_content_hash,
    })
}

pub(crate) fn load_label_u16(path: &Path) -> Result<LoadedVolume<u16>> {
    let source = read_source(path)?;
    let source_content_hash = source_content_hash(&source);
    let header = parse_header(path, &source.header_bytes, source.storage)?;
    let values = read_pixels(path, &source.pixel_bytes, &header, |value| {
        value.max(0.0).round() as u16
    })?;
    Ok(LoadedVolume {
        volume: Volume3D::new(header.shape, values)?,
        geometry: header.geometry,
        source_content_hash,
    })
}

fn read_source(path: &Path) -> Result<NiftiSource> {
    let header_bytes = read_all(path)?;
    if !is_detached_header_path(path) {
        return Ok(NiftiSource {
            storage: NiftiStorage::SingleFile,
            header_path: path.to_path_buf(),
            pixel_path: path.to_path_buf(),
            pixel_bytes: header_bytes.clone(),
            header_bytes,
        });
    }

    let (pixel_path, pixel_bytes) = read_detached_pixels(path)?;
    Ok(NiftiSource {
        storage: NiftiStorage::Detached,
        header_path: path.to_path_buf(),
        header_bytes,
        pixel_path,
        pixel_bytes,
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

fn read_detached_pixels(header_path: &Path) -> Result<(PathBuf, Vec<u8>)> {
    let candidates = detached_pixel_candidates(header_path);
    let mut missing_paths = Vec::new();
    for candidate in candidates {
        match read_all(&candidate) {
            Ok(bytes) => return Ok((candidate, bytes)),
            Err(CacheError::Io { path, source })
                if matches!(source.kind(), io::ErrorKind::NotFound) =>
            {
                missing_paths.push(path);
            }
            Err(error) => return Err(error),
        }
    }
    let attempted = missing_paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    Err(CacheError::nifti(
        header_path,
        format!("detached NIfTI header requires companion image file; tried {attempted}"),
    ))
}

fn detached_pixel_candidates(header_path: &Path) -> Vec<PathBuf> {
    let Some(file_name) = header_path.file_name().and_then(|value| value.to_str()) else {
        return vec![header_path.with_extension("img")];
    };
    let lower = file_name.to_ascii_lowercase();
    if lower.ends_with(".hdr.gz") {
        let stem = &file_name[..file_name.len() - ".hdr.gz".len()];
        return vec![
            header_path.with_file_name(format!("{stem}.img.gz")),
            header_path.with_file_name(format!("{stem}.img")),
        ];
    }
    if lower.ends_with(".hdr") {
        let stem = &file_name[..file_name.len() - ".hdr".len()];
        return vec![
            header_path.with_file_name(format!("{stem}.img")),
            header_path.with_file_name(format!("{stem}.img.gz")),
        ];
    }
    vec![header_path.with_extension("img")]
}

fn is_detached_header_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|file_name| {
            let lower = file_name.to_ascii_lowercase();
            lower.ends_with(".hdr") || lower.ends_with(".hdr.gz")
        })
}

fn source_content_hash(source: &NiftiSource) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"medkit-nifti-source-v1");
    match source.storage {
        NiftiStorage::SingleFile => {
            hasher.update(b"single-file");
            hash_source_part(
                &mut hasher,
                b"nifti",
                &source.header_path,
                &source.header_bytes,
            );
        }
        NiftiStorage::Detached => {
            hasher.update(b"detached");
            hash_source_part(
                &mut hasher,
                b"header",
                &source.header_path,
                &source.header_bytes,
            );
            hash_source_part(
                &mut hasher,
                b"pixels",
                &source.pixel_path,
                &source.pixel_bytes,
            );
        }
    }
    format!("{:x}", hasher.finalize())
}

fn hash_source_part(hasher: &mut Sha256, role: &[u8], path: &Path, bytes: &[u8]) {
    hasher.update(role);
    hasher.update(path.to_string_lossy().as_bytes());
    hasher.update(bytes.len().to_le_bytes());
    hasher.update(bytes);
}

fn parse_header(path: &Path, bytes: &[u8], storage: NiftiStorage) -> Result<Header> {
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
    let mut dims = [1_usize; 7];
    for axis in 0..rank as usize {
        let dim = i16_from(&bytes[42 + axis * 2..44 + axis * 2], endian);
        if dim <= 0 {
            return Err(CacheError::nifti(
                path,
                format!("dimension {} must be positive", axis + 1),
            ));
        }
        dims[axis] = dim as usize;
    }
    if let Some((axis, dim)) = dims[..rank as usize]
        .iter()
        .enumerate()
        .skip(3)
        .find(|(_axis, dim)| **dim != 1)
    {
        return Err(CacheError::nifti(
            path,
            format!(
                "NIfTI rank {rank} with non-singleton dimension {}={dim} is not supported by the 3D cache loader",
                axis + 1
            ),
        ));
    }
    let shape = [dims[0], dims[1], dims[2]];
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
    let raw_vox_offset = f32_from(&bytes[108..112], endian);
    if !raw_vox_offset.is_finite() || raw_vox_offset < 0.0 {
        return Err(CacheError::nifti(
            path,
            format!("vox_offset must be finite and non-negative, got {raw_vox_offset}"),
        ));
    }
    let vox_offset = match storage {
        NiftiStorage::SingleFile => raw_vox_offset.max(352.0) as usize,
        NiftiStorage::Detached => raw_vox_offset as usize,
    };
    let raw_slope = f64::from(f32_from(&bytes[112..116], endian));
    let raw_inter = f64::from(f32_from(&bytes[116..120], endian));
    let (scl_slope, scl_inter) = if raw_slope == 0.0 || !raw_slope.is_finite() {
        (1.0, 0.0)
    } else if !raw_inter.is_finite() {
        return Err(CacheError::nifti(
            path,
            format!("scl_inter must be finite when scl_slope is set, got {raw_inter}"),
        ));
    } else {
        (raw_slope, raw_inter)
    };
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
        scl_slope,
        scl_inter,
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
        let raw = read_value(chunk, header.datatype, header.endian);
        out.push(convert(raw * header.scl_slope + header.scl_inter));
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

#[cfg(test)]
mod tests {
    use std::{
        fs,
        io::Write,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    use flate2::{write::GzEncoder, Compression};

    use super::*;

    const VOX_OFFSET: usize = 352;

    #[derive(Debug, Clone)]
    struct NiftiFixture {
        bytes: Vec<u8>,
        endian: Endian,
    }

    impl NiftiFixture {
        fn new(endian: Endian, dims: &[i16], datatype: i16, pixdim: &[f32]) -> Self {
            let mut fixture = Self {
                bytes: vec![0; VOX_OFFSET],
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
            fixture.put_f32(108, VOX_OFFSET as f32);
            fixture.bytes[344..348].copy_from_slice(b"n+1\0");
            fixture
        }

        fn with_sizeof_hdr(mut self, sizeof_hdr: i32) -> Self {
            self.put_i32(0, sizeof_hdr);
            self
        }

        fn with_rank(mut self, rank: i16) -> Self {
            self.put_i16(40, rank);
            self
        }

        fn with_dim(mut self, axis: usize, dim: i16) -> Self {
            self.put_i16(42 + axis * 2, dim);
            self
        }

        fn with_pixdim(mut self, index: usize, value: f32) -> Self {
            self.put_f32(76 + index * 4, value);
            self
        }

        fn with_vox_offset(mut self, vox_offset: f32) -> Self {
            self.put_f32(108, vox_offset);
            self
        }

        fn with_scaling(mut self, slope: f32, intercept: f32) -> Self {
            self.put_f32(112, slope);
            self.put_f32(116, intercept);
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

        fn append_f32_pixels(mut self, values: &[f32]) -> Vec<u8> {
            for value in values {
                let bytes = match self.endian {
                    Endian::Little => value.to_le_bytes(),
                    Endian::Big => value.to_be_bytes(),
                };
                self.bytes.extend_from_slice(&bytes);
            }
            self.bytes
        }

        fn append_u16_pixels(mut self, values: &[u16]) -> Vec<u8> {
            for value in values {
                let bytes = match self.endian {
                    Endian::Little => value.to_le_bytes(),
                    Endian::Big => value.to_be_bytes(),
                };
                self.bytes.extend_from_slice(&bytes);
            }
            self.bytes
        }

        fn bytes(&self) -> Vec<u8> {
            self.bytes.clone()
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

    #[test]
    fn loads_gzipped_image_and_converts_label_values() {
        let dir = temp_dir("load-pixels");
        let image_path = dir.join("image.NII.GZ");
        let label_path = dir.join("label.nii");
        write_gzip(
            &image_path,
            &NiftiFixture::new(Endian::Little, &[2, 2, 1], 16, &[1.5, 2.5, 3.5])
                .append_f32_pixels(&[1.25, -2.5, 3.0, 4.5]),
        );
        fs::write(
            &label_path,
            NiftiFixture::new(Endian::Little, &[2, 2, 1], 16, &[1.5, 2.5, 3.5])
                .append_f32_pixels(&[-1.0, 1.2, 1.6, 2.5]),
        )
        .unwrap();
        let big_image_path = dir.join("big-image.nii");
        fs::write(
            &big_image_path,
            NiftiFixture::new(Endian::Big, &[1, 1, 1], 16, &[1.0, 1.0, 1.0])
                .append_f32_pixels(&[7.5]),
        )
        .unwrap();

        let image = load_image_f32(&image_path).unwrap();
        assert_eq!(image.volume.shape, [2, 2, 1]);
        assert_eq!(image.volume.data, vec![1.25, -2.5, 3.0, 4.5]);
        assert_eq!(image.geometry.spacing, [1.5, 2.5, 3.5]);

        let label = load_label_u16(&label_path).unwrap();
        assert_eq!(label.volume.shape, [2, 2, 1]);
        assert_eq!(label.volume.data, vec![0, 1, 2, 3]);
        assert!(image.geometry.approximately_eq(&label.geometry, 1e-6));

        let big_image = load_image_f32(&big_image_path).unwrap();
        assert_eq!(big_image.volume.data, vec![7.5]);
    }

    #[test]
    fn applies_nifti_scaling_and_rejects_unsupported_4d_volumes() {
        let dir = temp_dir("scaling-and-4d");
        let scaled_path = dir.join("scaled.nii");
        fs::write(
            &scaled_path,
            NiftiFixture::new(Endian::Little, &[3, 1, 1], 512, &[1.0, 1.0, 1.0])
                .with_scaling(2.0, -10.0)
                .append_u16_pixels(&[5, 6, 7]),
        )
        .unwrap();
        let scaled = load_image_f32(&scaled_path).unwrap();
        assert_eq!(scaled.volume.data, vec![0.0, 2.0, 4.0]);

        let unsupported_path = dir.join("four-dimensional.nii");
        fs::write(
            &unsupported_path,
            NiftiFixture::new(Endian::Little, &[2, 2, 2, 2], 16, &[1.0, 1.0, 1.0, 1.0])
                .append_f32_pixels(&[0.0; 16]),
        )
        .unwrap();
        let err = load_image_f32(&unsupported_path).unwrap_err();
        assert!(err.to_string().contains("non-singleton dimension 4=2"));

        let singleton_4d_path = dir.join("singleton-four-dimensional.nii");
        fs::write(
            &singleton_4d_path,
            NiftiFixture::new(Endian::Little, &[2, 1, 1, 1], 16, &[1.0, 1.0, 1.0, 1.0])
                .append_f32_pixels(&[8.0, 9.0]),
        )
        .unwrap();
        let singleton = load_image_f32(&singleton_4d_path).unwrap();
        assert_eq!(singleton.volume.shape, [2, 1, 1]);
        assert_eq!(singleton.volume.data, vec![8.0, 9.0]);
    }

    #[test]
    fn detached_hdr_reads_companion_img_bytes_and_hashes_both_files() {
        let dir = temp_dir("detached-hdr");
        let header_path = dir.join("case.hdr");
        let image_path = dir.join("case.img");
        fs::write(
            &header_path,
            NiftiFixture::new(Endian::Little, &[2, 1, 1], 16, &[1.0, 1.0, 1.0])
                .with_vox_offset(0.0)
                .bytes(),
        )
        .unwrap();
        fs::write(
            &image_path,
            [3.0_f32.to_le_bytes(), 4.0_f32.to_le_bytes()].concat(),
        )
        .unwrap();

        let first = load_image_f32(&header_path).unwrap();
        assert_eq!(first.volume.shape, [2, 1, 1]);
        assert_eq!(first.volume.data, vec![3.0, 4.0]);

        fs::write(
            &image_path,
            [3.0_f32.to_le_bytes(), 5.0_f32.to_le_bytes()].concat(),
        )
        .unwrap();
        let second = load_image_f32(&header_path).unwrap();
        assert_eq!(second.volume.data, vec![3.0, 5.0]);
        assert_ne!(first.source_content_hash, second.source_content_hash);
    }

    #[test]
    fn parses_big_endian_sform_and_little_endian_qform_geometry() {
        let sform_fixture = NiftiFixture::new(Endian::Big, &[2, 3, 4], 512, &[1.0, 1.0, 1.0])
            .with_sform([
                [2.0, 0.0, 0.0, 10.0],
                [0.0, 3.0, 0.0, 20.0],
                [0.0, 0.0, -4.0, 30.0],
            ]);
        let sform = parse_header(
            Path::new("sform.nii"),
            &sform_fixture.bytes(),
            NiftiStorage::SingleFile,
        )
        .unwrap();
        assert_eq!(sform.endian, Endian::Big);
        assert_eq!(sform.shape, [2, 3, 4]);
        assert_eq!(sform.datatype, PixelKind::U16);
        assert_eq!(sform.geometry.spacing, [2.0, 3.0, 4.0]);
        assert_eq!(sform.geometry.origin, [10.0, 20.0, 30.0]);
        assert_close(sform.geometry.direction[2][2], -1.0);

        let qform_fixture = NiftiFixture::new(Endian::Little, &[2, 2, 2], 2, &[1.0, 2.0, 3.0])
            .with_qform([0.0, 0.0, 0.0], [10.0, 20.0, -5.0], -1.0);
        let qform = parse_header(
            Path::new("qform.nii"),
            &qform_fixture.bytes(),
            NiftiStorage::SingleFile,
        )
        .unwrap();
        assert_eq!(qform.geometry.spacing, [1.0, 2.0, 3.0]);
        assert_eq!(qform.geometry.origin, [10.0, 20.0, -5.0]);
        assert_close(qform.geometry.direction[2][2], -1.0);
    }

    #[test]
    fn rejects_invalid_headers_with_specific_reasons() {
        let path = Path::new("bad.nii");
        let err = parse_header(path, &[0; 16], NiftiStorage::SingleFile).unwrap_err();
        assert!(err.to_string().contains("shorter than a NIfTI-1 header"));

        assert_nifti_error(
            NiftiFixture::new(Endian::Little, &[2, 2, 2], 16, &[1.0, 1.0, 1.0])
                .with_sizeof_hdr(123)
                .bytes(),
            "sizeof_hdr must be 348",
        );
        assert_nifti_error(
            NiftiFixture::new(Endian::Little, &[2, 2, 2], 16, &[1.0, 1.0, 1.0])
                .with_rank(0)
                .bytes(),
            "rank must be between 1 and 7",
        );
        assert_nifti_error(
            NiftiFixture::new(Endian::Little, &[2, 2, 2], 16, &[1.0, 1.0, 1.0])
                .with_dim(1, 0)
                .bytes(),
            "dimension 2 must be positive",
        );
        assert_nifti_error(
            NiftiFixture::new(Endian::Little, &[2, 2, 2], 16, &[1.0, 1.0, 1.0])
                .with_pixdim(2, f32::NAN)
                .bytes(),
            "pixdim[2] must be finite and positive",
        );
        assert_nifti_error(
            NiftiFixture::new(Endian::Little, &[2, 2, 2], 128, &[1.0, 1.0, 1.0]).bytes(),
            "unsupported datatype code 128",
        );
        assert_nifti_error(
            NiftiFixture::new(Endian::Little, &[2, 2, 2], 16, &[1.0, 1.0, 1.0])
                .with_qform([1.0, 1.0, 0.0], [0.0, 0.0, 0.0], 1.0)
                .bytes(),
            "qform quaternion has magnitude greater than one",
        );
        assert_nifti_error(
            NiftiFixture::new(Endian::Little, &[2, 2, 2], 16, &[1.0, 1.0, 1.0])
                .with_sform([[0.0; 4]; 3])
                .bytes(),
            "affine column 0 has invalid norm",
        );
    }

    #[test]
    fn decodes_datatypes_and_rejects_truncated_pixel_data() {
        for (datatype_code, expected) in [
            (4, PixelKind::I16),
            (8, PixelKind::I32),
            (64, PixelKind::F64),
            (256, PixelKind::I8),
            (768, PixelKind::U32),
        ] {
            let header = parse_header(
                Path::new("datatype.nii"),
                &NiftiFixture::new(Endian::Little, &[1, 1, 1], datatype_code, &[1.0, 1.0, 1.0])
                    .bytes(),
                NiftiStorage::SingleFile,
            )
            .unwrap();
            assert_eq!(header.datatype, expected);
        }

        assert_eq!(bytes_per_value(PixelKind::U8), 1);
        assert_eq!(bytes_per_value(PixelKind::I16), 2);
        assert_eq!(bytes_per_value(PixelKind::I32), 4);
        assert_eq!(bytes_per_value(PixelKind::F64), 8);

        assert_eq!(read_value(&[255], PixelKind::U8, Endian::Little), 255.0);
        assert_eq!(read_value(&[254], PixelKind::I8, Endian::Little), -2.0);
        assert_eq!(
            read_value(&(-1234_i16).to_be_bytes(), PixelKind::I16, Endian::Big),
            -1234.0
        );
        assert_eq!(
            read_value(&54321_u16.to_le_bytes(), PixelKind::U16, Endian::Little),
            54321.0
        );
        assert_eq!(
            read_value(&12345_u16.to_be_bytes(), PixelKind::U16, Endian::Big),
            12345.0
        );
        assert_eq!(
            read_value(&(-123456_i32).to_le_bytes(), PixelKind::I32, Endian::Little),
            -123456.0
        );
        assert_eq!(
            read_value(&123456_u32.to_le_bytes(), PixelKind::U32, Endian::Little),
            123456.0
        );
        assert_eq!(
            read_value(&123456_u32.to_be_bytes(), PixelKind::U32, Endian::Big),
            123456.0
        );
        assert_close(
            read_value(&1.25_f32.to_be_bytes(), PixelKind::F32, Endian::Big),
            1.25,
        );
        assert_close(
            read_value(&(-2.5_f64).to_le_bytes(), PixelKind::F64, Endian::Little),
            -2.5,
        );
        assert_close(
            read_value(&3.5_f64.to_be_bytes(), PixelKind::F64, Endian::Big),
            3.5,
        );

        let header = Header {
            endian: Endian::Little,
            shape: [2, 1, 1],
            datatype: PixelKind::U16,
            vox_offset: VOX_OFFSET,
            scl_slope: 1.0,
            scl_inter: 0.0,
            geometry: VolumeGeometry::identity([2, 1, 1], [1.0, 1.0, 1.0]).unwrap(),
        };
        let err = read_pixels(
            Path::new("truncated.nii"),
            &[0; VOX_OFFSET + 1],
            &header,
            |v| v,
        )
        .unwrap_err();
        assert!(err.to_string().contains("pixel data ends at byte 356"));
    }

    #[test]
    fn read_all_reports_file_and_gzip_io_errors() {
        let dir = temp_dir("read-errors");
        let missing = dir.join("missing.nii");
        let err = read_all(&missing).unwrap_err();
        assert!(matches!(err, CacheError::Io { path, .. } if path == missing));

        let bad_gzip = dir.join("bad.nii.gz");
        fs::write(&bad_gzip, b"not gzip").unwrap();
        let err = read_all(&bad_gzip).unwrap_err();
        assert!(matches!(err, CacheError::Io { path, .. } if path == bad_gzip));
    }

    fn assert_nifti_error(bytes: Vec<u8>, expected: &str) {
        let err = parse_header(Path::new("bad.nii"), &bytes, NiftiStorage::SingleFile).unwrap_err();
        assert!(
            err.to_string().contains(expected),
            "expected {expected:?}, got {err}"
        );
    }

    fn bitpix_for(datatype: i16) -> i16 {
        match datatype {
            2 | 256 => 8,
            4 | 512 => 16,
            8 | 16 | 768 => 32,
            64 => 64,
            _ => 0,
        }
    }

    fn temp_dir(case: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "medkit-cache-nifti-{case}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_gzip(path: &Path, bytes: &[u8]) {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(bytes).unwrap();
        fs::write(path, encoder.finish().unwrap()).unwrap();
    }

    fn assert_close(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-6, "{left} != {right}");
    }
}
