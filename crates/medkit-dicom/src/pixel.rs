use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    parser::DicomDataSet,
    types::{
        EXPLICIT_VR_BIG_ENDIAN, EXPLICIT_VR_LITTLE_ENDIAN, IMPLICIT_VR_LITTLE_ENDIAN, RLE_LOSSLESS,
    },
    DicomError, Result,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PixelExplanation {
    pub path: String,
    pub width: usize,
    pub height: usize,
    pub transfer_syntax_uid: String,
    pub photometric_interpretation: String,
    pub source_pixel_hash: String,
    pub presented_pixel_hash: String,
    pub decoder_backend: String,
    pub decoder_version: String,
    pub compressed: bool,
    pub min_value: f32,
    pub max_value: f32,
    pub steps: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PresentedImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
    pub explanation: PixelExplanation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedPixelData {
    pub bytes: Vec<u8>,
    pub source_pixel_hash: String,
    pub backend: String,
    pub backend_version: String,
    pub compressed: bool,
    pub steps: Vec<String>,
}

pub trait DecoderBackend {
    fn name(&self) -> &'static str;
    fn version(&self) -> &'static str;
    fn supports_transfer_syntax(&self, transfer_syntax_uid: &str) -> bool;
    fn decode_pixels(
        &self,
        dataset: &DicomDataSet,
        expected_bytes: usize,
    ) -> Result<DecodedPixelData>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NativeDecoderBackend;

impl DecoderBackend for NativeDecoderBackend {
    fn name(&self) -> &'static str {
        "medkit-native"
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    fn supports_transfer_syntax(&self, transfer_syntax_uid: &str) -> bool {
        matches!(
            transfer_syntax_uid,
            EXPLICIT_VR_LITTLE_ENDIAN
                | IMPLICIT_VR_LITTLE_ENDIAN
                | EXPLICIT_VR_BIG_ENDIAN
                | RLE_LOSSLESS
        )
    }

    fn decode_pixels(
        &self,
        dataset: &DicomDataSet,
        expected_bytes: usize,
    ) -> Result<DecodedPixelData> {
        let pixel_data = dataset
            .element((0x7FE0, 0x0010))
            .ok_or_else(|| DicomError::parse(&dataset.path, "missing PixelData"))?;
        if !self.supports_transfer_syntax(&dataset.transfer_syntax_uid) {
            return Err(DicomError::unsupported(
                &dataset.path,
                format!(
                    "unsupported transfer syntax {}",
                    dataset.transfer_syntax_uid
                ),
            ));
        }
        let source_pixel_hash = sha256_hex(pixel_data);
        if dataset.transfer_syntax_uid == RLE_LOSSLESS {
            let bits_allocated = dataset.u16_value((0x0028, 0x0100)).unwrap_or(8);
            let samples_per_pixel = dataset.u16_value((0x0028, 0x0002)).unwrap_or(1);
            let bytes = decode_rle_lossless(
                pixel_data,
                expected_bytes,
                bits_allocated,
                samples_per_pixel,
                &dataset.path,
            )?;
            return Ok(DecodedPixelData {
                bytes,
                source_pixel_hash,
                backend: self.name().to_string(),
                backend_version: self.version().to_string(),
                compressed: true,
                steps: vec![
                    format!("decode transfer syntax {}", dataset.transfer_syntax_uid),
                    "decode RLE Lossless pixel segment".to_string(),
                ],
            });
        }
        if pixel_data.len() != expected_bytes {
            return Err(DicomError::parse(
                &dataset.path,
                format!(
                    "PixelData length mismatch: expected {expected_bytes} bytes, got {}",
                    pixel_data.len()
                ),
            ));
        }
        Ok(DecodedPixelData {
            bytes: pixel_data.to_vec(),
            source_pixel_hash,
            backend: self.name().to_string(),
            backend_version: self.version().to_string(),
            compressed: false,
            steps: vec![format!(
                "decode transfer syntax {}",
                dataset.transfer_syntax_uid
            )],
        })
    }
}

pub fn explain_pixels(path: impl AsRef<Path>) -> Result<PixelExplanation> {
    Ok(present_dicom_pixels(path)?.explanation)
}

pub fn present_dicom_pixels(path: impl AsRef<Path>) -> Result<PresentedImage> {
    let dataset = DicomDataSet::from_file(path.as_ref())?;
    present_dataset_pixels(&dataset)
}

pub fn present_dicom_pixels_with_backend(
    path: impl AsRef<Path>,
    backend: &dyn DecoderBackend,
) -> Result<PresentedImage> {
    let dataset = DicomDataSet::from_file(path.as_ref())?;
    present_dataset_pixels_with_backend(&dataset, backend)
}

pub(crate) fn present_dataset_pixels(dataset: &DicomDataSet) -> Result<PresentedImage> {
    present_dataset_pixels_with_backend(dataset, &NativeDecoderBackend)
}

pub(crate) fn present_dataset_pixels_with_backend(
    dataset: &DicomDataSet,
    backend: &dyn DecoderBackend,
) -> Result<PresentedImage> {
    let rows = required_u16(dataset, (0x0028, 0x0010), "Rows")? as usize;
    let columns = required_u16(dataset, (0x0028, 0x0011), "Columns")? as usize;
    if rows == 0 || columns == 0 {
        return Err(DicomError::parse(
            &dataset.path,
            "Rows and Columns must be greater than zero",
        ));
    }
    let samples_per_pixel = dataset.u16_value((0x0028, 0x0002)).unwrap_or(1);
    if samples_per_pixel != 1 {
        return Err(DicomError::unsupported(
            &dataset.path,
            format!("only single-sample grayscale pixels are supported, got {samples_per_pixel}"),
        ));
    }
    let bits_allocated = required_u16(dataset, (0x0028, 0x0100), "BitsAllocated")?;
    let bits_stored = dataset
        .u16_value((0x0028, 0x0101))
        .unwrap_or(bits_allocated);
    if bits_stored == 0 || bits_stored > bits_allocated {
        return Err(DicomError::parse(
            &dataset.path,
            format!("invalid BitsStored {bits_stored} for BitsAllocated {bits_allocated}"),
        ));
    }
    let signed = dataset.u16_value((0x0028, 0x0103)).unwrap_or(0) != 0;
    let expected = rows * columns * (bits_allocated as usize).div_ceil(8);
    let decoded = backend.decode_pixels(dataset, expected)?;

    let mut steps = decoded.steps.clone();
    steps.push(format!(
        "unpack {bits_allocated}-bit pixels with {bits_stored} stored bits"
    ));
    let mut values = unpack_pixels(
        &decoded.bytes,
        bits_allocated,
        bits_stored,
        signed,
        dataset.pixel_is_big_endian(),
        &dataset.path,
    )?;
    let slope = dataset.f32_value((0x0028, 0x1053)).unwrap_or(1.0);
    let intercept = dataset.f32_value((0x0028, 0x1052)).unwrap_or(0.0);
    if slope != 1.0 || intercept != 0.0 {
        steps.push(format!("apply rescale slope={slope} intercept={intercept}"));
    }
    for value in &mut values {
        *value = *value * slope + intercept;
    }
    let min_value = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max_value = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut pixels = if let (Some(center), Some(width)) = (
        dataset.f32_value((0x0028, 0x1050)),
        dataset.f32_value((0x0028, 0x1051)),
    ) {
        steps.push(format!("apply window center={center} width={width}"));
        window_pixels(&values, center, width)
    } else {
        steps.push("scale min/max to u8".to_string());
        min_max_pixels(&values, min_value, max_value)
    };
    let photometric = dataset
        .string((0x0028, 0x0004))
        .unwrap_or_else(|| "MONOCHROME2".to_string());
    if photometric.eq_ignore_ascii_case("MONOCHROME1") {
        steps.push("invert MONOCHROME1 to MONOCHROME2".to_string());
        for value in &mut pixels {
            *value = 255u8.saturating_sub(*value);
        }
    }
    steps.push("output canonical MONOCHROME2 u8 raster".to_string());
    let explanation = PixelExplanation {
        path: dataset.path.display().to_string(),
        width: columns,
        height: rows,
        transfer_syntax_uid: dataset.transfer_syntax_uid.clone(),
        photometric_interpretation: photometric,
        source_pixel_hash: decoded.source_pixel_hash,
        presented_pixel_hash: sha256_hex(&pixels),
        decoder_backend: decoded.backend,
        decoder_version: decoded.backend_version,
        compressed: decoded.compressed,
        min_value,
        max_value,
        steps,
    };
    Ok(PresentedImage {
        width: columns,
        height: rows,
        pixels,
        explanation,
    })
}

fn decode_rle_lossless(
    pixel_data: &[u8],
    expected_bytes: usize,
    bits_allocated: u16,
    samples_per_pixel: u16,
    path: &Path,
) -> Result<Vec<u8>> {
    if pixel_data.len() < 64 {
        return Err(DicomError::parse(
            path,
            "RLE Lossless PixelData is shorter than the 64-byte RLE header",
        ));
    }
    let segments =
        u32::from_le_bytes([pixel_data[0], pixel_data[1], pixel_data[2], pixel_data[3]]) as usize;
    if segments == 0 || segments > 15 {
        return Err(DicomError::parse(
            path,
            format!("invalid RLE segment count {segments}"),
        ));
    }
    let bytes_per_sample = (bits_allocated as usize).div_ceil(8);
    let expected_segments = bytes_per_sample
        .checked_mul(samples_per_pixel as usize)
        .ok_or_else(|| DicomError::parse(path, "RLE segment count overflow"))?;
    if segments != expected_segments {
        return Err(DicomError::unsupported(
            path,
            format!(
                "RLE Lossless expected {expected_segments} segment(s) for {bits_allocated}-bit {}-sample pixels, got {segments}",
                samples_per_pixel
            ),
        ));
    }
    let pixels = expected_bytes
        .checked_div(bytes_per_sample.max(1))
        .ok_or_else(|| DicomError::parse(path, "invalid RLE expected byte count"))?;
    let mut offsets = Vec::with_capacity(segments);
    for segment in 0..segments {
        let header_offset = 4 + segment * 4;
        let offset = u32::from_le_bytes([
            pixel_data[header_offset],
            pixel_data[header_offset + 1],
            pixel_data[header_offset + 2],
            pixel_data[header_offset + 3],
        ]) as usize;
        if offset < 64 || offset > pixel_data.len() {
            return Err(DicomError::parse(
                path,
                format!("invalid RLE segment offset {offset}"),
            ));
        }
        offsets.push(offset);
    }
    let mut planes = Vec::with_capacity(segments);
    for (index, offset) in offsets.iter().copied().enumerate() {
        let end = offsets.get(index + 1).copied().unwrap_or(pixel_data.len());
        if end < offset {
            return Err(DicomError::parse(
                path,
                "RLE segment offsets must be increasing",
            ));
        }
        let plane = decode_packbits_segment(&pixel_data[offset..end], pixels, path)?;
        if plane.len() != pixels {
            return Err(DicomError::parse(
                path,
                format!(
                    "RLE decoded segment length mismatch: expected {pixels} bytes, got {}",
                    plane.len()
                ),
            ));
        }
        planes.push(plane);
    }
    let mut decoded = Vec::with_capacity(expected_bytes);
    for pixel in 0..pixels {
        for byte_index in (0..bytes_per_sample).rev() {
            decoded.push(planes[byte_index][pixel]);
        }
    }
    if decoded.len() != expected_bytes {
        return Err(DicomError::parse(
            path,
            format!(
                "RLE decoded length mismatch: expected {expected_bytes} bytes, got {}",
                decoded.len()
            ),
        ));
    }
    Ok(decoded)
}

fn decode_packbits_segment(segment: &[u8], expected_bytes: usize, path: &Path) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(expected_bytes);
    let mut cursor = 0usize;
    while cursor < segment.len() && out.len() < expected_bytes {
        let control = segment[cursor] as i8;
        cursor += 1;
        match control {
            0..=127 => {
                let len = control as usize + 1;
                if cursor + len > segment.len() {
                    return Err(DicomError::parse(path, "truncated RLE literal run"));
                }
                out.extend_from_slice(&segment[cursor..cursor + len]);
                cursor += len;
            }
            -127..=-1 => {
                if cursor >= segment.len() {
                    return Err(DicomError::parse(path, "truncated RLE replicate run"));
                }
                let len = 1usize + (-control as usize);
                out.extend(std::iter::repeat(segment[cursor]).take(len));
                cursor += 1;
            }
            -128 => {}
        }
    }
    Ok(out)
}

fn required_u16(dataset: &DicomDataSet, tag: (u16, u16), name: &str) -> Result<u16> {
    dataset
        .u16_value(tag)
        .ok_or_else(|| DicomError::parse(&dataset.path, format!("missing or invalid {name}")))
}

fn unpack_pixels(
    pixel_data: &[u8],
    bits_allocated: u16,
    bits_stored: u16,
    signed: bool,
    big_endian: bool,
    path: &Path,
) -> Result<Vec<f32>> {
    match bits_allocated {
        8 => Ok(pixel_data
            .iter()
            .map(|value| sign_or_mask(*value as u32, bits_stored, signed) as f32)
            .collect()),
        16 => Ok(pixel_data
            .chunks_exact(2)
            .map(|chunk| {
                let raw = if big_endian {
                    u16::from_be_bytes([chunk[0], chunk[1]])
                } else {
                    u16::from_le_bytes([chunk[0], chunk[1]])
                };
                sign_or_mask(raw as u32, bits_stored, signed) as f32
            })
            .collect()),
        other => Err(DicomError::unsupported(
            path,
            format!("unsupported BitsAllocated {other}; expected 8 or 16"),
        )),
    }
}

fn sign_or_mask(raw: u32, bits_stored: u16, signed: bool) -> i32 {
    let mask = if bits_stored >= 32 {
        u32::MAX
    } else {
        (1u32 << bits_stored) - 1
    };
    let value = raw & mask;
    if signed && bits_stored > 0 {
        let sign_bit = 1u32 << (bits_stored - 1);
        if value & sign_bit != 0 {
            return (value as i32) - (1i32 << bits_stored);
        }
    }
    value as i32
}

fn min_max_pixels(values: &[f32], min_value: f32, max_value: f32) -> Vec<u8> {
    let span = (max_value - min_value).max(1.0e-6);
    values
        .iter()
        .map(|value| {
            (((*value - min_value) / span) * 255.0)
                .clamp(0.0, 255.0)
                .round() as u8
        })
        .collect()
}

fn window_pixels(values: &[f32], center: f32, width: f32) -> Vec<u8> {
    if width <= 1.0 {
        return values
            .iter()
            .map(|value| if *value > center { 255 } else { 0 })
            .collect();
    }
    let low = center - 0.5 - (width - 1.0) / 2.0;
    let high = center - 0.5 + (width - 1.0) / 2.0;
    values
        .iter()
        .map(|value| {
            if *value <= low {
                0
            } else if *value > high {
                255
            } else {
                (((*value - (center - 0.5)) / (width - 1.0) + 0.5) * 255.0)
                    .clamp(0.0, 255.0)
                    .round() as u8
            }
        })
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        path::{Path, PathBuf},
    };

    use crate::{
        parser::DicomElement,
        types::{EXPLICIT_VR_LITTLE_ENDIAN, RLE_LOSSLESS},
    };

    use super::*;

    #[test]
    fn presentation_rejects_invalid_dimensions_bits_and_missing_pixels() {
        let zero_rows = dataset_with([
            u16_element((0x0028, 0x0010), 0),
            u16_element((0x0028, 0x0011), 1),
            u16_element((0x0028, 0x0100), 8),
            u16_element((0x0028, 0x0101), 8),
            u16_element((0x7FE0, 0x0010), 0),
        ]);
        assert!(present_dataset_pixels(&zero_rows)
            .unwrap_err()
            .to_string()
            .contains("Rows and Columns"));

        let bad_bits = dataset_with([
            u16_element((0x0028, 0x0010), 1),
            u16_element((0x0028, 0x0011), 1),
            u16_element((0x0028, 0x0100), 8),
            u16_element((0x0028, 0x0101), 9),
            bytes_element((0x7FE0, 0x0010), vec![0]),
        ]);
        assert!(present_dataset_pixels(&bad_bits)
            .unwrap_err()
            .to_string()
            .contains("invalid BitsStored"));

        let missing_rows = dataset_with([
            u16_element((0x0028, 0x0011), 1),
            u16_element((0x0028, 0x0100), 8),
            bytes_element((0x7FE0, 0x0010), vec![0]),
        ]);
        assert!(present_dataset_pixels(&missing_rows)
            .unwrap_err()
            .to_string()
            .contains("missing or invalid Rows"));

        let missing_columns = dataset_with([
            u16_element((0x0028, 0x0010), 1),
            u16_element((0x0028, 0x0100), 8),
            bytes_element((0x7FE0, 0x0010), vec![0]),
        ]);
        assert!(present_dataset_pixels(&missing_columns)
            .unwrap_err()
            .to_string()
            .contains("missing or invalid Columns"));

        let missing_bits_allocated = dataset_with([
            u16_element((0x0028, 0x0010), 1),
            u16_element((0x0028, 0x0011), 1),
            bytes_element((0x7FE0, 0x0010), vec![0]),
        ]);
        assert!(present_dataset_pixels(&missing_bits_allocated)
            .unwrap_err()
            .to_string()
            .contains("missing or invalid BitsAllocated"));

        let missing_pixel_data = dataset_with([
            u16_element((0x0028, 0x0010), 1),
            u16_element((0x0028, 0x0011), 1),
            u16_element((0x0028, 0x0100), 8),
        ]);
        assert!(present_dataset_pixels(&missing_pixel_data)
            .unwrap_err()
            .to_string()
            .contains("missing PixelData"));
    }

    #[test]
    fn file_pixel_entrypoints_propagate_read_errors() {
        let missing = std::env::temp_dir().join(format!(
            "medkit-dicom-missing-pixels-{}-{}.dcm",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        assert!(present_dicom_pixels(&missing)
            .unwrap_err()
            .to_string()
            .contains("No such file"));
        assert!(explain_pixels(&missing)
            .unwrap_err()
            .to_string()
            .contains("No such file"));
        assert!(
            present_dicom_pixels_with_backend(&missing, &NativeDecoderBackend)
                .unwrap_err()
                .to_string()
                .contains("No such file")
        );
    }

    #[test]
    fn native_backend_decodes_rle_dataset_and_reports_metadata() {
        let dataset = dataset_with_syntax(
            RLE_LOSSLESS,
            [
                u16_element((0x0028, 0x0010), 1),
                u16_element((0x0028, 0x0011), 4),
                u16_element((0x0028, 0x0100), 8),
                u16_element((0x0028, 0x0101), 8),
                bytes_element((0x7FE0, 0x0010), rle_pixel_data(&[2, 4, 6, 8])),
            ],
        );

        let image = present_dataset_pixels(&dataset).unwrap();
        assert_eq!(image.pixels, vec![0, 85, 170, 255]);
        assert!(image.explanation.compressed);
        assert_eq!(image.explanation.decoder_backend, "medkit-native");

        let malformed = dataset_with_syntax(
            RLE_LOSSLESS,
            [
                u16_element((0x0028, 0x0010), 1),
                u16_element((0x0028, 0x0011), 1),
                u16_element((0x0028, 0x0100), 8),
                u16_element((0x0028, 0x0101), 8),
                bytes_element((0x7FE0, 0x0010), vec![0; 10]),
            ],
        );
        assert!(present_dataset_pixels(&malformed)
            .unwrap_err()
            .to_string()
            .contains("shorter than the 64-byte RLE header"));
    }

    #[test]
    fn rle_decoder_reports_malformed_segment_shapes() {
        let path = Path::new("bad-rle.dcm");
        assert!(decode_rle_lossless(&[0; 10], 1, 8, 1, path)
            .unwrap_err()
            .to_string()
            .contains("shorter than the 64-byte RLE header"));

        let mut invalid_count = vec![0; 64];
        invalid_count[..4].copy_from_slice(&0u32.to_le_bytes());
        assert!(decode_rle_lossless(&invalid_count, 1, 8, 1, path)
            .unwrap_err()
            .to_string()
            .contains("invalid RLE segment count 0"));

        let mut mismatch = rle_header(&[64]);
        mismatch.push(0);
        assert!(decode_rle_lossless(&mismatch, 2, 16, 1, path)
            .unwrap_err()
            .to_string()
            .contains("expected 2 segment"));

        let mut bad_offset = rle_header(&[63]);
        bad_offset.push(0);
        assert!(decode_rle_lossless(&bad_offset, 1, 8, 1, path)
            .unwrap_err()
            .to_string()
            .contains("invalid RLE segment offset 63"));

        let mut decreasing = rle_header(&[66, 65]);
        decreasing.extend_from_slice(&[0, 0]);
        assert!(decode_rle_lossless(&decreasing, 2, 16, 1, path)
            .unwrap_err()
            .to_string()
            .contains("offsets must be increasing"));

        let mut short_segment = rle_header(&[64]);
        short_segment.push(0x80);
        assert!(decode_rle_lossless(&short_segment, 1, 8, 1, path)
            .unwrap_err()
            .to_string()
            .contains("decoded segment length mismatch"));

        let mut truncated_literal = rle_header(&[64]);
        truncated_literal.extend_from_slice(&[2, 1]);
        assert!(decode_rle_lossless(&truncated_literal, 3, 8, 1, path)
            .unwrap_err()
            .to_string()
            .contains("truncated RLE literal run"));

        let mut truncated_replicate = rle_header(&[64]);
        truncated_replicate.push(0xFF);
        assert!(decode_rle_lossless(&truncated_replicate, 2, 8, 1, path)
            .unwrap_err()
            .to_string()
            .contains("truncated RLE replicate run"));
        assert_eq!(
            decode_packbits_segment(&[0xFF, 7], 2, path).unwrap(),
            vec![7, 7]
        );

        let mut odd_expected = rle_header(&[64, 66]);
        odd_expected.extend_from_slice(&[0, 1, 0, 2]);
        assert!(decode_rle_lossless(&odd_expected, 3, 16, 1, path)
            .unwrap_err()
            .to_string()
            .contains("decoded length mismatch"));
    }

    #[test]
    fn presentation_defaults_missing_photometric_to_monochrome2() {
        let dataset = dataset_with([
            u16_element((0x0028, 0x0010), 1),
            u16_element((0x0028, 0x0011), 1),
            u16_element((0x0028, 0x0100), 8),
            u16_element((0x0028, 0x0101), 8),
            bytes_element((0x7FE0, 0x0010), vec![7]),
        ]);

        let image = present_dataset_pixels(&dataset).unwrap();
        assert_eq!(image.explanation.photometric_interpretation, "MONOCHROME2");
    }

    #[test]
    fn unsupported_bits_allocated_propagates_from_unpacker() {
        let unsupported = dataset_with([
            u16_element((0x0028, 0x0010), 1),
            u16_element((0x0028, 0x0011), 1),
            u16_element((0x0028, 0x0100), 32),
            u16_element((0x0028, 0x0101), 32),
            bytes_element((0x7FE0, 0x0010), vec![0, 0, 0, 0]),
        ]);

        assert!(present_dataset_pixels(&unsupported)
            .unwrap_err()
            .to_string()
            .contains("unsupported BitsAllocated 32"));
    }

    #[test]
    fn low_level_scaling_edges_are_explicit() {
        assert_eq!(sign_or_mask(u32::MAX, 32, false), -1);
        assert_eq!(sign_or_mask(0b1111, 4, true), -1);
        assert_eq!(window_pixels(&[0.0, 2.0], 1.0, 1.0), vec![0, 255]);
        assert_eq!(
            decode_packbits_segment(&[0x80], 0, Path::new("noop.dcm")).unwrap(),
            Vec::<u8>::new()
        );
    }

    fn dataset_with(elements: impl IntoIterator<Item = DicomElement>) -> DicomDataSet {
        dataset_with_syntax(EXPLICIT_VR_LITTLE_ENDIAN, elements)
    }

    fn dataset_with_syntax(
        transfer_syntax_uid: &str,
        elements: impl IntoIterator<Item = DicomElement>,
    ) -> DicomDataSet {
        DicomDataSet {
            path: PathBuf::from("pixels.dcm"),
            sha256: "hash".to_string(),
            transfer_syntax_uid: transfer_syntax_uid.to_string(),
            elements: elements
                .into_iter()
                .map(|element| (element.tag, element))
                .collect::<BTreeMap<_, _>>(),
        }
    }

    fn u16_element(tag: (u16, u16), value: u16) -> DicomElement {
        bytes_element(tag, value.to_le_bytes().to_vec())
    }

    fn bytes_element(tag: (u16, u16), value: Vec<u8>) -> DicomElement {
        DicomElement {
            tag,
            vr: None,
            value,
        }
    }

    fn rle_pixel_data(values: &[u8]) -> Vec<u8> {
        let mut data = rle_header(&[64]);
        data.push(values.len() as u8 - 1);
        data.extend_from_slice(values);
        data
    }

    fn rle_header(offsets: &[usize]) -> Vec<u8> {
        let mut data = vec![0; 64];
        data[..4].copy_from_slice(&(offsets.len() as u32).to_le_bytes());
        for (index, offset) in offsets.iter().copied().enumerate() {
            let start = 4 + index * 4;
            data[start..start + 4].copy_from_slice(&(offset as u32).to_le_bytes());
        }
        data
    }
}
