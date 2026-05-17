use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{parser::DicomDataSet, DicomError, Result};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PixelExplanation {
    pub path: String,
    pub width: usize,
    pub height: usize,
    pub transfer_syntax_uid: String,
    pub photometric_interpretation: String,
    pub source_pixel_hash: String,
    pub presented_pixel_hash: String,
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

pub fn explain_pixels(path: impl AsRef<Path>) -> Result<PixelExplanation> {
    Ok(present_dicom_pixels(path)?.explanation)
}

pub fn present_dicom_pixels(path: impl AsRef<Path>) -> Result<PresentedImage> {
    let dataset = DicomDataSet::from_file(path.as_ref())?;
    present_dataset_pixels(&dataset)
}

pub(crate) fn present_dataset_pixels(dataset: &DicomDataSet) -> Result<PresentedImage> {
    if !dataset.is_supported_transfer_syntax() {
        return Err(DicomError::unsupported(
            &dataset.path,
            format!(
                "unsupported transfer syntax {}",
                dataset.transfer_syntax_uid
            ),
        ));
    }
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
    let pixel_data = dataset
        .element((0x7FE0, 0x0010))
        .ok_or_else(|| DicomError::parse(&dataset.path, "missing PixelData"))?;
    let expected = rows * columns * (bits_allocated as usize).div_ceil(8);
    if pixel_data.len() != expected {
        return Err(DicomError::parse(
            &dataset.path,
            format!(
                "PixelData length mismatch: expected {expected} bytes, got {}",
                pixel_data.len()
            ),
        ));
    }

    let mut steps = vec![
        format!("decode transfer syntax {}", dataset.transfer_syntax_uid),
        format!("unpack {bits_allocated}-bit pixels with {bits_stored} stored bits"),
    ];
    let mut values = unpack_pixels(
        pixel_data,
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
        source_pixel_hash: sha256_hex(pixel_data),
        presented_pixel_hash: sha256_hex(&pixels),
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
    use std::{collections::BTreeMap, path::PathBuf};

    use crate::{parser::DicomElement, types::EXPLICIT_VR_LITTLE_ENDIAN};

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
    }

    fn dataset_with(elements: impl IntoIterator<Item = DicomElement>) -> DicomDataSet {
        DicomDataSet {
            path: PathBuf::from("pixels.dcm"),
            sha256: "hash".to_string(),
            transfer_syntax_uid: EXPLICIT_VR_LITTLE_ENDIAN.to_string(),
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
}
