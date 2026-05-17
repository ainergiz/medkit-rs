use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use sha2::{Digest, Sha256};

use crate::{
    types::{
        DicomInspectReport, DicomInventoryRecord, DicomWarning, EXPLICIT_VR_BIG_ENDIAN,
        EXPLICIT_VR_LITTLE_ENDIAN, IMPLICIT_VR_LITTLE_ENDIAN, RLE_LOSSLESS,
    },
    DicomError, Result,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DicomElement {
    pub tag: (u16, u16),
    pub vr: Option<String>,
    pub value: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct DicomDataSet {
    pub path: PathBuf,
    pub sha256: String,
    pub transfer_syntax_uid: String,
    pub elements: BTreeMap<(u16, u16), DicomElement>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endian {
    Little,
    Big,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VrMode {
    Explicit,
    Implicit,
}

pub fn inspect_dicom_file(path: impl AsRef<Path>) -> Result<DicomInspectReport> {
    let dataset = DicomDataSet::from_file(path)?;
    let record = dataset.inventory_record();
    let elements = dataset
        .elements
        .values()
        .map(|element| (format_tag(element.tag), dataset.display_value(element.tag)))
        .collect();
    Ok(DicomInspectReport { record, elements })
}

impl DicomDataSet {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|source| DicomError::io(path, source))?;
        Self::from_bytes(path.to_path_buf(), bytes)
    }

    pub fn from_bytes(path: PathBuf, bytes: Vec<u8>) -> Result<Self> {
        if bytes.len() < 132 || &bytes[128..132] != b"DICM" {
            return Err(DicomError::parse(path, "missing DICOM Part 10 preamble"));
        }
        let sha256 = sha256_hex(&bytes);
        let mut cursor = 132usize;
        let mut elements = BTreeMap::new();
        let mut transfer_syntax_uid = EXPLICIT_VR_LITTLE_ENDIAN.to_string();
        while cursor + 8 <= bytes.len() {
            if read_u16(&bytes, cursor, Endian::Little) != 0x0002 {
                break;
            }
            let element =
                parse_element(&bytes, &mut cursor, Endian::Little, VrMode::Explicit, &path)?;
            let tag = element.tag;
            if tag == (0x0002, 0x0010) {
                transfer_syntax_uid = clean_text(&element.value);
            }
            elements.insert(tag, element);
        }

        let (endian, vr_mode) = syntax_modes(&transfer_syntax_uid);
        if endian.is_none() || vr_mode.is_none() {
            parse_dataset_best_effort(
                &bytes,
                &mut cursor,
                Endian::Little,
                VrMode::Explicit,
                &path,
                &mut elements,
            );
            return Ok(Self {
                path,
                sha256,
                transfer_syntax_uid,
                elements,
            });
        }
        parse_dataset(
            &bytes,
            &mut cursor,
            endian.expect("checked"),
            vr_mode.expect("checked"),
            &path,
            &mut elements,
        )?;
        Ok(Self {
            path,
            sha256,
            transfer_syntax_uid,
            elements,
        })
    }

    pub fn inventory_record(&self) -> DicomInventoryRecord {
        let mut warnings = Vec::new();
        if self.has((0x0010, 0x0010)) {
            warnings.push(DicomWarning::new(
                "phi_patient_name",
                "PatientName is present",
            ));
        }
        if self.has((0x0010, 0x0030)) {
            warnings.push(DicomWarning::new(
                "phi_patient_birth_date",
                "PatientBirthDate is present",
            ));
        }
        if !self.is_supported_transfer_syntax() {
            warnings.push(DicomWarning::new(
                "unsupported_transfer_syntax",
                format!("unsupported transfer syntax {}", self.transfer_syntax_uid),
            ));
        }
        for (tag, code, label) in [
            ((0x0010, 0x0020), "missing_patient_id", "PatientID"),
            (
                (0x0020, 0x000D),
                "missing_study_instance_uid",
                "StudyInstanceUID",
            ),
            (
                (0x0020, 0x000E),
                "missing_series_instance_uid",
                "SeriesInstanceUID",
            ),
            (
                (0x0008, 0x0018),
                "missing_sop_instance_uid",
                "SOPInstanceUID",
            ),
        ] {
            if !self.has(tag) {
                warnings.push(DicomWarning::new(code, format!("{label} is missing")));
            }
        }
        if self.string((0x0018, 0x5101)).is_none() {
            warnings.push(DicomWarning::new(
                "missing_view_position",
                "ViewPosition is missing",
            ));
        }
        if self.pixel_spacing((0x0028, 0x0030)).is_none()
            && self.pixel_spacing((0x0018, 0x1164)).is_none()
        {
            warnings.push(DicomWarning::new(
                "missing_pixel_spacing",
                "PixelSpacing and ImagerPixelSpacing are missing",
            ));
        }
        let rows = self.u16_value((0x0028, 0x0010));
        let columns = self.u16_value((0x0028, 0x0011));
        if matches!(rows, Some(0)) || matches!(columns, Some(0)) {
            warnings.push(DicomWarning::new(
                "invalid_image_dimensions",
                "Rows and Columns must be greater than zero",
            ));
        }

        DicomInventoryRecord {
            path: self.path.display().to_string(),
            sha256: self.sha256.clone(),
            patient_id: self.string((0x0010, 0x0020)),
            study_instance_uid: self.string((0x0020, 0x000D)),
            series_instance_uid: self.string((0x0020, 0x000E)),
            sop_instance_uid: self.string((0x0008, 0x0018)),
            modality: self.string((0x0008, 0x0060)),
            body_part_examined: self.string((0x0018, 0x0015)),
            view_position: self.string((0x0018, 0x5101)),
            laterality: self
                .string((0x0020, 0x0062))
                .or_else(|| self.string((0x0020, 0x0060))),
            rows,
            columns,
            samples_per_pixel: self.u16_value((0x0028, 0x0002)),
            bits_allocated: self.u16_value((0x0028, 0x0100)),
            bits_stored: self.u16_value((0x0028, 0x0101)),
            high_bit: self.u16_value((0x0028, 0x0102)),
            pixel_representation: self.u16_value((0x0028, 0x0103)).map(|value| {
                if value == 0 {
                    "unsigned".to_string()
                } else {
                    "signed".to_string()
                }
            }),
            photometric_interpretation: self.string((0x0028, 0x0004)),
            transfer_syntax_uid: self.transfer_syntax_uid.clone(),
            pixel_spacing: self.pixel_spacing((0x0028, 0x0030)),
            imager_pixel_spacing: self.pixel_spacing((0x0018, 0x1164)),
            rescale_intercept: self.f32_value((0x0028, 0x1052)),
            rescale_slope: self.f32_value((0x0028, 0x1053)),
            window_center: self.f32_value((0x0028, 0x1050)),
            window_width: self.f32_value((0x0028, 0x1051)),
            pixel_hash: self.element((0x7FE0, 0x0010)).map(sha256_hex),
            decoder_backend: None,
            decoder_version: None,
            warnings,
        }
    }

    pub fn element(&self, tag: (u16, u16)) -> Option<&[u8]> {
        self.elements
            .get(&tag)
            .map(|element| element.value.as_slice())
    }

    pub fn has(&self, tag: (u16, u16)) -> bool {
        self.elements.contains_key(&tag)
    }

    pub fn string(&self, tag: (u16, u16)) -> Option<String> {
        self.element(tag)
            .map(clean_text)
            .filter(|value| !value.is_empty())
    }

    pub fn u16_value(&self, tag: (u16, u16)) -> Option<u16> {
        let value = self.element(tag)?;
        if value.len() < 2 {
            return None;
        }
        let endian = self.dataset_endian();
        Some(match endian {
            Endian::Little => u16::from_le_bytes([value[0], value[1]]),
            Endian::Big => u16::from_be_bytes([value[0], value[1]]),
        })
    }

    pub fn f32_value(&self, tag: (u16, u16)) -> Option<f32> {
        self.string(tag)
            .and_then(|value| value.split('\\').next().map(str::trim).map(str::to_string))
            .and_then(|value| value.parse::<f32>().ok())
    }

    pub fn pixel_spacing(&self, tag: (u16, u16)) -> Option<[f32; 2]> {
        let text = self.string(tag)?;
        let mut values = text
            .split('\\')
            .filter_map(|item| item.trim().parse::<f32>().ok());
        Some([values.next()?, values.next()?])
    }

    pub fn display_value(&self, tag: (u16, u16)) -> String {
        match tag {
            (0x7FE0, 0x0010) => {
                format!("<{} pixel bytes>", self.element(tag).map_or(0, <[u8]>::len))
            }
            (0x0028, 0x0010)
            | (0x0028, 0x0011)
            | (0x0028, 0x0002)
            | (0x0028, 0x0100)
            | (0x0028, 0x0101)
            | (0x0028, 0x0102)
            | (0x0028, 0x0103) => self
                .u16_value(tag)
                .map(|value| value.to_string())
                .unwrap_or_else(|| "<invalid u16>".to_string()),
            _ => self
                .string(tag)
                .unwrap_or_else(|| format!("<{} bytes>", self.element(tag).map_or(0, <[u8]>::len))),
        }
    }

    pub fn is_supported_transfer_syntax(&self) -> bool {
        matches!(
            self.transfer_syntax_uid.as_str(),
            EXPLICIT_VR_LITTLE_ENDIAN
                | IMPLICIT_VR_LITTLE_ENDIAN
                | EXPLICIT_VR_BIG_ENDIAN
                | RLE_LOSSLESS
        )
    }

    fn dataset_endian(&self) -> Endian {
        if self.transfer_syntax_uid == EXPLICIT_VR_BIG_ENDIAN {
            Endian::Big
        } else {
            Endian::Little
        }
    }

    pub(crate) fn pixel_is_big_endian(&self) -> bool {
        matches!(self.dataset_endian(), Endian::Big)
    }
}

fn parse_dataset(
    bytes: &[u8],
    cursor: &mut usize,
    endian: Endian,
    vr_mode: VrMode,
    path: &Path,
    elements: &mut BTreeMap<(u16, u16), DicomElement>,
) -> Result<()> {
    while *cursor + 8 <= bytes.len() {
        let element = parse_element(bytes, cursor, endian, vr_mode, path)?;
        elements.insert(element.tag, element);
    }
    Ok(())
}

fn parse_dataset_best_effort(
    bytes: &[u8],
    cursor: &mut usize,
    endian: Endian,
    vr_mode: VrMode,
    path: &Path,
    elements: &mut BTreeMap<(u16, u16), DicomElement>,
) {
    let _ = parse_dataset(bytes, cursor, endian, vr_mode, path, elements);
}

fn parse_element(
    bytes: &[u8],
    cursor: &mut usize,
    endian: Endian,
    vr_mode: VrMode,
    path: &Path,
) -> Result<DicomElement> {
    if *cursor + 8 > bytes.len() {
        return Err(DicomError::parse(path, "truncated DICOM element header"));
    }
    let group = read_u16(bytes, *cursor, endian);
    let element = read_u16(bytes, *cursor + 2, endian);
    *cursor += 4;

    let (vr, length) = match vr_mode {
        VrMode::Implicit => {
            let length = read_u32(bytes, *cursor, endian);
            *cursor += 4;
            (None, length)
        }
        VrMode::Explicit => {
            let vr = std::str::from_utf8(&bytes[*cursor..*cursor + 2])
                .map_err(|_| DicomError::parse(path, "invalid explicit VR bytes"))?
                .to_string();
            *cursor += 2;
            if long_vr(&vr) {
                if *cursor + 6 > bytes.len() {
                    return Err(DicomError::parse(path, "truncated long VR length"));
                }
                *cursor += 2;
                let length = read_u32(bytes, *cursor, endian);
                *cursor += 4;
                (Some(vr), length)
            } else {
                let length = read_u16(bytes, *cursor, endian) as u32;
                *cursor += 2;
                (Some(vr), length)
            }
        }
    };

    if length == u32::MAX && (group, element) == (0x7FE0, 0x0010) {
        let value = parse_undefined_length_pixel_data(bytes, cursor, path)?;
        return Ok(DicomElement {
            tag: (group, element),
            vr,
            value,
        });
    }
    if length == u32::MAX {
        return Err(DicomError::unsupported(
            path,
            format!("undefined length element {}", format_tag((group, element))),
        ));
    }
    let length = length as usize;
    if *cursor + length > bytes.len() {
        return Err(DicomError::parse(
            path,
            format!(
                "element {} length {} exceeds remaining bytes",
                format_tag((group, element)),
                length
            ),
        ));
    }
    let value = bytes[*cursor..*cursor + length].to_vec();
    *cursor += length;
    if length % 2 == 1 && *cursor < bytes.len() {
        *cursor += 1;
    }
    Ok(DicomElement {
        tag: (group, element),
        vr,
        value,
    })
}

fn syntax_modes(uid: &str) -> (Option<Endian>, Option<VrMode>) {
    match uid {
        EXPLICIT_VR_LITTLE_ENDIAN => (Some(Endian::Little), Some(VrMode::Explicit)),
        IMPLICIT_VR_LITTLE_ENDIAN => (Some(Endian::Little), Some(VrMode::Implicit)),
        EXPLICIT_VR_BIG_ENDIAN => (Some(Endian::Big), Some(VrMode::Explicit)),
        RLE_LOSSLESS => (Some(Endian::Little), Some(VrMode::Explicit)),
        _ => (None, None),
    }
}

fn parse_undefined_length_pixel_data(
    bytes: &[u8],
    cursor: &mut usize,
    path: &Path,
) -> Result<Vec<u8>> {
    let mut fragments = Vec::new();
    let mut item_index = 0usize;
    loop {
        if *cursor + 8 > bytes.len() {
            return Err(DicomError::parse(
                path,
                "truncated undefined length PixelData item header",
            ));
        }
        let group = read_u16(bytes, *cursor, Endian::Little);
        let element = read_u16(bytes, *cursor + 2, Endian::Little);
        let length = read_u32(bytes, *cursor + 4, Endian::Little) as usize;
        *cursor += 8;
        match (group, element) {
            (0xFFFE, 0xE0DD) => {
                if length != 0 {
                    return Err(DicomError::parse(
                        path,
                        "PixelData sequence delimitation item must have zero length",
                    ));
                }
                return Ok(fragments);
            }
            (0xFFFE, 0xE000) => {
                if *cursor + length > bytes.len() {
                    return Err(DicomError::parse(
                        path,
                        "PixelData fragment length exceeds remaining bytes",
                    ));
                }
                if item_index > 0 {
                    fragments.extend_from_slice(&bytes[*cursor..*cursor + length]);
                }
                *cursor += length;
                item_index += 1;
            }
            _ => {
                return Err(DicomError::parse(
                    path,
                    format!(
                        "expected PixelData item or sequence delimitation, got {}",
                        format_tag((group, element))
                    ),
                ));
            }
        }
    }
}

fn read_u16(bytes: &[u8], offset: usize, endian: Endian) -> u16 {
    match endian {
        Endian::Little => u16::from_le_bytes([bytes[offset], bytes[offset + 1]]),
        Endian::Big => u16::from_be_bytes([bytes[offset], bytes[offset + 1]]),
    }
}

fn read_u32(bytes: &[u8], offset: usize, endian: Endian) -> u32 {
    match endian {
        Endian::Little => u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]),
        Endian::Big => u32::from_be_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]),
    }
}

fn long_vr(vr: &str) -> bool {
    matches!(
        vr,
        "OB" | "OD" | "OF" | "OL" | "OW" | "SQ" | "UC" | "UR" | "UT" | "UN"
    )
}

fn clean_text(value: &[u8]) -> String {
    String::from_utf8_lossy(value)
        .trim_end_matches(['\0', ' '])
        .trim()
        .to_string()
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn format_tag(tag: (u16, u16)) -> String {
    format!("({:04X},{:04X})", tag.0, tag.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_warnings_cover_missing_unsupported_and_invalid_metadata() {
        let mut elements = BTreeMap::new();
        elements.insert(
            (0x0010, 0x0010),
            text_element((0x0010, 0x0010), "Patient^Name"),
        );
        elements.insert((0x0010, 0x0030), text_element((0x0010, 0x0030), "19700101"));
        elements.insert((0x0028, 0x0010), u16_element((0x0028, 0x0010), 0));
        elements.insert((0x0028, 0x0011), u16_element((0x0028, 0x0011), 2));
        elements.insert((0x0028, 0x0103), u16_element((0x0028, 0x0103), 1));
        elements.insert(
            (0x0028, 0x0100),
            DicomElement {
                tag: (0x0028, 0x0100),
                vr: Some("US".to_string()),
                value: vec![1],
            },
        );

        let dataset = DicomDataSet {
            path: PathBuf::from("bad-metadata.dcm"),
            sha256: "hash".to_string(),
            transfer_syntax_uid: "1.2.3.unsupported".to_string(),
            elements,
        };

        assert_eq!(dataset.u16_value((0x0028, 0x0100)), None);
        assert_eq!(dataset.display_value((0x0028, 0x0100)), "<invalid u16>");
        let record = dataset.inventory_record();
        assert_eq!(record.pixel_representation.as_deref(), Some("signed"));
        for code in [
            "phi_patient_name",
            "phi_patient_birth_date",
            "unsupported_transfer_syntax",
            "missing_patient_id",
            "missing_study_instance_uid",
            "missing_series_instance_uid",
            "missing_sop_instance_uid",
            "missing_view_position",
            "missing_pixel_spacing",
            "invalid_image_dimensions",
        ] {
            assert!(
                record.warnings.iter().any(|warning| warning.code == code),
                "missing warning {code}"
            );
        }
    }

    #[test]
    fn parse_element_reports_malformed_element_shapes() {
        let path = Path::new("broken.dcm");
        let mut cursor = 0;
        assert!(
            parse_element(&[0; 7], &mut cursor, Endian::Little, VrMode::Explicit, path)
                .unwrap_err()
                .to_string()
                .contains("truncated DICOM element header")
        );

        let mut invalid_vr = le_tag((0x0010, 0x0020));
        invalid_vr.extend_from_slice(&[0xFF, 0xFF, 0, 0]);
        cursor = 0;
        assert!(parse_element(
            &invalid_vr,
            &mut cursor,
            Endian::Little,
            VrMode::Explicit,
            path
        )
        .unwrap_err()
        .to_string()
        .contains("invalid explicit VR bytes"));

        let mut truncated_long = le_tag((0x7FE0, 0x0010));
        truncated_long.extend_from_slice(b"OB");
        truncated_long.extend_from_slice(&[0, 0]);
        cursor = 0;
        assert!(parse_element(
            &truncated_long,
            &mut cursor,
            Endian::Little,
            VrMode::Explicit,
            path
        )
        .unwrap_err()
        .to_string()
        .contains("truncated long VR length"));

        let mut undefined = le_tag((0x7FE0, 0x0010));
        undefined.extend_from_slice(b"OB");
        undefined.extend_from_slice(&[0, 0]);
        undefined.extend_from_slice(&u32::MAX.to_le_bytes());
        cursor = 0;
        assert!(parse_element(
            &undefined,
            &mut cursor,
            Endian::Little,
            VrMode::Explicit,
            path
        )
        .unwrap_err()
        .to_string()
        .contains("truncated undefined length PixelData item header"));

        let mut encapsulated = undefined.clone();
        encapsulated.extend_from_slice(&le_item((0xFFFE, 0xE000), b""));
        encapsulated.extend_from_slice(&le_item((0xFFFE, 0xE000), b"abc"));
        encapsulated.extend_from_slice(&le_item((0xFFFE, 0xE0DD), b""));
        cursor = 0;
        let element = parse_element(
            &encapsulated,
            &mut cursor,
            Endian::Little,
            VrMode::Explicit,
            path,
        )
        .unwrap();
        assert_eq!(element.value, b"abc");

        let mut bad_delimitation = undefined.clone();
        bad_delimitation.extend_from_slice(&le_item((0xFFFE, 0xE0DD), b"x"));
        cursor = 0;
        assert!(parse_element(
            &bad_delimitation,
            &mut cursor,
            Endian::Little,
            VrMode::Explicit,
            path
        )
        .unwrap_err()
        .to_string()
        .contains("delimitation item must have zero length"));

        let mut long_fragment = undefined.clone();
        long_fragment.extend_from_slice(&le_tag((0xFFFE, 0xE000)));
        long_fragment.extend_from_slice(&4u32.to_le_bytes());
        long_fragment.extend_from_slice(b"ab");
        cursor = 0;
        assert!(parse_element(
            &long_fragment,
            &mut cursor,
            Endian::Little,
            VrMode::Explicit,
            path
        )
        .unwrap_err()
        .to_string()
        .contains("fragment length exceeds remaining bytes"));

        let mut bad_item_tag = undefined.clone();
        bad_item_tag.extend_from_slice(&le_item((0x0010, 0x0010), b""));
        cursor = 0;
        assert!(parse_element(
            &bad_item_tag,
            &mut cursor,
            Endian::Little,
            VrMode::Explicit,
            path
        )
        .unwrap_err()
        .to_string()
        .contains("expected PixelData item"));

        let mut undefined_non_pixel = le_tag((0x0008, 0x1111));
        undefined_non_pixel.extend_from_slice(b"SQ");
        undefined_non_pixel.extend_from_slice(&[0, 0]);
        undefined_non_pixel.extend_from_slice(&u32::MAX.to_le_bytes());
        cursor = 0;
        assert!(parse_element(
            &undefined_non_pixel,
            &mut cursor,
            Endian::Little,
            VrMode::Explicit,
            path
        )
        .unwrap_err()
        .to_string()
        .contains("undefined length element"));

        let mut too_long = le_tag((0x0010, 0x0020));
        too_long.extend_from_slice(b"LO");
        too_long.extend_from_slice(&4u16.to_le_bytes());
        too_long.extend_from_slice(b"ab");
        cursor = 0;
        assert!(parse_element(
            &too_long,
            &mut cursor,
            Endian::Little,
            VrMode::Explicit,
            path
        )
        .unwrap_err()
        .to_string()
        .contains("exceeds remaining bytes"));

        let mut odd = le_tag((0x0010, 0x0020));
        odd.extend_from_slice(b"LO");
        odd.extend_from_slice(&1u16.to_le_bytes());
        odd.extend_from_slice(b"a ");
        cursor = 0;
        let element =
            parse_element(&odd, &mut cursor, Endian::Little, VrMode::Explicit, path).unwrap();
        assert_eq!(element.value, b"a");
        assert_eq!(cursor, odd.len());

        assert_eq!(read_u32(&[1, 2, 3, 4], 0, Endian::Little), 0x0403_0201);
    }

    #[test]
    fn public_parser_entrypoints_report_read_and_meta_errors() {
        let missing = std::env::temp_dir().join(format!(
            "medkit-dicom-missing-{}-{}.dcm",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        assert!(DicomDataSet::from_file(&missing)
            .unwrap_err()
            .to_string()
            .contains("No such file"));
        assert!(inspect_dicom_file(&missing)
            .unwrap_err()
            .to_string()
            .contains("No such file"));

        let mut bad_meta = vec![0u8; 128];
        bad_meta.extend_from_slice(b"DICM");
        bad_meta.extend_from_slice(&le_tag((0x0002, 0x0010)));
        bad_meta.extend_from_slice(&[0xFF, 0xFF, 0, 0]);
        assert!(
            DicomDataSet::from_bytes(PathBuf::from("bad-meta.dcm"), bad_meta)
                .unwrap_err()
                .to_string()
                .contains("invalid explicit VR bytes")
        );
    }

    #[test]
    fn text_helpers_cover_empty_binary_and_incomplete_spacing() {
        let mut elements = BTreeMap::new();
        elements.insert((0x0010, 0x0020), bytes_element((0x0010, 0x0020), vec![]));
        elements.insert((0x0028, 0x0030), text_element((0x0028, 0x0030), "0.5"));
        elements.insert(
            (0x0018, 0x1164),
            text_element((0x0018, 0x1164), "not-a-number"),
        );
        let dataset = DicomDataSet {
            path: PathBuf::from("text.dcm"),
            sha256: "hash".to_string(),
            transfer_syntax_uid: EXPLICIT_VR_LITTLE_ENDIAN.to_string(),
            elements,
        };

        assert_eq!(dataset.display_value((0x0010, 0x0020)), "<0 bytes>");
        assert_eq!(dataset.pixel_spacing((0x0028, 0x0030)), None);
        assert_eq!(dataset.pixel_spacing((0x0018, 0x1164)), None);
    }

    #[test]
    fn dataset_parse_error_is_propagated_for_known_transfer_syntax() {
        let mut bytes = vec![0u8; 128];
        bytes.extend_from_slice(b"DICM");
        push_explicit(
            &mut bytes,
            (0x0002, 0x0010),
            "UI",
            EXPLICIT_VR_LITTLE_ENDIAN,
        );
        push_explicit(&mut bytes, (0x0002, 0x0013), "SH", "A");
        push_explicit(&mut bytes, (0x0002, 0x0016), "AE", "AB");
        bytes.extend_from_slice(&le_tag((0x0010, 0x0020)));
        bytes.extend_from_slice(b"LO");
        bytes.extend_from_slice(&8u16.to_le_bytes());
        bytes.extend_from_slice(b"short");

        assert!(
            DicomDataSet::from_bytes(PathBuf::from("bad-known.dcm"), bytes)
                .unwrap_err()
                .to_string()
                .contains("exceeds remaining bytes")
        );
    }

    fn text_element(tag: (u16, u16), value: &str) -> DicomElement {
        DicomElement {
            tag,
            vr: Some("LO".to_string()),
            value: value.as_bytes().to_vec(),
        }
    }

    fn u16_element(tag: (u16, u16), value: u16) -> DicomElement {
        DicomElement {
            tag,
            vr: Some("US".to_string()),
            value: value.to_le_bytes().to_vec(),
        }
    }

    fn bytes_element(tag: (u16, u16), value: Vec<u8>) -> DicomElement {
        DicomElement {
            tag,
            vr: None,
            value,
        }
    }

    fn push_explicit(out: &mut Vec<u8>, tag: (u16, u16), vr: &str, value: &str) {
        out.extend_from_slice(&le_tag(tag));
        out.extend_from_slice(vr.as_bytes());
        let mut bytes = value.as_bytes().to_vec();
        if bytes.len() % 2 == 1 {
            bytes.push(if vr == "UI" { 0 } else { b' ' });
        }
        out.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(&bytes);
    }

    fn le_tag(tag: (u16, u16)) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&tag.0.to_le_bytes());
        out.extend_from_slice(&tag.1.to_le_bytes());
        out
    }

    fn le_item(tag: (u16, u16), value: &[u8]) -> Vec<u8> {
        let mut out = le_tag(tag);
        out.extend_from_slice(&(value.len() as u32).to_le_bytes());
        out.extend_from_slice(value);
        out
    }
}
