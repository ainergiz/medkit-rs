use std::path::Path;

use serde::{Deserialize, Serialize};

/// Dataset image naming layout used for image/label pairing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetLayout {
    /// Use full NIfTI stems as case IDs without stripping channel suffixes.
    #[default]
    Flat,
    /// Treat image stems ending in `_dddd` as nnU-Net channel image names.
    Nnunet,
}

impl DatasetLayout {
    /// Stable manifest and report representation.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::Nnunet => "nnunet",
        }
    }
}

/// Derives a case id from an image path.
///
/// This uses the safe flat layout and preserves the full NIfTI stem.
pub fn case_id_from_image_path(path: &Path) -> Option<String> {
    case_id_from_image_path_for_layout(path, DatasetLayout::Flat)
}

pub(crate) fn case_id_from_image_path_for_layout(
    path: &Path,
    layout: DatasetLayout,
) -> Option<String> {
    let stem = nifti_stem(path)?;
    Some(
        match layout {
            DatasetLayout::Flat => stem,
            DatasetLayout::Nnunet => strip_nnunet_channel_suffix(stem),
        }
        .to_string(),
    )
}

pub(crate) fn channel_index_from_image_path_for_layout(
    path: &Path,
    layout: DatasetLayout,
) -> Option<u16> {
    match layout {
        DatasetLayout::Flat => None,
        DatasetLayout::Nnunet => nifti_stem(path).and_then(nnunet_channel_index),
    }
}

/// Derives a case id from a label path.
pub fn case_id_from_label_path(path: &Path) -> Option<String> {
    nifti_stem(path).map(ToString::to_string)
}

pub(crate) fn is_nifti_path(path: &Path) -> bool {
    nifti_stem(path).is_some()
}

fn nifti_stem(path: &Path) -> Option<&str> {
    let file_name = path.file_name()?.to_str()?;
    let lower = file_name.to_ascii_lowercase();
    if lower.ends_with(".hdr.gz") {
        Some(&file_name[..file_name.len() - ".hdr.gz".len()])
    } else if lower.ends_with(".nii.gz") {
        Some(&file_name[..file_name.len() - ".nii.gz".len()])
    } else if lower.ends_with(".nii") {
        Some(&file_name[..file_name.len() - ".nii".len()])
    } else if lower.ends_with(".hdr") {
        Some(&file_name[..file_name.len() - ".hdr".len()])
    } else {
        None
    }
}

fn strip_nnunet_channel_suffix(stem: &str) -> &str {
    let Some((case_id, suffix)) = stem.rsplit_once('_') else {
        return stem;
    };
    if suffix.len() == 4
        && suffix.bytes().all(|byte| byte.is_ascii_digit())
        && suffix.parse::<u16>().is_ok_and(|value| value < 1000)
    {
        case_id
    } else {
        stem
    }
}

fn nnunet_channel_index(stem: &str) -> Option<u16> {
    let (_, suffix) = stem.rsplit_once('_')?;
    if suffix.len() == 4 && suffix.bytes().all(|byte| byte.is_ascii_digit()) {
        suffix.parse::<u16>().ok().filter(|value| *value < 1000)
    } else {
        None
    }
}
