use std::path::Path;

/// Dataset image naming layout used for image/label pairing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DatasetLayout {
    /// Use full NIfTI stems as case IDs without stripping channel suffixes.
    #[default]
    Flat,
    /// Treat image stems ending in `_0000` as nnU-Net single-channel image names.
    Nnunet,
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
    if lower.ends_with(".nii.gz") {
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
    if suffix == "0000" {
        case_id
    } else {
        stem
    }
}
