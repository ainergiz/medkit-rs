use std::path::Path;

/// Derives a case id from an image path.
///
/// This accepts regular NIfTI names such as `case_001.nii.gz` and nnU-Net-style
/// image channel names such as `case_001_0000.nii.gz`, where the trailing
/// channel suffix is removed.
pub fn case_id_from_image_path(path: &Path) -> Option<String> {
    let stem = nifti_stem(path)?;
    Some(strip_nnunet_channel_suffix(stem).to_string())
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
    if suffix.len() == 4 && suffix.chars().all(|character| character.is_ascii_digit()) {
        case_id
    } else {
        stem
    }
}
