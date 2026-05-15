use std::{
    collections::HashMap,
    ffi::{c_char, CStr},
    fs,
    path::{Path, PathBuf},
    ptr, slice,
};

use medkit_cache::{read_cache_manifest, CachedCase};
use rayon::prelude::*;
use serde::Deserialize;

#[derive(Debug)]
pub struct DatasetHandle {
    cases: Vec<LoadedCase>,
    records: Vec<ResolvedPatch>,
    patch_size: [usize; 3],
}

#[derive(Debug)]
struct LoadedCase {
    case_id: String,
    shape: [usize; 3],
    image: Vec<f32>,
    label_f32: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
struct ResolvedPatch {
    case_index: usize,
    start: [usize; 3],
}

#[derive(Debug, Deserialize)]
struct PatchRecord {
    case_id: String,
    patch_start: [usize; 3],
    patch_size: [usize; 3],
}

#[no_mangle]
/// Opens a medkit cache and sampled patch plan for FFI batch extraction.
///
/// # Safety
///
/// `cache_dir` and `patches_path` must be valid, non-null, NUL-terminated C
/// strings. The returned pointer must be released exactly once with
/// `medkit_dataset_free`.
pub unsafe extern "C" fn medkit_dataset_open(
    cache_dir: *const c_char,
    patches_path: *const c_char,
) -> *mut DatasetHandle {
    let result = (|| {
        let cache_dir = c_path(cache_dir)?;
        let patches_path = c_path(patches_path)?;
        load_dataset(&cache_dir, &patches_path)
    })();
    match result {
        Ok(dataset) => Box::into_raw(Box::new(dataset)),
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
/// Releases a dataset handle returned by `medkit_dataset_open`.
///
/// # Safety
///
/// `handle` must be null or a pointer returned by `medkit_dataset_open` that
/// has not already been freed.
pub unsafe extern "C" fn medkit_dataset_free(handle: *mut DatasetHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

#[no_mangle]
/// Returns the number of patch records in an opened dataset.
///
/// # Safety
///
/// `handle` must be null or a live pointer returned by `medkit_dataset_open`.
pub unsafe extern "C" fn medkit_dataset_len(handle: *const DatasetHandle) -> usize {
    handle.as_ref().map_or(0, |dataset| dataset.records.len())
}

#[no_mangle]
/// Returns the patch x size for an opened dataset.
///
/// # Safety
///
/// `handle` must be null or a live pointer returned by `medkit_dataset_open`.
pub unsafe extern "C" fn medkit_dataset_patch_x(handle: *const DatasetHandle) -> usize {
    handle.as_ref().map_or(0, |dataset| dataset.patch_size[0])
}

#[no_mangle]
/// Returns the patch y size for an opened dataset.
///
/// # Safety
///
/// `handle` must be null or a live pointer returned by `medkit_dataset_open`.
pub unsafe extern "C" fn medkit_dataset_patch_y(handle: *const DatasetHandle) -> usize {
    handle.as_ref().map_or(0, |dataset| dataset.patch_size[1])
}

#[no_mangle]
/// Returns the patch z size for an opened dataset.
///
/// # Safety
///
/// `handle` must be null or a live pointer returned by `medkit_dataset_open`.
pub unsafe extern "C" fn medkit_dataset_patch_z(handle: *const DatasetHandle) -> usize {
    handle.as_ref().map_or(0, |dataset| dataset.patch_size[2])
}

#[no_mangle]
/// Fills caller-owned contiguous image and u16 label batch buffers.
///
/// # Safety
///
/// `handle` must be a live pointer returned by `medkit_dataset_open`.
/// `image_out` and `label_out` must point to writable buffers large enough for
/// `batch_size * patch_x * patch_y * patch_z` values of their respective
/// element types.
pub unsafe extern "C" fn medkit_dataset_fill_batch(
    handle: *const DatasetHandle,
    start_index: usize,
    batch_size: usize,
    image_out: *mut f32,
    label_out: *mut u16,
) -> usize {
    let Some(dataset) = handle.as_ref() else {
        return 0;
    };
    if image_out.is_null() || label_out.is_null() || dataset.records.is_empty() {
        return 0;
    }
    for batch_index in 0..batch_size {
        let record = dataset.records[(start_index + batch_index) % dataset.records.len()];
        let case = &dataset.cases[record.case_index];
        copy_patch(
            case,
            record.start,
            dataset.patch_size,
            image_out,
            label_out,
            batch_index,
        );
    }
    batch_size
}

#[no_mangle]
/// Fills caller-owned contiguous image and f32 label batch buffers.
///
/// # Safety
///
/// `handle` must be a live pointer returned by `medkit_dataset_open`.
/// `image_out` and `label_out` must point to writable buffers large enough for
/// `batch_size * patch_x * patch_y * patch_z` `f32` values.
pub unsafe extern "C" fn medkit_dataset_fill_batch_f32_labels(
    handle: *const DatasetHandle,
    start_index: usize,
    batch_size: usize,
    image_out: *mut f32,
    label_out: *mut f32,
) -> usize {
    let Some(dataset) = handle.as_ref() else {
        return 0;
    };
    if image_out.is_null() || label_out.is_null() || dataset.records.is_empty() {
        return 0;
    }
    let patch_voxels = dataset.patch_size[0] * dataset.patch_size[1] * dataset.patch_size[2];
    let Some(total_values) = patch_voxels.checked_mul(batch_size) else {
        return 0;
    };
    let image_out = slice::from_raw_parts_mut(image_out, total_values);
    let label_out = slice::from_raw_parts_mut(label_out, total_values);
    image_out
        .par_chunks_mut(patch_voxels)
        .zip(label_out.par_chunks_mut(patch_voxels))
        .enumerate()
        .for_each(|(batch_index, (image_patch, label_patch))| {
            let record = dataset.records[(start_index + batch_index) % dataset.records.len()];
            let case = &dataset.cases[record.case_index];
            copy_patch_f32_labels(
                case,
                record.start,
                dataset.patch_size,
                image_patch,
                label_patch,
            );
        });
    batch_size
}

fn load_dataset(cache_dir: &Path, patches_path: &Path) -> Result<DatasetHandle, String> {
    let manifest = read_cache_manifest(cache_dir).map_err(|error| error.to_string())?;
    let mut cases = Vec::with_capacity(manifest.cases.len());
    for case in &manifest.cases {
        cases.push(load_case(case)?);
    }
    let case_indices = cases
        .iter()
        .enumerate()
        .map(|(index, case)| (case.case_id.as_str(), index))
        .collect::<HashMap<_, _>>();
    let records = read_patch_records(patches_path)?;
    if records.is_empty() {
        return Err(format!(
            "patch plan has no records: {}",
            patches_path.display()
        ));
    }
    let patch_size = records[0].patch_size;
    let mut resolved = Vec::with_capacity(records.len());
    for record in records {
        if record.patch_size != patch_size {
            return Err("mixed patch sizes are not supported".to_string());
        }
        let Some(case_index) = case_indices.get(record.case_id.as_str()).copied() else {
            return Err(format!("missing cached case {}", record.case_id));
        };
        resolved.push(ResolvedPatch {
            case_index,
            start: record.patch_start,
        });
    }
    Ok(DatasetHandle {
        cases,
        records: resolved,
        patch_size,
    })
}

fn load_case(case: &CachedCase) -> Result<LoadedCase, String> {
    Ok(LoadedCase {
        case_id: case.case_id.clone(),
        shape: case.shape,
        image: read_f32_volume(Path::new(&case.image_cache_path), case.shape)?,
        label_f32: read_u16_volume_as_f32(Path::new(&case.label_cache_path), case.shape)?,
    })
}

fn read_patch_records(path: &Path) -> Result<Vec<PatchRecord>, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).map_err(|error| error.to_string()))
        .collect()
}

fn read_f32_volume(path: &Path, shape: [usize; 3]) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let expected = shape[0] * shape[1] * shape[2] * 4;
    if bytes.len() != expected {
        return Err(format!(
            "{} has {} bytes, expected {expected}",
            path.display(),
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk length")))
        .collect())
}

fn read_u16_volume_as_f32(path: &Path, shape: [usize; 3]) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let expected = shape[0] * shape[1] * shape[2] * 2;
    if bytes.len() != expected {
        return Err(format!(
            "{} has {} bytes, expected {expected}",
            path.display(),
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| f32::from(u16::from_le_bytes(chunk.try_into().expect("chunk length"))))
        .collect())
}

unsafe fn copy_patch(
    case: &LoadedCase,
    start: [usize; 3],
    patch_size: [usize; 3],
    image_out: *mut f32,
    label_out: *mut u16,
    batch_index: usize,
) {
    let patch_voxels = patch_size[0] * patch_size[1] * patch_size[2];
    let batch_offset = batch_index * patch_voxels;
    for local_z in 0..patch_size[2] {
        let z = start[2] + local_z;
        for local_y in 0..patch_size[1] {
            let y = start[1] + local_y;
            let source_start = start[0] + case.shape[0] * (y + case.shape[1] * z);
            let destination_start =
                batch_offset + patch_size[0] * (local_y + patch_size[1] * local_z);
            ptr::copy_nonoverlapping(
                case.image.as_ptr().add(source_start),
                image_out.add(destination_start),
                patch_size[0],
            );
            let label_row = &case.label_f32[source_start..source_start + patch_size[0]];
            for (offset, value) in label_row.iter().enumerate() {
                *label_out.add(destination_start + offset) = *value as u16;
            }
        }
    }
}

fn copy_patch_f32_labels(
    case: &LoadedCase,
    start: [usize; 3],
    patch_size: [usize; 3],
    image_out: &mut [f32],
    label_out: &mut [f32],
) {
    for local_z in 0..patch_size[2] {
        let z = start[2] + local_z;
        for local_y in 0..patch_size[1] {
            let y = start[1] + local_y;
            let source_start = start[0] + case.shape[0] * (y + case.shape[1] * z);
            let destination_start = patch_size[0] * (local_y + patch_size[1] * local_z);
            let destination_end = destination_start + patch_size[0];
            let source_end = source_start + patch_size[0];
            image_out[destination_start..destination_end]
                .copy_from_slice(&case.image[source_start..source_end]);
            label_out[destination_start..destination_end]
                .copy_from_slice(&case.label_f32[source_start..source_end]);
        }
    }
}

unsafe fn c_path(value: *const c_char) -> Result<PathBuf, String> {
    if value.is_null() {
        return Err("null path".to_string());
    }
    let text = CStr::from_ptr(value)
        .to_str()
        .map_err(|error| error.to_string())?;
    Ok(PathBuf::from(text))
}
