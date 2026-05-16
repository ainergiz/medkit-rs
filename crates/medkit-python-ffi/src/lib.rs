use std::{
    collections::HashMap,
    ffi::{c_char, CStr},
    fs,
    path::{Path, PathBuf},
    ptr, slice,
};

use medkit_cache::{read_cache_manifest, CachedCase};
use memmap2::Mmap;
use rayon::prelude::*;
use serde::Deserialize;

#[derive(Debug)]
pub struct DatasetHandle {
    cases: Vec<LoadedCase>,
    records: Vec<ResolvedPatch>,
    patch_size: [usize; 3],
}

impl DatasetHandle {
    pub fn open(
        cache_dir: impl AsRef<Path>,
        patches_path: impl AsRef<Path>,
    ) -> Result<Self, String> {
        Self::open_with_storage(cache_dir, patches_path, StorageMode::Resident)
    }

    pub fn open_with_storage(
        cache_dir: impl AsRef<Path>,
        patches_path: impl AsRef<Path>,
        storage: StorageMode,
    ) -> Result<Self, String> {
        load_dataset(cache_dir.as_ref(), patches_path.as_ref(), storage)
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn patch_size(&self) -> [usize; 3] {
        self.patch_size
    }

    pub unsafe fn fill_batch_u16_ptr(
        &self,
        start_index: usize,
        batch_size: usize,
        image_out: *mut f32,
        label_out: *mut u16,
    ) -> Result<usize, String> {
        if image_out.is_null() || label_out.is_null() {
            return Err("null output buffer".to_string());
        }
        if self.records.is_empty() {
            return Err("patch plan contains no records".to_string());
        }
        for batch_index in 0..batch_size {
            let record = self.records[(start_index + batch_index) % self.records.len()];
            let case = &self.cases[record.case_index];
            copy_patch_u16(
                case,
                record.start,
                self.patch_size,
                image_out,
                label_out,
                batch_index,
            )?;
        }
        Ok(batch_size)
    }

    pub unsafe fn fill_batch_f32_ptr(
        &self,
        start_index: usize,
        batch_size: usize,
        image_out: *mut f32,
        label_out: *mut f32,
    ) -> Result<usize, String> {
        if image_out.is_null() || label_out.is_null() {
            return Err("null output buffer".to_string());
        }
        if self.records.is_empty() {
            return Err("patch plan contains no records".to_string());
        }
        let patch_voxels = self.patch_size[0] * self.patch_size[1] * self.patch_size[2];
        let Some(total_values) = patch_voxels.checked_mul(batch_size) else {
            return Err("batch output size overflow".to_string());
        };
        let image_out = slice::from_raw_parts_mut(image_out, total_values);
        let label_out = slice::from_raw_parts_mut(label_out, total_values);
        image_out
            .par_chunks_mut(patch_voxels)
            .zip(label_out.par_chunks_mut(patch_voxels))
            .enumerate()
            .try_for_each(|(batch_index, (image_patch, label_patch))| {
                let record = self.records[(start_index + batch_index) % self.records.len()];
                let case = &self.cases[record.case_index];
                copy_patch_f32_labels(
                    case,
                    record.start,
                    self.patch_size,
                    image_patch,
                    label_patch,
                )
            })?;
        Ok(batch_size)
    }
}

#[derive(Debug)]
struct LoadedCase {
    case_id: String,
    shape: [usize; 3],
    storage: CaseStorage,
}

#[derive(Debug)]
enum CaseStorage {
    Resident {
        image: Vec<f32>,
        label_f32: Vec<f32>,
    },
    Chunked {
        image_mmap: Mmap,
        label_mmap: Mmap,
        chunk_shape: [usize; 3],
        chunk_grid: [usize; 3],
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageMode {
    Resident,
    Chunked,
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
        DatasetHandle::open(&cache_dir, &patches_path)
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
    handle.as_ref().map_or(0, DatasetHandle::len)
}

#[no_mangle]
/// Returns the patch x size for an opened dataset.
///
/// # Safety
///
/// `handle` must be null or a live pointer returned by `medkit_dataset_open`.
pub unsafe extern "C" fn medkit_dataset_patch_x(handle: *const DatasetHandle) -> usize {
    handle.as_ref().map_or(0, |dataset| dataset.patch_size()[0])
}

#[no_mangle]
/// Returns the patch y size for an opened dataset.
///
/// # Safety
///
/// `handle` must be null or a live pointer returned by `medkit_dataset_open`.
pub unsafe extern "C" fn medkit_dataset_patch_y(handle: *const DatasetHandle) -> usize {
    handle.as_ref().map_or(0, |dataset| dataset.patch_size()[1])
}

#[no_mangle]
/// Returns the patch z size for an opened dataset.
///
/// # Safety
///
/// `handle` must be null or a live pointer returned by `medkit_dataset_open`.
pub unsafe extern "C" fn medkit_dataset_patch_z(handle: *const DatasetHandle) -> usize {
    handle.as_ref().map_or(0, |dataset| dataset.patch_size()[2])
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
    dataset
        .fill_batch_u16_ptr(start_index, batch_size, image_out, label_out)
        .unwrap_or(0)
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
    dataset
        .fill_batch_f32_ptr(start_index, batch_size, image_out, label_out)
        .unwrap_or(0)
}

fn load_dataset(
    cache_dir: &Path,
    patches_path: &Path,
    storage: StorageMode,
) -> Result<DatasetHandle, String> {
    let manifest = read_cache_manifest(cache_dir).map_err(|error| error.to_string())?;
    let mut cases = Vec::with_capacity(manifest.cases.len());
    for case in &manifest.cases {
        cases.push(load_case(case, storage)?);
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

fn load_case(case: &CachedCase, storage: StorageMode) -> Result<LoadedCase, String> {
    let storage = match storage {
        StorageMode::Resident => CaseStorage::Resident {
            image: read_f32_volume(Path::new(&case.image_cache_path), case.shape)?,
            label_f32: read_u16_volume_as_f32(Path::new(&case.label_cache_path), case.shape)?,
        },
        StorageMode::Chunked => {
            let image_path = case
                .image_chunk_cache_path
                .as_ref()
                .ok_or_else(|| format!("missing image chunk cache for {}", case.case_id))?;
            let label_path = case
                .label_chunk_cache_path
                .as_ref()
                .ok_or_else(|| format!("missing label chunk cache for {}", case.case_id))?;
            let chunk_grid = case
                .chunk_grid
                .ok_or_else(|| format!("missing chunk grid for {}", case.case_id))?;
            CaseStorage::Chunked {
                image_mmap: mmap_file(Path::new(image_path))?,
                label_mmap: mmap_file(Path::new(label_path))?,
                chunk_shape: case.chunk_shape,
                chunk_grid,
            }
        }
    };
    Ok(LoadedCase {
        case_id: case.case_id.clone(),
        shape: case.shape,
        storage,
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

unsafe fn copy_patch_u16(
    case: &LoadedCase,
    start: [usize; 3],
    patch_size: [usize; 3],
    image_out: *mut f32,
    label_out: *mut u16,
    batch_index: usize,
) -> Result<(), String> {
    let (image, label_f32) = resident_case(case)?;
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
                image.as_ptr().add(source_start),
                image_out.add(destination_start),
                patch_size[0],
            );
            let label_row = &label_f32[source_start..source_start + patch_size[0]];
            for (offset, value) in label_row.iter().enumerate() {
                *label_out.add(destination_start + offset) = *value as u16;
            }
        }
    }
    Ok(())
}

fn copy_patch_f32_labels(
    case: &LoadedCase,
    start: [usize; 3],
    patch_size: [usize; 3],
    image_out: &mut [f32],
    label_out: &mut [f32],
) -> Result<(), String> {
    match &case.storage {
        CaseStorage::Resident { image, label_f32 } => {
            copy_patch_f32_labels_resident(
                case.shape, image, label_f32, start, patch_size, image_out, label_out,
            );
            Ok(())
        }
        CaseStorage::Chunked {
            image_mmap,
            label_mmap,
            chunk_shape,
            chunk_grid,
        } => copy_patch_f32_labels_chunked(
            case.shape,
            image_mmap,
            label_mmap,
            *chunk_shape,
            *chunk_grid,
            start,
            patch_size,
            image_out,
            label_out,
        ),
    }
}

fn copy_patch_f32_labels_resident(
    shape: [usize; 3],
    image: &[f32],
    label_f32: &[f32],
    start: [usize; 3],
    patch_size: [usize; 3],
    image_out: &mut [f32],
    label_out: &mut [f32],
) {
    for local_z in 0..patch_size[2] {
        let z = start[2] + local_z;
        for local_y in 0..patch_size[1] {
            let y = start[1] + local_y;
            let source_start = start[0] + shape[0] * (y + shape[1] * z);
            let destination_start = patch_size[0] * (local_y + patch_size[1] * local_z);
            let destination_end = destination_start + patch_size[0];
            let source_end = source_start + patch_size[0];
            image_out[destination_start..destination_end]
                .copy_from_slice(&image[source_start..source_end]);
            label_out[destination_start..destination_end]
                .copy_from_slice(&label_f32[source_start..source_end]);
        }
    }
}

fn copy_patch_f32_labels_chunked(
    shape: [usize; 3],
    image_mmap: &Mmap,
    label_mmap: &Mmap,
    chunk_shape: [usize; 3],
    chunk_grid: [usize; 3],
    start: [usize; 3],
    patch_size: [usize; 3],
    image_out: &mut [f32],
    label_out: &mut [f32],
) -> Result<(), String> {
    image_out.fill(0.0);
    label_out.fill(0.0);
    let end = [
        start[0] + patch_size[0],
        start[1] + patch_size[1],
        start[2] + patch_size[2],
    ];
    let chunk_min = [
        start[0] / chunk_shape[0],
        start[1] / chunk_shape[1],
        start[2] / chunk_shape[2],
    ];
    let chunk_max = [
        (end[0] - 1) / chunk_shape[0],
        (end[1] - 1) / chunk_shape[1],
        (end[2] - 1) / chunk_shape[2],
    ];
    let chunk_voxels = chunk_shape[0] * chunk_shape[1] * chunk_shape[2];
    for chunk_z in chunk_min[2]..=chunk_max[2] {
        for chunk_y in chunk_min[1]..=chunk_max[1] {
            for chunk_x in chunk_min[0]..=chunk_max[0] {
                let chunk_index = chunk_x + chunk_grid[0] * (chunk_y + chunk_grid[1] * chunk_z);
                let chunk_value_offset = chunk_index * chunk_voxels;
                let image_chunk = f32_mmap_values(image_mmap, chunk_value_offset, chunk_voxels)?;
                let label_chunk = u16_mmap_values(label_mmap, chunk_value_offset, chunk_voxels)?;
                copy_chunk_overlap(
                    shape,
                    chunk_shape,
                    [chunk_x, chunk_y, chunk_z],
                    start,
                    patch_size,
                    image_chunk,
                    label_chunk,
                    image_out,
                    label_out,
                );
            }
        }
    }
    Ok(())
}

fn copy_chunk_overlap(
    shape: [usize; 3],
    chunk_shape: [usize; 3],
    chunk_index: [usize; 3],
    patch_start: [usize; 3],
    patch_size: [usize; 3],
    image_chunk: &[f32],
    label_chunk: &[u16],
    image_out: &mut [f32],
    label_out: &mut [f32],
) {
    let chunk_start = [
        chunk_index[0] * chunk_shape[0],
        chunk_index[1] * chunk_shape[1],
        chunk_index[2] * chunk_shape[2],
    ];
    let patch_end = [
        patch_start[0] + patch_size[0],
        patch_start[1] + patch_size[1],
        patch_start[2] + patch_size[2],
    ];
    let overlap_start = [
        patch_start[0].max(chunk_start[0]),
        patch_start[1].max(chunk_start[1]),
        patch_start[2].max(chunk_start[2]),
    ];
    let overlap_end = [
        patch_end[0].min((chunk_start[0] + chunk_shape[0]).min(shape[0])),
        patch_end[1].min((chunk_start[1] + chunk_shape[1]).min(shape[1])),
        patch_end[2].min((chunk_start[2] + chunk_shape[2]).min(shape[2])),
    ];
    if overlap_start[0] >= overlap_end[0]
        || overlap_start[1] >= overlap_end[1]
        || overlap_start[2] >= overlap_end[2]
    {
        return;
    }
    let row = overlap_end[0] - overlap_start[0];
    for z in overlap_start[2]..overlap_end[2] {
        for y in overlap_start[1]..overlap_end[1] {
            let chunk_source = (overlap_start[0] - chunk_start[0])
                + chunk_shape[0] * ((y - chunk_start[1]) + chunk_shape[1] * (z - chunk_start[2]));
            let patch_dest = (overlap_start[0] - patch_start[0])
                + patch_size[0] * ((y - patch_start[1]) + patch_size[1] * (z - patch_start[2]));
            image_out[patch_dest..patch_dest + row]
                .copy_from_slice(&image_chunk[chunk_source..chunk_source + row]);
            for offset in 0..row {
                label_out[patch_dest + offset] = f32::from(label_chunk[chunk_source + offset]);
            }
        }
    }
}

fn f32_mmap_values(mmap: &Mmap, value_offset: usize, values: usize) -> Result<&[f32], String> {
    mmap_values(mmap, value_offset, values, 4)
        .map(|bytes| unsafe { slice::from_raw_parts(bytes.as_ptr() as *const f32, values) })
}

fn u16_mmap_values(mmap: &Mmap, value_offset: usize, values: usize) -> Result<&[u16], String> {
    mmap_values(mmap, value_offset, values, 2)
        .map(|bytes| unsafe { slice::from_raw_parts(bytes.as_ptr() as *const u16, values) })
}

#[cfg(target_endian = "big")]
fn mmap_values(
    mmap: &Mmap,
    value_offset: usize,
    values: usize,
    bytes_per_value: usize,
) -> Result<&[u8], String> {
    let _ = (mmap, value_offset, values, bytes_per_value);
    return Err("chunk mmap path currently requires a little-endian target".to_string());
}

#[cfg(target_endian = "little")]
fn mmap_values(
    mmap: &Mmap,
    value_offset: usize,
    values: usize,
    bytes_per_value: usize,
) -> Result<&[u8], String> {
    let byte_offset = value_offset
        .checked_mul(bytes_per_value)
        .ok_or_else(|| "mmap byte offset overflow".to_string())?;
    let byte_len = values
        .checked_mul(bytes_per_value)
        .ok_or_else(|| "mmap byte length overflow".to_string())?;
    let byte_end = byte_offset
        .checked_add(byte_len)
        .ok_or_else(|| "mmap byte range overflow".to_string())?;
    mmap.get(byte_offset..byte_end)
        .ok_or_else(|| "mmap chunk range is out of bounds".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_records_report_fill_errors() {
        let dataset = DatasetHandle {
            cases: Vec::new(),
            records: Vec::new(),
            patch_size: [1, 1, 1],
        };
        let mut image = [0.0_f32; 1];
        let mut label_u16 = [0_u16; 1];
        let mut label_f32 = [0.0_f32; 1];

        let u16_error =
            unsafe { dataset.fill_batch_u16_ptr(0, 1, image.as_mut_ptr(), label_u16.as_mut_ptr()) }
                .unwrap_err();
        assert_eq!(u16_error, "patch plan contains no records");

        let f32_error =
            unsafe { dataset.fill_batch_f32_ptr(0, 1, image.as_mut_ptr(), label_f32.as_mut_ptr()) }
                .unwrap_err();
        assert_eq!(f32_error, "patch plan contains no records");
    }

    #[test]
    fn copy_chunk_overlap_noops_when_patch_and_chunk_do_not_overlap() {
        let image_chunk = [1.0_f32; 1];
        let label_chunk = [7_u16; 1];
        let mut image_out = [0.0_f32; 1];
        let mut label_out = [0.0_f32; 1];

        copy_chunk_overlap(
            [2, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
            &image_chunk,
            &label_chunk,
            &mut image_out,
            &mut label_out,
        );

        assert_eq!(image_out, [0.0]);
        assert_eq!(label_out, [0.0]);
    }
}

fn mmap_file(path: &Path) -> Result<Mmap, String> {
    let file = fs::File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    unsafe { Mmap::map(&file).map_err(|error| format!("{}: {error}", path.display())) }
}

fn resident_case(case: &LoadedCase) -> Result<(&[f32], &[f32]), String> {
    match &case.storage {
        CaseStorage::Resident { image, label_f32 } => Ok((image, label_f32)),
        CaseStorage::Chunked { .. } => {
            Err("u16 pointer fill is not implemented for chunked storage".to_string())
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
