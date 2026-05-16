use std::{
    ffi::CString,
    fs,
    path::{Path, PathBuf},
    ptr,
    time::{SystemTime, UNIX_EPOCH},
};

use medkit_python_ffi::{
    medkit_dataset_fill_batch, medkit_dataset_fill_batch_f32_labels, medkit_dataset_free,
    medkit_dataset_len, medkit_dataset_open, medkit_dataset_patch_x, medkit_dataset_patch_y,
    medkit_dataset_patch_z, DatasetHandle, StorageMode,
};

const CASE_ID: &str = "case_a";
const SHAPE: [usize; 3] = [4, 3, 2];
const CHUNK_SHAPE: [usize; 3] = [2, 2, 1];
const PATCH_SIZE: [usize; 3] = [2, 2, 2];
const FIRST_PATCH_START: [usize; 3] = [1, 1, 0];
const SECOND_PATCH_START: [usize; 3] = [0, 0, 0];

#[test]
fn open_reports_manifest_and_patch_plan_errors() {
    let temp = TempDir::new("open-errors");
    let missing_cache = temp.path().join("missing-cache");
    let missing_patches = temp.path().join("missing-patches.jsonl");
    let err = DatasetHandle::open(&missing_cache, &missing_patches).unwrap_err();
    assert!(err.contains("cache_manifest.json"), "{err}");

    let fixture = TinyCache::new("open-errors-fixture");
    let empty_patches = fixture.root.path().join("empty-patches.jsonl");
    fs::write(&empty_patches, "").unwrap();
    let err = DatasetHandle::open(&fixture.cache_dir, &empty_patches).unwrap_err();
    assert!(err.contains("patch plan has no records"), "{err}");

    let missing_case_patches = fixture.root.path().join("missing-case-patches.jsonl");
    fs::write(
        &missing_case_patches,
        r#"{"case_id":"missing","patch_start":[0,0,0],"patch_size":[2,2,2]}"#,
    )
    .unwrap();
    let err = DatasetHandle::open(&fixture.cache_dir, &missing_case_patches).unwrap_err();
    assert!(err.contains("missing cached case missing"), "{err}");

    let mixed_size_patches = fixture.root.path().join("mixed-size-patches.jsonl");
    fs::write(
        &mixed_size_patches,
        format!(
            "{}\n{}\n",
            patch_record(CASE_ID, FIRST_PATCH_START, PATCH_SIZE),
            patch_record(CASE_ID, SECOND_PATCH_START, [1, 2, 2])
        ),
    )
    .unwrap();
    let err = DatasetHandle::open(&fixture.cache_dir, &mixed_size_patches).unwrap_err();
    assert!(err.contains("mixed patch sizes are not supported"), "{err}");
}

#[test]
fn resident_storage_fills_u16_and_f32_batches() {
    let fixture = TinyCache::new("resident-fill");
    let dataset = DatasetHandle::open_with_storage(
        &fixture.cache_dir,
        &fixture.patches_path,
        StorageMode::Resident,
    )
    .unwrap();
    assert_eq!(dataset.len(), 2);
    assert!(!dataset.is_empty());
    assert_eq!(dataset.patch_size(), PATCH_SIZE);

    let patch_voxels = volume_len(PATCH_SIZE);
    let mut image = vec![0.0; patch_voxels * 2];
    let mut label = vec![0.0; patch_voxels * 2];
    let filled =
        unsafe { dataset.fill_batch_f32_ptr(0, 2, image.as_mut_ptr(), label.as_mut_ptr()) };
    assert_eq!(filled.unwrap(), 2);
    assert_eq!(
        image,
        [
            expected_image_patch(FIRST_PATCH_START),
            expected_image_patch(SECOND_PATCH_START),
        ]
        .concat()
    );
    assert_eq!(
        label,
        [
            expected_label_patch_f32(FIRST_PATCH_START),
            expected_label_patch_f32(SECOND_PATCH_START),
        ]
        .concat()
    );

    let mut image = vec![0.0; patch_voxels];
    let mut label = vec![0_u16; patch_voxels];
    let filled =
        unsafe { dataset.fill_batch_u16_ptr(1, 1, image.as_mut_ptr(), label.as_mut_ptr()) };
    assert_eq!(filled.unwrap(), 1);
    assert_eq!(image, expected_image_patch(SECOND_PATCH_START));
    assert_eq!(label, expected_label_patch_u16(SECOND_PATCH_START));
}

#[test]
fn chunked_storage_fills_patch_across_chunk_boundaries() {
    let fixture = TinyCache::new("chunked-fill");
    let dataset = DatasetHandle::open_with_storage(
        &fixture.cache_dir,
        &fixture.patches_path,
        StorageMode::Chunked,
    )
    .unwrap();
    assert_eq!(dataset.len(), 2);
    assert_eq!(dataset.patch_size(), PATCH_SIZE);

    let patch_voxels = volume_len(PATCH_SIZE);
    let mut image = vec![0.0; patch_voxels];
    let mut label = vec![0.0; patch_voxels];
    let filled =
        unsafe { dataset.fill_batch_f32_ptr(0, 1, image.as_mut_ptr(), label.as_mut_ptr()) };
    assert_eq!(filled.unwrap(), 1);
    assert_eq!(image, expected_image_patch(FIRST_PATCH_START));
    assert_eq!(label, expected_label_patch_f32(FIRST_PATCH_START));

    let mut u16_label = vec![0_u16; patch_voxels];
    let err =
        unsafe { dataset.fill_batch_u16_ptr(0, 1, image.as_mut_ptr(), u16_label.as_mut_ptr()) }
            .unwrap_err();
    assert!(
        err.contains("u16 pointer fill is not implemented for chunked storage"),
        "{err}"
    );
}

#[test]
fn null_pointers_and_buffer_size_errors_are_reported() {
    let fixture = TinyCache::new("pointer-errors");
    let dataset = DatasetHandle::open_with_storage(
        &fixture.cache_dir,
        &fixture.patches_path,
        StorageMode::Resident,
    )
    .unwrap();
    let patch_voxels = volume_len(PATCH_SIZE);
    let mut image = vec![0.0; patch_voxels];
    let mut label_f32 = vec![0.0; patch_voxels];
    let mut label_u16 = vec![0_u16; patch_voxels];

    let err = unsafe { dataset.fill_batch_f32_ptr(0, 1, ptr::null_mut(), label_f32.as_mut_ptr()) }
        .unwrap_err();
    assert_eq!(err, "null output buffer");

    let err = unsafe { dataset.fill_batch_u16_ptr(0, 1, image.as_mut_ptr(), ptr::null_mut()) }
        .unwrap_err();
    assert_eq!(err, "null output buffer");

    let err = unsafe {
        dataset.fill_batch_f32_ptr(0, usize::MAX, image.as_mut_ptr(), label_f32.as_mut_ptr())
    }
    .unwrap_err();
    assert_eq!(err, "batch output size overflow");

    let fixture = TinyCache::new("short-resident-buffer");
    fs::write(&fixture.image_path, [0_u8; 3]).unwrap();
    let err = DatasetHandle::open_with_storage(
        &fixture.cache_dir,
        &fixture.patches_path,
        StorageMode::Resident,
    )
    .unwrap_err();
    assert!(err.contains("has 3 bytes, expected 96"), "{err}");

    let fixture = TinyCache::new("short-resident-label-buffer");
    fs::write(&fixture.label_path, [0_u8; 3]).unwrap();
    let err = DatasetHandle::open_with_storage(
        &fixture.cache_dir,
        &fixture.patches_path,
        StorageMode::Resident,
    )
    .unwrap_err();
    assert!(err.contains("has 3 bytes, expected 48"), "{err}");

    let fixture = TinyCache::new("short-chunk-buffer");
    fs::write(&fixture.label_chunk_path, [0_u8; 2]).unwrap();
    let dataset = DatasetHandle::open_with_storage(
        &fixture.cache_dir,
        &fixture.patches_path,
        StorageMode::Chunked,
    )
    .unwrap();
    let err =
        unsafe { dataset.fill_batch_f32_ptr(0, 1, image.as_mut_ptr(), label_f32.as_mut_ptr()) }
            .unwrap_err();
    assert!(err.contains("mmap chunk range is out of bounds"), "{err}");

    let opened = unsafe { medkit_dataset_open(ptr::null(), ptr::null()) };
    assert!(opened.is_null());

    let cache_c = CString::new(
        fixture
            .root
            .path()
            .join("missing-cache")
            .to_string_lossy()
            .as_bytes(),
    )
    .unwrap();
    let patches_c = CString::new(fixture.patches_path.to_string_lossy().as_bytes()).unwrap();
    let opened = unsafe { medkit_dataset_open(cache_c.as_ptr(), patches_c.as_ptr()) };
    assert!(opened.is_null());

    let filled = unsafe {
        medkit_dataset_fill_batch(
            ptr::null(),
            0,
            1,
            image.as_mut_ptr(),
            label_u16.as_mut_ptr(),
        )
    };
    assert_eq!(filled, 0);

    let filled = unsafe {
        medkit_dataset_fill_batch_f32_labels(
            ptr::null(),
            0,
            1,
            image.as_mut_ptr(),
            label_f32.as_mut_ptr(),
        )
    };
    assert_eq!(filled, 0);
}

#[test]
fn c_abi_opens_reports_shape_fills_and_frees() {
    let fixture = TinyCache::new("c-abi");
    let cache_c = CString::new(fixture.cache_dir.to_string_lossy().as_bytes()).unwrap();
    let patches_c = CString::new(fixture.patches_path.to_string_lossy().as_bytes()).unwrap();
    let handle = unsafe { medkit_dataset_open(cache_c.as_ptr(), patches_c.as_ptr()) };
    assert!(!handle.is_null());

    assert_eq!(unsafe { medkit_dataset_len(handle) }, 2);
    assert_eq!(unsafe { medkit_dataset_patch_x(handle) }, PATCH_SIZE[0]);
    assert_eq!(unsafe { medkit_dataset_patch_y(handle) }, PATCH_SIZE[1]);
    assert_eq!(unsafe { medkit_dataset_patch_z(handle) }, PATCH_SIZE[2]);

    let patch_voxels = volume_len(PATCH_SIZE);
    let mut image = vec![0.0; patch_voxels];
    let mut label_u16 = vec![0_u16; patch_voxels];
    let filled = unsafe {
        medkit_dataset_fill_batch(handle, 0, 1, image.as_mut_ptr(), label_u16.as_mut_ptr())
    };
    assert_eq!(filled, 1);
    assert_eq!(image, expected_image_patch(FIRST_PATCH_START));
    assert_eq!(label_u16, expected_label_patch_u16(FIRST_PATCH_START));

    let mut label_f32 = vec![0.0; patch_voxels];
    let filled = unsafe {
        medkit_dataset_fill_batch_f32_labels(
            handle,
            1,
            1,
            image.as_mut_ptr(),
            label_f32.as_mut_ptr(),
        )
    };
    assert_eq!(filled, 1);
    assert_eq!(image, expected_image_patch(SECOND_PATCH_START));
    assert_eq!(label_f32, expected_label_patch_f32(SECOND_PATCH_START));

    let filled = unsafe {
        medkit_dataset_fill_batch_f32_labels(handle, 0, 1, ptr::null_mut(), label_f32.as_mut_ptr())
    };
    assert_eq!(filled, 0);

    unsafe { medkit_dataset_free(handle) };
}

struct TinyCache {
    root: TempDir,
    cache_dir: PathBuf,
    patches_path: PathBuf,
    image_path: PathBuf,
    label_path: PathBuf,
    label_chunk_path: PathBuf,
}

impl TinyCache {
    fn new(name: &str) -> Self {
        let root = TempDir::new(name);
        let cache_dir = root.path().join("cache");
        let case_dir = cache_dir.join("case-a-key");
        fs::create_dir_all(&case_dir).unwrap();

        let image_path = case_dir.join("image.f32.raw");
        let label_path = case_dir.join("label.u16.raw");
        let image_chunk_path = case_dir.join("image.chunks.f32.raw");
        let label_chunk_path = case_dir.join("label.chunks.u16.raw");
        let image_values = image_values();
        let label_values = label_values();
        write_f32_raw(&image_path, &image_values);
        write_u16_raw(&label_path, &label_values);
        write_f32_raw(
            &image_chunk_path,
            &chunked_values(&image_values, SHAPE, CHUNK_SHAPE, 0.0),
        );
        write_u16_raw(
            &label_chunk_path,
            &chunked_values(&label_values, SHAPE, CHUNK_SHAPE, 0),
        );

        let manifest = serde_json::json!({
            "version": 1,
            "cache_dir": path_string(&cache_dir),
            "dataset_manifest_path": path_string(&root.path().join("manifest.json")),
            "transform_plan_hash": "test-plan-hash",
            "transform_plan": {
                "name": "test-plan",
                "operations": [],
                "image_interpolation": "linear",
                "label_interpolation": "nearest"
            },
            "summary": {
                "input_cases": 1,
                "cached_cases": 1,
                "failed_cases": 0,
                "foreground_voxels": label_values.iter().filter(|value| **value != 0).count(),
                "bytes_written": image_values.len() * 4 + label_values.len() * 2
            },
            "cases": [{
                "case_id": CASE_ID,
                "cache_key": "case-a-key",
                "source_metadata_hash": "source-hash",
                "transform_plan_hash": "test-plan-hash",
                "image_path": path_string(&root.path().join("source-image.nii")),
                "label_path": path_string(&root.path().join("source-label.nii")),
                "source_geometry": geometry_json(),
                "output_geometry": geometry_json(),
                "image_cache_path": path_string(&image_path),
                "label_cache_path": path_string(&label_path),
                "image_chunk_cache_path": path_string(&image_chunk_path),
                "label_chunk_cache_path": path_string(&label_chunk_path),
                "chunk_grid": chunk_grid(SHAPE, CHUNK_SHAPE),
                "shape": SHAPE,
                "chunk_shape": CHUNK_SHAPE,
                "crop_origin": [0, 0, 0],
                "applied_operations": [],
                "foreground_voxels": label_values.iter().filter(|value| **value != 0).count(),
                "bytes_written": image_values.len() * 4 + label_values.len() * 2
            }]
        });
        fs::write(
            cache_dir.join("cache_manifest.json"),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let patches_path = root.path().join("patches.jsonl");
        fs::write(
            &patches_path,
            format!(
                "{}\n{}\n",
                patch_record(CASE_ID, FIRST_PATCH_START, PATCH_SIZE),
                patch_record(CASE_ID, SECOND_PATCH_START, PATCH_SIZE)
            ),
        )
        .unwrap();

        Self {
            root,
            cache_dir,
            patches_path,
            image_path,
            label_path,
            label_chunk_path,
        }
    }
}

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(name: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after UNIX_EPOCH")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "medkit-python-ffi-{name}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn patch_record(case_id: &str, patch_start: [usize; 3], patch_size: [usize; 3]) -> String {
    serde_json::json!({
        "case_id": case_id,
        "patch_start": patch_start,
        "patch_size": patch_size
    })
    .to_string()
}

fn expected_image_patch(start: [usize; 3]) -> Vec<f32> {
    expected_patch(start, |index| index as f32 + 0.25)
}

fn expected_label_patch_f32(start: [usize; 3]) -> Vec<f32> {
    expected_patch(start, |index| f32::from(label_value(index)))
}

fn expected_label_patch_u16(start: [usize; 3]) -> Vec<u16> {
    expected_patch(start, label_value)
}

fn expected_patch<T>(start: [usize; 3], value: impl Fn(usize) -> T) -> Vec<T> {
    let mut out = Vec::with_capacity(volume_len(PATCH_SIZE));
    for local_z in 0..PATCH_SIZE[2] {
        for local_y in 0..PATCH_SIZE[1] {
            for local_x in 0..PATCH_SIZE[0] {
                let index = flat_index(
                    start[0] + local_x,
                    start[1] + local_y,
                    start[2] + local_z,
                    SHAPE,
                );
                out.push(value(index));
            }
        }
    }
    out
}

fn image_values() -> Vec<f32> {
    (0..volume_len(SHAPE))
        .map(|index| index as f32 + 0.25)
        .collect()
}

fn label_values() -> Vec<u16> {
    (0..volume_len(SHAPE)).map(label_value).collect()
}

fn label_value(index: usize) -> u16 {
    index as u16 + 100
}

fn write_f32_raw(path: &Path, values: &[f32]) {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).unwrap();
}

fn write_u16_raw(path: &Path, values: &[u16]) {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).unwrap();
}

fn chunked_values<T: Copy>(
    values: &[T],
    shape: [usize; 3],
    chunk_shape: [usize; 3],
    fill: T,
) -> Vec<T> {
    let grid = chunk_grid(shape, chunk_shape);
    let mut out = Vec::with_capacity(
        grid[0] * grid[1] * grid[2] * chunk_shape[0] * chunk_shape[1] * chunk_shape[2],
    );
    for chunk_z in 0..grid[2] {
        for chunk_y in 0..grid[1] {
            for chunk_x in 0..grid[0] {
                let start = [
                    chunk_x * chunk_shape[0],
                    chunk_y * chunk_shape[1],
                    chunk_z * chunk_shape[2],
                ];
                for local_z in 0..chunk_shape[2] {
                    let z = start[2] + local_z;
                    for local_y in 0..chunk_shape[1] {
                        let y = start[1] + local_y;
                        for local_x in 0..chunk_shape[0] {
                            let x = start[0] + local_x;
                            let value = if x < shape[0] && y < shape[1] && z < shape[2] {
                                values[flat_index(x, y, z, shape)]
                            } else {
                                fill
                            };
                            out.push(value);
                        }
                    }
                }
            }
        }
    }
    out
}

fn chunk_grid(shape: [usize; 3], chunk_shape: [usize; 3]) -> [usize; 3] {
    [
        div_ceil(shape[0], chunk_shape[0]),
        div_ceil(shape[1], chunk_shape[1]),
        div_ceil(shape[2], chunk_shape[2]),
    ]
}

fn div_ceil(value: usize, divisor: usize) -> usize {
    (value + divisor - 1) / divisor
}

fn flat_index(x: usize, y: usize, z: usize, shape: [usize; 3]) -> usize {
    x + shape[0] * (y + shape[1] * z)
}

fn volume_len(shape: [usize; 3]) -> usize {
    shape[0] * shape[1] * shape[2]
}

fn geometry_json() -> serde_json::Value {
    serde_json::json!({
        "shape": SHAPE,
        "spacing": [1.0, 1.0, 1.0],
        "origin": [0.0, 0.0, 0.0],
        "direction": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    })
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}
