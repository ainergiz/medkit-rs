use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use medkit_cache::{CacheManifest, CacheSummary, CachedCase};
use medkit_sampler::{
    extract_patch_pair, extract_patch_pair_chunked_into, extract_patch_pair_into,
    extract_patch_pair_mmap_into, foreground_voxels_in_patch, load_chunked_cached_cases,
    load_mmap_cached_cases, plan_batches, CachedImageVolume, ForegroundPrefix, LoadedCachedCase,
    PatchRecord, SamplingStrategy,
};
use medkit_transform::{TransformPlan, Volume3D, VolumeGeometry};

#[test]
fn extracts_aligned_image_and_label_patches() {
    let image =
        CachedImageVolume::new([4, 4, 4], 1, (0..64).map(|value| value as f32).collect()).unwrap();
    let mut label_values = vec![0_u16; 64];
    label_values[1 + 4 * (1 + 4)] = 7;
    let label = Volume3D::new([4, 4, 4], label_values).unwrap();
    let foreground_prefix = ForegroundPrefix::from_label(&label).unwrap();
    let case = LoadedCachedCase {
        metadata: cached_case(),
        image,
        label,
        foreground_indices: vec![1 + 4 * (1 + 4)],
        foreground_prefix,
    };

    let patch = extract_patch_pair(&case, [0, 0, 0], [2, 2, 2]).unwrap();

    assert_eq!(patch.image.shape, [2, 2, 2]);
    assert_eq!(patch.label.shape, [2, 2, 2]);
    assert!(patch.has_foreground);
    assert_eq!(*patch.image.get(0, 1, 1, 1), 21.0);
    assert_eq!(*patch.label.get(1, 1, 1), 7);
}

#[test]
fn extracts_into_reusable_buffers_and_counts_foreground() {
    let image =
        CachedImageVolume::new([4, 4, 4], 1, (0..64).map(|value| value as f32).collect()).unwrap();
    let mut label_values = vec![0_u16; 64];
    label_values[1 + 4 * (1 + 4)] = 7;
    let label = Volume3D::new([4, 4, 4], label_values).unwrap();
    let foreground_prefix = ForegroundPrefix::from_label(&label).unwrap();
    let case = LoadedCachedCase {
        metadata: cached_case(),
        image,
        label,
        foreground_indices: vec![1 + 4 * (1 + 4)],
        foreground_prefix,
    };
    let mut image_out = vec![0.0; 8];
    let mut label_out = vec![0_u16; 8];

    let has_foreground =
        extract_patch_pair_into(&case, [0, 0, 0], [2, 2, 2], &mut image_out, &mut label_out)
            .unwrap();

    assert!(has_foreground);
    assert_eq!(
        foreground_voxels_in_patch(&case, [0, 0, 0], [2, 2, 2]).unwrap(),
        1
    );
    assert_eq!(image_out[7], 21.0);
    assert_eq!(label_out[7], 7);
}

#[test]
fn extracts_multichannel_image_patches_channel_major() {
    let image = CachedImageVolume::new(
        [2, 2, 1],
        2,
        vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
    )
    .unwrap();
    let label = Volume3D::new([2, 2, 1], vec![0_u16, 1, 0, 2]).unwrap();
    let foreground_prefix = ForegroundPrefix::from_label(&label).unwrap();
    let case = LoadedCachedCase {
        metadata: cached_case(),
        image,
        label,
        foreground_indices: vec![1, 3],
        foreground_prefix,
    };
    let mut image_out = vec![0.0; 8];
    let mut label_out = vec![0_u16; 4];

    let has_foreground =
        extract_patch_pair_into(&case, [0, 0, 0], [2, 2, 1], &mut image_out, &mut label_out)
            .unwrap();

    assert!(has_foreground);
    assert_eq!(image_out, vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]);
    assert_eq!(label_out, vec![0, 1, 0, 2]);
}

#[test]
fn plans_batches_without_python() {
    let records = (0..5)
        .map(|index| PatchRecord {
            index,
            case_id: "case".to_string(),
            patch_start: [index, 0, 0],
            patch_size: [2, 2, 2],
            has_foreground: index % 2 == 0,
            strategy: SamplingStrategy::ForegroundBalanced,
            epoch: 0,
            worker: 0,
        })
        .collect::<Vec<_>>();

    let plan = plan_batches(records, 2).unwrap();

    assert_eq!(plan.batch_size, 2);
    assert_eq!(plan.batches.len(), 3);
    assert_eq!(plan.batches[0].len(), 2);
    assert_eq!(plan.batches[2].len(), 1);
}

#[test]
fn mmap_resident_extraction_reads_patch_without_loading_case_volume() {
    let fixture = CacheFixture::new("mmap-resident");
    let cases = load_mmap_cached_cases(&fixture.cache_dir).unwrap();
    let mut image = vec![0.0_f32; 9];
    let mut label = vec![0_u16; 9];

    let has_foreground =
        extract_patch_pair_mmap_into(&cases[0], [1, 1, 0], [3, 3, 1], &mut image, &mut label)
            .unwrap();

    assert!(has_foreground);
    assert_eq!(
        image,
        vec![5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0]
    );
    assert_eq!(label, vec![5, 6, 7, 9, 10, 11, 13, 14, 15]);
}

#[test]
fn mmap_resident_extraction_reads_multichannel_patch() {
    let fixture = CacheFixture::new_with_channels("mmap-multichannel", 2);
    let cases = load_mmap_cached_cases(&fixture.cache_dir).unwrap();
    let mut image = vec![0.0_f32; 18];
    let mut label = vec![0_u16; 9];

    extract_patch_pair_mmap_into(&cases[0], [1, 1, 0], [3, 3, 1], &mut image, &mut label).unwrap();

    let expected = vec![5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0];
    assert_eq!(&image[..9], expected.as_slice());
    assert_eq!(
        &image[9..],
        expected
            .iter()
            .map(|value| value + 100.0)
            .collect::<Vec<_>>()
            .as_slice()
    );
    assert_eq!(label, vec![5, 6, 7, 9, 10, 11, 13, 14, 15]);
}

#[test]
fn chunked_extraction_fills_patch_across_chunk_boundaries() {
    let fixture = CacheFixture::new("chunked-span");
    let cases = load_chunked_cached_cases(&fixture.cache_dir).unwrap();
    let mut image = vec![0.0_f32; 9];
    let mut label = vec![0_u16; 9];

    let has_foreground =
        extract_patch_pair_chunked_into(&cases[0], [1, 1, 0], [3, 3, 1], &mut image, &mut label)
            .unwrap();

    assert!(has_foreground);
    assert_eq!(cases[0].chunk_shape, [2, 2, 1]);
    assert_eq!(cases[0].chunk_grid, [2, 2, 1]);
    assert_eq!(
        image,
        vec![5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0]
    );
    assert_eq!(label, vec![5, 6, 7, 9, 10, 11, 13, 14, 15]);
}

#[test]
fn chunked_extraction_reads_multichannel_patch() {
    let fixture = CacheFixture::new_with_channels("chunked-multichannel", 2);
    let cases = load_chunked_cached_cases(&fixture.cache_dir).unwrap();
    let mut image = vec![0.0_f32; 18];
    let mut label = vec![0_u16; 9];

    extract_patch_pair_chunked_into(&cases[0], [1, 1, 0], [3, 3, 1], &mut image, &mut label)
        .unwrap();

    let expected = vec![5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0];
    assert_eq!(&image[..9], expected.as_slice());
    assert_eq!(
        &image[9..],
        expected
            .iter()
            .map(|value| value + 100.0)
            .collect::<Vec<_>>()
            .as_slice()
    );
    assert_eq!(label, vec![5, 6, 7, 9, 10, 11, 13, 14, 15]);
}

fn cached_case() -> CachedCase {
    let geometry = VolumeGeometry::identity([4, 4, 4], [1.0, 1.0, 1.0]).unwrap();
    CachedCase {
        case_id: "case".to_string(),
        cache_key: "case-key".to_string(),
        source_metadata_hash: "source".to_string(),
        transform_plan_hash: "plan".to_string(),
        image_path: "image.nii".to_string(),
        image_paths: vec!["image.nii".to_string()],
        image_channel_count: 1,
        label_path: "label.nii".to_string(),
        source_geometry: geometry,
        output_geometry: geometry,
        image_cache_path: "image.raw".to_string(),
        image_cache_sha256: String::new(),
        label_cache_path: "label.raw".to_string(),
        label_cache_sha256: String::new(),
        foreground_indices_path: None,
        foreground_indices_sha256: None,
        foreground_prefix_path: None,
        foreground_prefix_sha256: None,
        foreground_prefix_shape: None,
        image_chunk_cache_path: None,
        image_chunk_cache_sha256: None,
        label_chunk_cache_path: None,
        label_chunk_cache_sha256: None,
        shape: [4, 4, 4],
        chunk_shape: [4, 4, 4],
        chunk_grid: None,
        crop_origin: [0, 0, 0],
        applied_operations: vec!["ct_window".to_string()],
        foreground_voxels: 1,
        bytes_written: 64 * 6,
    }
}

struct CacheFixture {
    cache_dir: PathBuf,
}

impl CacheFixture {
    fn new(name: &str) -> Self {
        Self::new_with_channels(name, 1)
    }

    fn new_with_channels(name: &str, image_channel_count: usize) -> Self {
        const SHAPE: [usize; 3] = [4, 4, 1];
        const CHUNK_SHAPE: [usize; 3] = [2, 2, 1];
        let cache_dir = temp_dir(name);
        let case_dir = cache_dir.join("case-key");
        fs::create_dir_all(&case_dir).unwrap();
        let image_path = case_dir.join("image.f32.raw");
        let label_path = case_dir.join("label.u16.raw");
        let image_chunk_path = case_dir.join("image.chunks.f32.raw");
        let label_chunk_path = case_dir.join("label.chunks.u16.raw");
        let image_values = (0..image_channel_count)
            .flat_map(|channel| (0..16).map(move |value| value as f32 + channel as f32 * 100.0))
            .collect::<Vec<_>>();
        let label_values = (0..16).map(|value| value as u16).collect::<Vec<_>>();
        write_f32(&image_path, &image_values);
        write_u16(&label_path, &label_values);
        let image_chunks = image_values
            .chunks_exact(16)
            .flat_map(|values| chunked_values(values, SHAPE, CHUNK_SHAPE, 0.0))
            .collect::<Vec<_>>();
        write_f32(&image_chunk_path, &image_chunks);
        write_u16(
            &label_chunk_path,
            &chunked_values(&label_values, SHAPE, CHUNK_SHAPE, 0),
        );

        let geometry = VolumeGeometry::identity(SHAPE, [1.0, 1.0, 1.0]).unwrap();
        let case = CachedCase {
            case_id: "case".to_string(),
            cache_key: "case-key".to_string(),
            source_metadata_hash: "source".to_string(),
            transform_plan_hash: "plan".to_string(),
            image_path: "image.nii".to_string(),
            image_paths: (0..image_channel_count)
                .map(|channel| format!("image_{channel:04}.nii"))
                .collect(),
            image_channel_count,
            label_path: "label.nii".to_string(),
            source_geometry: geometry,
            output_geometry: geometry,
            image_cache_path: path_string(&image_path),
            image_cache_sha256: String::new(),
            label_cache_path: path_string(&label_path),
            label_cache_sha256: String::new(),
            foreground_indices_path: None,
            foreground_indices_sha256: None,
            foreground_prefix_path: None,
            foreground_prefix_sha256: None,
            foreground_prefix_shape: None,
            image_chunk_cache_path: Some(path_string(&image_chunk_path)),
            image_chunk_cache_sha256: None,
            label_chunk_cache_path: Some(path_string(&label_chunk_path)),
            label_chunk_cache_sha256: None,
            shape: SHAPE,
            chunk_shape: CHUNK_SHAPE,
            chunk_grid: Some([2, 2, 1]),
            crop_origin: [0, 0, 0],
            applied_operations: Vec::new(),
            foreground_voxels: 15,
            bytes_written: image_values.len() * 4 + 16 * 2 + image_chunks.len() * 4 + 16 * 2,
        };
        let manifest = CacheManifest {
            version: 1,
            cache_dir: path_string(&cache_dir),
            dataset_manifest_path: "manifest.json".to_string(),
            transform_plan_hash: "plan".to_string(),
            transform_plan: TransformPlan::from_toml_str(
                r#"
name = "identity"
image_interpolation = "linear"
label_interpolation = "nearest"
operations = []
"#,
            )
            .unwrap(),
            summary: CacheSummary {
                input_cases: 1,
                cached_cases: 1,
                failed_cases: 0,
                foreground_voxels: 15,
                bytes_written: case.bytes_written,
            },
            cases: vec![case],
        };
        fs::write(
            cache_dir.join("cache_manifest.json"),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();
        Self { cache_dir }
    }
}

fn temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "medkit-sampler-{name}-{}-{nanos}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_f32(path: &Path, values: &[f32]) {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    fs::write(path, bytes).unwrap();
}

fn write_u16(path: &Path, values: &[u16]) {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    fs::write(path, bytes).unwrap();
}

fn chunked_values<T: Copy>(
    values: &[T],
    shape: [usize; 3],
    chunk_shape: [usize; 3],
    fill: T,
) -> Vec<T> {
    let grid = [
        shape[0].div_ceil(chunk_shape[0]),
        shape[1].div_ceil(chunk_shape[1]),
        shape[2].div_ceil(chunk_shape[2]),
    ];
    let mut out = Vec::new();
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
                                values[x + shape[0] * (y + shape[1] * z)]
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

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}
