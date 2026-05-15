use medkit_cache::CachedCase;
use medkit_sampler::{
    extract_patch_pair, extract_patch_pair_into, foreground_voxels_in_patch, plan_batches,
    ForegroundPrefix, LoadedCachedCase, PatchRecord, SamplingStrategy,
};
use medkit_transform::{Volume3D, VolumeGeometry};

#[test]
fn extracts_aligned_image_and_label_patches() {
    let image = Volume3D::new([4, 4, 4], (0..64).map(|value| value as f32).collect()).unwrap();
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
    assert_eq!(*patch.image.get(1, 1, 1), 21.0);
    assert_eq!(*patch.label.get(1, 1, 1), 7);
}

#[test]
fn extracts_into_reusable_buffers_and_counts_foreground() {
    let image = Volume3D::new([4, 4, 4], (0..64).map(|value| value as f32).collect()).unwrap();
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

fn cached_case() -> CachedCase {
    let geometry = VolumeGeometry::identity([4, 4, 4], [1.0, 1.0, 1.0]).unwrap();
    CachedCase {
        case_id: "case".to_string(),
        cache_key: "case-key".to_string(),
        source_metadata_hash: "source".to_string(),
        transform_plan_hash: "plan".to_string(),
        image_path: "image.nii".to_string(),
        label_path: "label.nii".to_string(),
        source_geometry: geometry,
        output_geometry: geometry,
        image_cache_path: "image.raw".to_string(),
        label_cache_path: "label.raw".to_string(),
        foreground_indices_path: None,
        foreground_prefix_path: None,
        foreground_prefix_shape: None,
        image_chunk_cache_path: None,
        label_chunk_cache_path: None,
        shape: [4, 4, 4],
        chunk_shape: [4, 4, 4],
        chunk_grid: None,
        crop_origin: [0, 0, 0],
        applied_operations: vec!["ct_window".to_string()],
        foreground_voxels: 1,
        bytes_written: 64 * 6,
    }
}
