use std::{
    fs,
    path::{Path, PathBuf},
};

use medkit_cache::{read_cache_manifest, CachedCase};
use medkit_transform::Volume3D;
use serde::{Deserialize, Serialize};

use crate::{
    rng::{seed_for, SplitMix64},
    Result, SamplerError,
};

/// Supported patch sampling strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingStrategy {
    /// Alternate foreground-centered and background/random patches.
    ForegroundBalanced,
}

/// Configuration for JSONL patch sampling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleConfig {
    /// Cache directory.
    pub cache_dir: PathBuf,
    /// Patch size in x, y, z order.
    pub patch_size: [usize; 3],
    /// Sampling strategy.
    pub strategy: SamplingStrategy,
    /// Number of patch records to emit.
    pub count: usize,
    /// Output JSONL path.
    pub out_path: PathBuf,
    /// Deterministic base seed.
    pub seed: u64,
    /// Training epoch component of the deterministic seed.
    pub epoch: u64,
    /// Worker id component of the deterministic seed.
    pub worker: u64,
}

/// Summary returned by sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SampleSummary {
    /// Records written.
    pub records: usize,
    /// Foreground records written.
    pub foreground_records: usize,
    /// Background records written.
    pub background_records: usize,
}

/// JSONL patch record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatchRecord {
    /// Sample index.
    pub index: usize,
    /// Case id.
    pub case_id: String,
    /// Patch start in x, y, z order.
    pub patch_start: [usize; 3],
    /// Patch size in x, y, z order.
    pub patch_size: [usize; 3],
    /// Whether the extracted patch contains foreground label voxels.
    pub has_foreground: bool,
    /// Sampling strategy.
    pub strategy: SamplingStrategy,
    /// Epoch used in deterministic seeding.
    pub epoch: u64,
    /// Worker used in deterministic seeding.
    pub worker: u64,
}

/// A Python-free batch plan made from sampled patch records.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchPlan {
    /// Requested batch size.
    pub batch_size: usize,
    /// Planned batches of patch records.
    pub batches: Vec<Vec<PatchRecord>>,
}

/// Loaded cached case with image, label, and foreground index.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedCachedCase {
    /// Cached case metadata.
    pub metadata: CachedCase,
    /// Cached image volume.
    pub image: Volume3D<f32>,
    /// Cached label volume.
    pub label: Volume3D<u16>,
    /// Flat indices of non-zero label voxels.
    pub foreground_indices: Vec<usize>,
    /// Integral volume for O(1) foreground counts.
    pub foreground_prefix: ForegroundPrefix,
}

#[derive(Debug, Clone, PartialEq)]
struct SamplingCase {
    metadata: CachedCase,
    shape: [usize; 3],
    foreground_indices: Vec<usize>,
    foreground_prefix: ForegroundPrefix,
}

/// Aligned image/label patch payload.
#[derive(Debug, Clone, PartialEq)]
pub struct CachedPatch {
    /// Patch image values.
    pub image: Volume3D<f32>,
    /// Patch label values.
    pub label: Volume3D<u16>,
    /// Whether the patch contains non-zero labels.
    pub has_foreground: bool,
}

/// Integral foreground volume for fast patch occupancy checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForegroundPrefix {
    /// Source shape in x, y, z order.
    pub shape: [usize; 3],
    prefix_shape: [usize; 3],
    values: Vec<u32>,
}

/// Loads cached cases into memory.
pub fn load_cached_cases(cache_dir: impl AsRef<Path>) -> Result<Vec<LoadedCachedCase>> {
    let manifest = read_cache_manifest(cache_dir)?;
    manifest.cases.iter().map(load_case).collect()
}

/// Samples patch records from a cache and writes JSONL.
pub fn sample_cache(config: &SampleConfig) -> Result<SampleSummary> {
    validate_patch(config.patch_size)?;
    let cases = load_sampling_cases(&config.cache_dir)?;
    if cases.is_empty() {
        return Err(SamplerError::invalid_input("cache contains no cases"));
    }
    if let Some(parent) = config.out_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|source| SamplerError::io(parent, source))?;
        }
    }
    let mut text = String::new();
    let mut summary = SampleSummary {
        records: config.count,
        foreground_records: 0,
        background_records: 0,
    };
    for index in 0..config.count {
        let case = &cases[index % cases.len()];
        let want_foreground = index % 2 == 0;
        let record = sample_record(config, case, index, want_foreground)?;
        if record.has_foreground {
            summary.foreground_records += 1;
        } else {
            summary.background_records += 1;
        }
        text.push_str(&serde_json::to_string(&record)?);
        text.push('\n');
    }
    fs::write(&config.out_path, text)
        .map_err(|source| SamplerError::io(&config.out_path, source))?;
    Ok(summary)
}

/// Groups patch records into fixed-size batches without requiring Python.
pub fn plan_batches(records: Vec<PatchRecord>, batch_size: usize) -> Result<BatchPlan> {
    if batch_size == 0 {
        return Err(SamplerError::invalid_input(
            "batch size must be greater than zero",
        ));
    }
    let batches = records
        .chunks(batch_size)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();
    Ok(BatchPlan {
        batch_size,
        batches,
    })
}

/// Extracts an aligned image/label patch from a loaded cached case.
pub fn extract_patch_pair(
    case: &LoadedCachedCase,
    start: [usize; 3],
    patch_size: [usize; 3],
) -> Result<CachedPatch> {
    validate_patch(patch_size)?;
    validate_patch_start(case.image.shape, start, patch_size)?;
    let voxels = patch_voxels(patch_size);
    let mut image = vec![0.0; voxels];
    let mut label = vec![0; voxels];
    let has_foreground = extract_patch_pair_into(case, start, patch_size, &mut image, &mut label)?;
    Ok(CachedPatch {
        image: Volume3D::new(patch_size, image).map_err(|error| {
            SamplerError::invalid_input(format!("invalid extracted image patch: {error}"))
        })?,
        label: Volume3D::new(patch_size, label).map_err(|error| {
            SamplerError::invalid_input(format!("invalid extracted label patch: {error}"))
        })?,
        has_foreground,
    })
}

/// Extracts an aligned image/label patch into reusable caller-owned buffers.
pub fn extract_patch_pair_into(
    case: &LoadedCachedCase,
    start: [usize; 3],
    patch_size: [usize; 3],
    image_out: &mut [f32],
    label_out: &mut [u16],
) -> Result<bool> {
    validate_patch(patch_size)?;
    validate_patch_start(case.image.shape, start, patch_size)?;
    let voxels = patch_voxels(patch_size);
    if image_out.len() != voxels {
        return Err(SamplerError::invalid_input(format!(
            "image output buffer has {} values, expected {voxels}",
            image_out.len()
        )));
    }
    if label_out.len() != voxels {
        return Err(SamplerError::invalid_input(format!(
            "label output buffer has {} values, expected {voxels}",
            label_out.len()
        )));
    }
    copy_patch_rows(&case.image, start, patch_size, image_out);
    copy_patch_rows(&case.label, start, patch_size, label_out);
    Ok(case.foreground_prefix.count(start, patch_size) != 0)
}

/// Counts foreground label voxels in a patch without materializing the patch.
pub fn foreground_voxels_in_patch(
    case: &LoadedCachedCase,
    start: [usize; 3],
    patch_size: [usize; 3],
) -> Result<usize> {
    validate_patch(patch_size)?;
    validate_patch_start(case.label.shape, start, patch_size)?;
    Ok(case.foreground_prefix.count(start, patch_size))
}

fn sample_record(
    config: &SampleConfig,
    case: &SamplingCase,
    index: usize,
    want_foreground: bool,
) -> Result<PatchRecord> {
    validate_patch_start(case.shape, [0, 0, 0], config.patch_size)?;
    let mut rng = SplitMix64::new(seed_for(
        config.seed,
        &case.metadata.case_id,
        config.epoch,
        config.worker,
        index as u64,
    ));
    let start = if want_foreground && !case.foreground_indices.is_empty() {
        let flat = case.foreground_indices[rng.next_usize(case.foreground_indices.len())];
        start_for_center(case.shape, flat_to_xyz(flat, case.shape), config.patch_size)
    } else {
        random_start(case.shape, config.patch_size, &mut rng)
    };
    let has_foreground = if want_foreground && !case.foreground_indices.is_empty() {
        true
    } else {
        case.foreground_prefix.count(start, config.patch_size) != 0
    };
    Ok(PatchRecord {
        index,
        case_id: case.metadata.case_id.clone(),
        patch_start: start,
        patch_size: config.patch_size,
        has_foreground,
        strategy: config.strategy,
        epoch: config.epoch,
        worker: config.worker,
    })
}

fn load_sampling_cases(cache_dir: impl AsRef<Path>) -> Result<Vec<SamplingCase>> {
    let manifest = read_cache_manifest(cache_dir)?;
    manifest.cases.iter().map(load_sampling_case).collect()
}

fn load_sampling_case(case: &CachedCase) -> Result<SamplingCase> {
    let (foreground_indices, foreground_prefix) = if has_foreground_artifacts(case) {
        read_foreground_artifacts(case)?
    } else {
        let label = read_u16_volume(Path::new(&case.label_cache_path), case.shape)?;
        foreground_from_label(&label)?
    };
    Ok(SamplingCase {
        metadata: case.clone(),
        shape: case.shape,
        foreground_indices,
        foreground_prefix,
    })
}

fn load_case(case: &CachedCase) -> Result<LoadedCachedCase> {
    let image = read_f32_volume(Path::new(&case.image_cache_path), case.shape)?;
    let label = read_u16_volume(Path::new(&case.label_cache_path), case.shape)?;
    let (foreground_indices, foreground_prefix) = if has_foreground_artifacts(case) {
        read_foreground_artifacts(case)?
    } else {
        foreground_from_label(&label)?
    };
    Ok(LoadedCachedCase {
        metadata: case.clone(),
        image,
        label,
        foreground_indices,
        foreground_prefix,
    })
}

impl ForegroundPrefix {
    /// Builds a foreground integral volume from a label map.
    pub fn from_label(label: &Volume3D<u16>) -> Result<Self> {
        let shape = label.shape;
        let prefix_shape = [shape[0] + 1, shape[1] + 1, shape[2] + 1];
        let mut values = vec![0_u32; prefix_shape[0] * prefix_shape[1] * prefix_shape[2]];
        let prefix_y_stride = prefix_shape[0];
        let prefix_z_stride = prefix_shape[0] * prefix_shape[1];
        let label_y_stride = shape[0];
        let label_z_stride = shape[0] * shape[1];
        for z in 1..prefix_shape[2] {
            let prefix_z_base = z * prefix_z_stride;
            let previous_prefix_z_base = (z - 1) * prefix_z_stride;
            let label_z_base = (z - 1) * label_z_stride;
            for y in 1..prefix_shape[1] {
                let mut row_sum = 0_u32;
                let row_base = prefix_z_base + y * prefix_y_stride;
                let above_base = prefix_z_base + (y - 1) * prefix_y_stride;
                let behind_base = previous_prefix_z_base + y * prefix_y_stride;
                let above_behind_base = previous_prefix_z_base + (y - 1) * prefix_y_stride;
                let label_row_base = label_z_base + (y - 1) * label_y_stride;
                for x in 1..prefix_shape[0] {
                    row_sum += u32::from(label.data[label_row_base + x - 1] != 0);
                    values[row_base + x] =
                        row_sum + values[above_base + x] + values[behind_base + x]
                            - values[above_behind_base + x];
                }
            }
        }
        Ok(Self {
            shape,
            prefix_shape,
            values,
        })
    }

    /// Builds a foreground integral volume from persisted raw values.
    pub fn from_values(shape: [usize; 3], values: Vec<u32>) -> Result<Self> {
        let prefix_shape = [shape[0] + 1, shape[1] + 1, shape[2] + 1];
        let expected = prefix_shape[0] * prefix_shape[1] * prefix_shape[2];
        if values.len() != expected {
            return Err(SamplerError::invalid_input(format!(
                "foreground prefix for shape {shape:?} has {} values, expected {expected}",
                values.len()
            )));
        }
        Ok(Self {
            shape,
            prefix_shape,
            values,
        })
    }

    /// Counts foreground voxels inside a half-open patch.
    pub fn count(&self, start: [usize; 3], size: [usize; 3]) -> usize {
        let end = [start[0] + size[0], start[1] + size[1], start[2] + size[2]];
        let [x0, y0, z0] = start;
        let [x1, y1, z1] = end;
        let value = i64::from(self.at(x1, y1, z1))
            - i64::from(self.at(x0, y1, z1))
            - i64::from(self.at(x1, y0, z1))
            - i64::from(self.at(x1, y1, z0))
            + i64::from(self.at(x0, y0, z1))
            + i64::from(self.at(x0, y1, z0))
            + i64::from(self.at(x1, y0, z0))
            - i64::from(self.at(x0, y0, z0));
        value as usize
    }

    fn at(&self, x: usize, y: usize, z: usize) -> u32 {
        self.values[prefix_index(self.prefix_shape, x, y, z)]
    }
}

fn read_f32_volume(path: &Path, shape: [usize; 3]) -> Result<Volume3D<f32>> {
    let bytes = fs::read(path).map_err(|source| SamplerError::io(path, source))?;
    if bytes.len() % 4 != 0 {
        return Err(SamplerError::invalid_input(format!(
            "f32 cache file {} has invalid length {}",
            path.display(),
            bytes.len()
        )));
    }
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk length")))
        .collect();
    Volume3D::new(shape, values)
        .map_err(|error| SamplerError::invalid_input(format!("invalid cached image: {error}")))
}

fn read_u16_volume(path: &Path, shape: [usize; 3]) -> Result<Volume3D<u16>> {
    let bytes = fs::read(path).map_err(|source| SamplerError::io(path, source))?;
    if bytes.len() % 2 != 0 {
        return Err(SamplerError::invalid_input(format!(
            "u16 cache file {} has invalid length {}",
            path.display(),
            bytes.len()
        )));
    }
    let values = bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes(chunk.try_into().expect("chunk length")))
        .collect();
    Volume3D::new(shape, values)
        .map_err(|error| SamplerError::invalid_input(format!("invalid cached label: {error}")))
}

fn has_foreground_artifacts(case: &CachedCase) -> bool {
    case.foreground_indices_path.is_some() && case.foreground_prefix_path.is_some()
}

fn foreground_from_label(label: &Volume3D<u16>) -> Result<(Vec<usize>, ForegroundPrefix)> {
    let foreground_indices = label
        .data
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value != 0).then_some(index))
        .collect();
    let foreground_prefix = ForegroundPrefix::from_label(label)?;
    Ok((foreground_indices, foreground_prefix))
}

fn read_foreground_artifacts(case: &CachedCase) -> Result<(Vec<usize>, ForegroundPrefix)> {
    if let Some(prefix_shape) = case.foreground_prefix_shape {
        let expected = [case.shape[0] + 1, case.shape[1] + 1, case.shape[2] + 1];
        if prefix_shape != expected {
            return Err(SamplerError::invalid_input(format!(
                "foreground prefix shape {prefix_shape:?} does not match cached shape {:?}",
                case.shape
            )));
        }
    }
    let indices_path = case
        .foreground_indices_path
        .as_deref()
        .ok_or_else(|| SamplerError::invalid_input("cache is missing foreground index path"))?;
    let prefix_path = case
        .foreground_prefix_path
        .as_deref()
        .ok_or_else(|| SamplerError::invalid_input("cache is missing foreground prefix path"))?;
    let foreground_indices = read_u64_indices(Path::new(indices_path))?;
    let values = read_u32_values(Path::new(prefix_path))?;
    let foreground_prefix = ForegroundPrefix::from_values(case.shape, values)?;
    Ok((foreground_indices, foreground_prefix))
}

fn read_u64_indices(path: &Path) -> Result<Vec<usize>> {
    let bytes = fs::read(path).map_err(|source| SamplerError::io(path, source))?;
    if bytes.len() % 8 != 0 {
        return Err(SamplerError::invalid_input(format!(
            "u64 index cache file {} has invalid length {}",
            path.display(),
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("chunk length")) as usize)
        .collect())
}

fn read_u32_values(path: &Path) -> Result<Vec<u32>> {
    let bytes = fs::read(path).map_err(|source| SamplerError::io(path, source))?;
    if bytes.len() % 4 != 0 {
        return Err(SamplerError::invalid_input(format!(
            "u32 cache file {} has invalid length {}",
            path.display(),
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("chunk length")))
        .collect())
}

fn copy_patch_rows<T: Copy>(
    volume: &Volume3D<T>,
    start: [usize; 3],
    patch_size: [usize; 3],
    out: &mut [T],
) {
    let row = patch_size[0];
    for local_z in 0..patch_size[2] {
        let source_z = start[2] + local_z;
        for local_y in 0..patch_size[1] {
            let source_y = start[1] + local_y;
            let source_start = volume.index(start[0], source_y, source_z);
            let destination_start = row * (local_y + patch_size[1] * local_z);
            out[destination_start..destination_start + row]
                .copy_from_slice(&volume.data[source_start..source_start + row]);
        }
    }
}

fn patch_voxels(size: [usize; 3]) -> usize {
    size[0] * size[1] * size[2]
}

fn prefix_index(shape: [usize; 3], x: usize, y: usize, z: usize) -> usize {
    x + shape[0] * (y + shape[1] * z)
}

fn validate_patch(size: [usize; 3]) -> Result<()> {
    if size.contains(&0) {
        return Err(SamplerError::invalid_input(format!(
            "patch size must be non-zero, got {size:?}"
        )));
    }
    Ok(())
}

fn validate_patch_start(shape: [usize; 3], start: [usize; 3], size: [usize; 3]) -> Result<()> {
    for axis in 0..3 {
        if size[axis] > shape[axis] {
            return Err(SamplerError::invalid_input(format!(
                "patch size {size:?} exceeds cached shape {shape:?}"
            )));
        }
        if start[axis] + size[axis] > shape[axis] {
            return Err(SamplerError::invalid_input(format!(
                "patch start {start:?} with size {size:?} exceeds shape {shape:?}"
            )));
        }
    }
    Ok(())
}

fn random_start(shape: [usize; 3], size: [usize; 3], rng: &mut SplitMix64) -> [usize; 3] {
    [
        rng.next_usize(shape[0] - size[0] + 1),
        rng.next_usize(shape[1] - size[1] + 1),
        rng.next_usize(shape[2] - size[2] + 1),
    ]
}

fn start_for_center(shape: [usize; 3], center: [usize; 3], size: [usize; 3]) -> [usize; 3] {
    let mut start = [0_usize; 3];
    for axis in 0..3 {
        let half = size[axis] / 2;
        start[axis] = center[axis]
            .saturating_sub(half)
            .min(shape[axis] - size[axis]);
    }
    start
}

fn flat_to_xyz(index: usize, shape: [usize; 3]) -> [usize; 3] {
    let plane = shape[0] * shape[1];
    let z = index / plane;
    let rem = index % plane;
    let y = rem / shape[0];
    let x = rem % shape[0];
    [x, y, z]
}
