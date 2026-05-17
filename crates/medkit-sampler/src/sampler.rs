use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use medkit_cache::{read_cache_manifest, CachedCase};
use medkit_transform::Volume3D;
use memmap2::Mmap;
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

/// Memory-mapped resident cache case.
#[derive(Debug)]
pub struct MmapCachedCase {
    /// Cached case metadata.
    pub metadata: CachedCase,
    /// Cached volume shape in x, y, z order.
    pub shape: [usize; 3],
    image_mmap: Mmap,
    label_mmap: Mmap,
}

/// Memory-mapped fixed-size chunked cache case.
#[derive(Debug)]
pub struct ChunkedCachedCase {
    /// Cached case metadata.
    pub metadata: CachedCase,
    /// Cached volume shape in x, y, z order.
    pub shape: [usize; 3],
    /// Fixed chunk shape in x, y, z order.
    pub chunk_shape: [usize; 3],
    /// Chunk grid in x, y, z order.
    pub chunk_grid: [usize; 3],
    image_mmap: Mmap,
    label_mmap: Mmap,
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

/// Loads resident cache files as memory maps without materializing full volumes.
pub fn load_mmap_cached_cases(cache_dir: impl AsRef<Path>) -> Result<Vec<MmapCachedCase>> {
    let manifest = read_cache_manifest(cache_dir)?;
    manifest.cases.iter().map(load_mmap_case).collect()
}

/// Loads fixed-size chunked cache files as memory maps.
pub fn load_chunked_cached_cases(cache_dir: impl AsRef<Path>) -> Result<Vec<ChunkedCachedCase>> {
    let manifest = read_cache_manifest(cache_dir)?;
    manifest.cases.iter().map(load_chunked_case).collect()
}

/// Samples patch records from a cache and writes JSONL.
pub fn sample_cache(config: &SampleConfig) -> Result<SampleSummary> {
    validate_patch(config.patch_size)?;
    let cases = load_sampling_cases(&config.cache_dir)?;
    if cases.is_empty() {
        return Err(SamplerError::invalid_input("cache contains no cases"));
    }
    if let Some(parent) = config
        .out_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|source| SamplerError::io(parent, source))?;
    }
    let file = File::create(&config.out_path)
        .map_err(|source| SamplerError::io(&config.out_path, source))?;
    let mut writer = BufWriter::new(file);
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
        serde_json::to_writer(&mut writer, &record)?;
        writer
            .write_all(b"\n")
            .map_err(|source| SamplerError::io(&config.out_path, source))?;
    }
    writer
        .flush()
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
        image: Volume3D {
            shape: patch_size,
            data: image,
        },
        label: Volume3D {
            shape: patch_size,
            data: label,
        },
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

/// Extracts an aligned image/label patch from memory-mapped resident cache files.
pub fn extract_patch_pair_mmap_into(
    case: &MmapCachedCase,
    start: [usize; 3],
    patch_size: [usize; 3],
    image_out: &mut [f32],
    label_out: &mut [u16],
) -> Result<bool> {
    validate_patch_outputs(
        case.shape,
        start,
        patch_size,
        image_out.len(),
        label_out.len(),
    )?;
    for local_z in 0..patch_size[2] {
        let source_z = start[2] + local_z;
        for local_y in 0..patch_size[1] {
            let source_y = start[1] + local_y;
            let source_start = flat_index(case.shape, start[0], source_y, source_z);
            let destination_start = patch_size[0] * (local_y + patch_size[1] * local_z);
            read_f32_values_from_mmap(
                &case.image_mmap,
                source_start,
                &mut image_out[destination_start..destination_start + patch_size[0]],
            )?;
            read_u16_values_from_mmap(
                &case.label_mmap,
                source_start,
                &mut label_out[destination_start..destination_start + patch_size[0]],
            )?;
        }
    }
    Ok(label_out.iter().any(|value| *value != 0))
}

/// Extracts an aligned image/label patch from memory-mapped chunked cache files.
pub fn extract_patch_pair_chunked_into(
    case: &ChunkedCachedCase,
    start: [usize; 3],
    patch_size: [usize; 3],
    image_out: &mut [f32],
    label_out: &mut [u16],
) -> Result<bool> {
    validate_patch_outputs(
        case.shape,
        start,
        patch_size,
        image_out.len(),
        label_out.len(),
    )?;
    image_out.fill(0.0);
    label_out.fill(0);
    let end = [
        start[0] + patch_size[0],
        start[1] + patch_size[1],
        start[2] + patch_size[2],
    ];
    let chunk_min = [
        start[0] / case.chunk_shape[0],
        start[1] / case.chunk_shape[1],
        start[2] / case.chunk_shape[2],
    ];
    let chunk_max = [
        (end[0] - 1) / case.chunk_shape[0],
        (end[1] - 1) / case.chunk_shape[1],
        (end[2] - 1) / case.chunk_shape[2],
    ];
    let chunk_voxels = patch_voxels(case.chunk_shape);
    for chunk_z in chunk_min[2]..=chunk_max[2] {
        for chunk_y in chunk_min[1]..=chunk_max[1] {
            for chunk_x in chunk_min[0]..=chunk_max[0] {
                let chunk_index =
                    chunk_x + case.chunk_grid[0] * (chunk_y + case.chunk_grid[1] * chunk_z);
                copy_chunk_overlap_from_mmap(
                    case,
                    [chunk_x, chunk_y, chunk_z],
                    chunk_index * chunk_voxels,
                    start,
                    patch_size,
                    image_out,
                    label_out,
                )?;
            }
        }
    }
    Ok(label_out.iter().any(|value| *value != 0))
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

fn load_mmap_case(case: &CachedCase) -> Result<MmapCachedCase> {
    let image_path = Path::new(&case.image_cache_path);
    let label_path = Path::new(&case.label_cache_path);
    let image_mmap = mmap_file(image_path)?;
    let label_mmap = mmap_file(label_path)?;
    validate_mmap_len(image_path, &image_mmap, value_count(case.shape)?, 4)?;
    validate_mmap_len(label_path, &label_mmap, value_count(case.shape)?, 2)?;
    Ok(MmapCachedCase {
        metadata: case.clone(),
        shape: case.shape,
        image_mmap,
        label_mmap,
    })
}

fn load_chunked_case(case: &CachedCase) -> Result<ChunkedCachedCase> {
    let image_path = case.image_chunk_cache_path.as_deref().ok_or_else(|| {
        SamplerError::invalid_input(format!("missing image chunk cache for {}", case.case_id))
    })?;
    let label_path = case.label_chunk_cache_path.as_deref().ok_or_else(|| {
        SamplerError::invalid_input(format!("missing label chunk cache for {}", case.case_id))
    })?;
    let chunk_grid = case.chunk_grid.ok_or_else(|| {
        SamplerError::invalid_input(format!("missing chunk grid for {}", case.case_id))
    })?;
    let image_path = Path::new(image_path);
    let label_path = Path::new(label_path);
    let image_mmap = mmap_file(image_path)?;
    let label_mmap = mmap_file(label_path)?;
    let chunk_values = value_count(chunk_grid)?
        .checked_mul(value_count(case.chunk_shape)?)
        .ok_or_else(|| SamplerError::invalid_input("chunked value count overflow"))?;
    validate_mmap_len(image_path, &image_mmap, chunk_values, 4)?;
    validate_mmap_len(label_path, &label_mmap, chunk_values, 2)?;
    Ok(ChunkedCachedCase {
        metadata: case.clone(),
        shape: case.shape,
        chunk_shape: case.chunk_shape,
        chunk_grid,
        image_mmap,
        label_mmap,
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

fn copy_chunk_overlap_from_mmap(
    case: &ChunkedCachedCase,
    chunk_index: [usize; 3],
    chunk_value_offset: usize,
    patch_start: [usize; 3],
    patch_size: [usize; 3],
    image_out: &mut [f32],
    label_out: &mut [u16],
) -> Result<()> {
    let chunk_start = [
        chunk_index[0] * case.chunk_shape[0],
        chunk_index[1] * case.chunk_shape[1],
        chunk_index[2] * case.chunk_shape[2],
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
        patch_end[0].min((chunk_start[0] + case.chunk_shape[0]).min(case.shape[0])),
        patch_end[1].min((chunk_start[1] + case.chunk_shape[1]).min(case.shape[1])),
        patch_end[2].min((chunk_start[2] + case.chunk_shape[2]).min(case.shape[2])),
    ];
    if overlap_start[0] >= overlap_end[0]
        || overlap_start[1] >= overlap_end[1]
        || overlap_start[2] >= overlap_end[2]
    {
        return Ok(());
    }
    let row = overlap_end[0] - overlap_start[0];
    for z in overlap_start[2]..overlap_end[2] {
        for y in overlap_start[1]..overlap_end[1] {
            let chunk_source = (overlap_start[0] - chunk_start[0])
                + case.chunk_shape[0]
                    * ((y - chunk_start[1]) + case.chunk_shape[1] * (z - chunk_start[2]));
            let patch_dest = (overlap_start[0] - patch_start[0])
                + patch_size[0] * ((y - patch_start[1]) + patch_size[1] * (z - patch_start[2]));
            read_f32_values_from_mmap(
                &case.image_mmap,
                chunk_value_offset + chunk_source,
                &mut image_out[patch_dest..patch_dest + row],
            )?;
            read_u16_values_from_mmap(
                &case.label_mmap,
                chunk_value_offset + chunk_source,
                &mut label_out[patch_dest..patch_dest + row],
            )?;
        }
    }
    Ok(())
}

fn read_f32_values_from_mmap(mmap: &Mmap, value_offset: usize, out: &mut [f32]) -> Result<()> {
    let bytes = mmap_bytes(mmap, value_offset, out.len(), 4)?;
    for (slot, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *slot = f32::from_le_bytes(chunk.try_into().expect("chunk length"));
    }
    Ok(())
}

fn read_u16_values_from_mmap(mmap: &Mmap, value_offset: usize, out: &mut [u16]) -> Result<()> {
    let bytes = mmap_bytes(mmap, value_offset, out.len(), 2)?;
    for (slot, chunk) in out.iter_mut().zip(bytes.chunks_exact(2)) {
        *slot = u16::from_le_bytes(chunk.try_into().expect("chunk length"));
    }
    Ok(())
}

fn mmap_bytes(
    mmap: &Mmap,
    value_offset: usize,
    values: usize,
    bytes_per_value: usize,
) -> Result<&[u8]> {
    let byte_offset = value_offset
        .checked_mul(bytes_per_value)
        .ok_or_else(|| SamplerError::invalid_input("mmap byte offset overflow"))?;
    let byte_len = values
        .checked_mul(bytes_per_value)
        .ok_or_else(|| SamplerError::invalid_input("mmap byte length overflow"))?;
    let byte_end = byte_offset
        .checked_add(byte_len)
        .ok_or_else(|| SamplerError::invalid_input("mmap byte range overflow"))?;
    mmap.get(byte_offset..byte_end)
        .ok_or_else(|| SamplerError::invalid_input("mmap range is out of bounds"))
}

fn mmap_file(path: &Path) -> Result<Mmap> {
    let file = File::open(path).map_err(|source| SamplerError::io(path, source))?;
    // SAFETY: the mapping is read-only and the file handle is kept alive until
    // the OS mapping is created. Callers validate length before reading.
    unsafe { Mmap::map(&file).map_err(|source| SamplerError::io(path, source)) }
}

fn validate_mmap_len(
    path: &Path,
    mmap: &Mmap,
    values: usize,
    bytes_per_value: usize,
) -> Result<()> {
    let expected = values
        .checked_mul(bytes_per_value)
        .ok_or_else(|| SamplerError::invalid_input("mmap byte length overflow"))?;
    if mmap.len() != expected {
        return Err(SamplerError::invalid_input(format!(
            "{} has {} bytes, expected {expected}",
            path.display(),
            mmap.len()
        )));
    }
    Ok(())
}

fn validate_patch_outputs(
    shape: [usize; 3],
    start: [usize; 3],
    patch_size: [usize; 3],
    image_len: usize,
    label_len: usize,
) -> Result<()> {
    validate_patch(patch_size)?;
    validate_patch_start(shape, start, patch_size)?;
    let voxels = patch_voxels(patch_size);
    if image_len != voxels {
        return Err(SamplerError::invalid_input(format!(
            "image output buffer has {image_len} values, expected {voxels}"
        )));
    }
    if label_len != voxels {
        return Err(SamplerError::invalid_input(format!(
            "label output buffer has {label_len} values, expected {voxels}"
        )));
    }
    Ok(())
}

fn value_count(shape: [usize; 3]) -> Result<usize> {
    shape[0]
        .checked_mul(shape[1])
        .and_then(|value| value.checked_mul(shape[2]))
        .ok_or_else(|| SamplerError::invalid_input("volume value count overflow"))
}

fn flat_index(shape: [usize; 3], x: usize, y: usize, z: usize) -> usize {
    x + shape[0] * (y + shape[1] * z)
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
