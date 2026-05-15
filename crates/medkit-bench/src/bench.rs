use std::{collections::HashMap, fs, path::PathBuf, thread, time::Instant};

use medkit_sampler::{
    extract_patch_pair_into, foreground_voxels_in_patch, load_cached_cases, LoadedCachedCase,
    PatchRecord,
};
use serde::{Deserialize, Serialize};

use crate::{BenchError, Result};

/// Benchmark configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchConfig {
    /// Cache directory.
    pub cache_dir: PathBuf,
    /// Patch size in x, y, z order.
    pub patch_size: [usize; 3],
    /// Worker count used to shard deterministic extraction loops.
    pub workers: usize,
    /// Number of patch extractions per cold and warm pass.
    pub samples: usize,
}

/// Benchmark configuration for a sampled patch JSONL plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanBenchConfig {
    /// Cache directory.
    pub cache_dir: PathBuf,
    /// Sampled patch JSONL path.
    pub patches_path: PathBuf,
    /// Worker count used to shard deterministic extraction loops.
    pub workers: usize,
    /// Number of patch extractions per cold and warm pass.
    pub samples: usize,
}

/// Benchmark report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchReport {
    /// Cache directory.
    pub cache_dir: String,
    /// Patch size in x, y, z order.
    pub patch_size: [usize; 3],
    /// Worker count.
    pub workers: usize,
    /// Extracted samples per cold and warm pass.
    pub samples: usize,
    /// Cold pass includes cache manifest and volume reads.
    pub cold: BenchMetrics,
    /// Warm pass reuses loaded cached volumes.
    pub warm: BenchMetrics,
    /// Baseline status for future Python/MONAI comparison.
    pub python_monai_baseline: String,
}

/// Benchmark report for extraction driven by sampled JSONL patch records.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanBenchReport {
    /// Cache directory.
    pub cache_dir: String,
    /// Sampled patch JSONL path.
    pub patches_path: String,
    /// Patch size in x, y, z order.
    pub patch_size: [usize; 3],
    /// Worker count.
    pub workers: usize,
    /// Patch records read from the JSONL plan.
    pub records: usize,
    /// Extracted samples per cold and warm pass.
    pub samples: usize,
    /// Cold pass includes JSONL plan reads, cache manifest reads, and volume reads.
    pub cold: BenchMetrics,
    /// Warm pass reuses parsed patch records and loaded cached volumes.
    pub warm: BenchMetrics,
}

/// Throughput metrics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BenchMetrics {
    /// Extracted samples.
    pub samples: usize,
    /// Elapsed wall time in milliseconds.
    pub elapsed_ms: f64,
    /// Approximate CPU time in milliseconds for this single-process pass.
    pub cpu_time_ms: f64,
    /// Samples per second.
    pub samples_per_second: f64,
    /// Megabytes per second based on extracted image+label patch bytes.
    pub mb_per_second: f64,
}

/// Benchmarks cache loading and patch extraction.
pub fn bench_cache(config: &BenchConfig) -> Result<BenchReport> {
    if config.workers == 0 {
        return Err(BenchError::invalid_input(
            "workers must be greater than zero",
        ));
    }
    if config.patch_size.contains(&0) {
        return Err(BenchError::invalid_input(
            "patch dimensions must be non-zero",
        ));
    }
    if config.samples == 0 {
        return Err(BenchError::invalid_input(
            "samples must be greater than zero",
        ));
    }

    let cold_start = Instant::now();
    let cases = load_cached_cases(&config.cache_dir)?;
    let cold_metrics = run_pass(
        &cases,
        config.patch_size,
        config.samples,
        config.workers,
        cold_start.elapsed().as_secs_f64(),
    )?;

    let warm_start = Instant::now();
    let warm_metrics = run_pass(
        &cases,
        config.patch_size,
        config.samples,
        config.workers,
        warm_start.elapsed().as_secs_f64(),
    )?;

    Ok(BenchReport {
        cache_dir: config.cache_dir.to_string_lossy().into_owned(),
        patch_size: config.patch_size,
        workers: config.workers,
        samples: config.samples,
        cold: cold_metrics,
        warm: warm_metrics,
        python_monai_baseline: "planned: compare against a simple MONAI/PyTorch Dataset baseline"
            .to_string(),
    })
}

/// Benchmarks cache loading and patch extraction using sampled JSONL patch records.
pub fn bench_patch_plan(config: &PlanBenchConfig) -> Result<PlanBenchReport> {
    if config.workers == 0 {
        return Err(BenchError::invalid_input(
            "workers must be greater than zero",
        ));
    }
    if config.samples == 0 {
        return Err(BenchError::invalid_input(
            "samples must be greater than zero",
        ));
    }

    let cold_start = Instant::now();
    let records = read_patch_records(&config.patches_path)?;
    let cases = load_cached_cases(&config.cache_dir)?;
    let plan = resolve_patch_plan(&records, &cases)?;
    let cold_metrics = run_plan_pass(
        &cases,
        &plan,
        config.samples,
        config.workers,
        cold_start.elapsed().as_secs_f64(),
    )?;

    let warm_start = Instant::now();
    let warm_metrics = run_plan_pass(
        &cases,
        &plan,
        config.samples,
        config.workers,
        warm_start.elapsed().as_secs_f64(),
    )?;

    Ok(PlanBenchReport {
        cache_dir: config.cache_dir.to_string_lossy().into_owned(),
        patches_path: config.patches_path.to_string_lossy().into_owned(),
        patch_size: plan.patch_size,
        workers: config.workers,
        records: records.len(),
        samples: config.samples,
        cold: cold_metrics,
        warm: warm_metrics,
    })
}

fn run_pass(
    cases: &[LoadedCachedCase],
    patch_size: [usize; 3],
    samples: usize,
    workers: usize,
    already_elapsed_secs: f64,
) -> Result<BenchMetrics> {
    if cases.is_empty() {
        return Err(BenchError::invalid_input("cache contains no cases"));
    }
    let start = Instant::now();
    let worker_count = workers.min(samples).max(1);
    let checksum = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for worker in 0..worker_count {
            handles.push(
                scope.spawn(move || worker_pass(cases, patch_size, samples, worker_count, worker)),
            );
        }
        let mut checksum = 0_u64;
        for handle in handles {
            checksum = checksum.wrapping_add(
                handle
                    .join()
                    .map_err(|_| BenchError::invalid_input("benchmark worker panicked"))??,
            );
        }
        Ok::<u64, BenchError>(checksum)
    })?;
    std::hint::black_box(checksum);
    let elapsed = already_elapsed_secs + start.elapsed().as_secs_f64();
    let bytes = samples as f64 * patch_bytes(patch_size) as f64;
    Ok(BenchMetrics {
        samples,
        elapsed_ms: elapsed * 1000.0,
        cpu_time_ms: elapsed * 1000.0,
        samples_per_second: samples as f64 / elapsed.max(f64::EPSILON),
        mb_per_second: (bytes / (1024.0 * 1024.0)) / elapsed.max(f64::EPSILON),
    })
}

fn run_plan_pass(
    cases: &[LoadedCachedCase],
    plan: &ResolvedPatchPlan,
    samples: usize,
    workers: usize,
    already_elapsed_secs: f64,
) -> Result<BenchMetrics> {
    if plan.records.is_empty() {
        return Err(BenchError::invalid_input("patch plan contains no records"));
    }
    let start = Instant::now();
    let worker_count = workers.min(samples).max(1);
    let checksum = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for worker in 0..worker_count {
            handles.push(
                scope.spawn(move || worker_plan_pass(cases, plan, samples, worker_count, worker)),
            );
        }
        let mut checksum = 0_u64;
        for handle in handles {
            checksum = checksum.wrapping_add(
                handle
                    .join()
                    .map_err(|_| BenchError::invalid_input("benchmark worker panicked"))??,
            );
        }
        Ok::<u64, BenchError>(checksum)
    })?;
    std::hint::black_box(checksum);
    let elapsed = already_elapsed_secs + start.elapsed().as_secs_f64();
    let bytes = samples as f64 * patch_bytes(plan.patch_size) as f64;
    Ok(BenchMetrics {
        samples,
        elapsed_ms: elapsed * 1000.0,
        cpu_time_ms: elapsed * 1000.0,
        samples_per_second: samples as f64 / elapsed.max(f64::EPSILON),
        mb_per_second: (bytes / (1024.0 * 1024.0)) / elapsed.max(f64::EPSILON),
    })
}

fn worker_pass(
    cases: &[LoadedCachedCase],
    patch_size: [usize; 3],
    samples: usize,
    worker_count: usize,
    worker: usize,
) -> Result<u64> {
    let mut checksum = 0_u64;
    let mut image = vec![0.0_f32; patch_voxels(patch_size)];
    let mut label = vec![0_u16; image.len()];
    for index in (worker..samples).step_by(worker_count) {
        let case = &cases[index % cases.len()];
        let start = centered_start(case.image.shape, patch_size)?;
        let has_foreground =
            extract_patch_pair_into(case, start, patch_size, &mut image, &mut label)?;
        let foreground = foreground_voxels_in_patch(case, start, patch_size)?;
        checksum = checksum
            .wrapping_add(foreground as u64)
            .wrapping_add(u64::from(has_foreground))
            .wrapping_add(label.first().copied().unwrap_or_default() as u64);
    }
    std::hint::black_box(&image);
    std::hint::black_box(&label);
    Ok(checksum)
}

fn worker_plan_pass(
    cases: &[LoadedCachedCase],
    plan: &ResolvedPatchPlan,
    samples: usize,
    worker_count: usize,
    worker: usize,
) -> Result<u64> {
    let mut checksum = 0_u64;
    let mut image = vec![0.0_f32; patch_voxels(plan.patch_size)];
    let mut label = vec![0_u16; image.len()];
    for index in (worker..samples).step_by(worker_count) {
        let record_index = index % plan.records.len();
        let record = &plan.records[record_index];
        let case = &cases[plan.case_indices[record_index]];
        let has_foreground = extract_patch_pair_into(
            case,
            record.patch_start,
            record.patch_size,
            &mut image,
            &mut label,
        )?;
        let foreground = foreground_voxels_in_patch(case, record.patch_start, record.patch_size)?;
        checksum = checksum
            .wrapping_add(foreground as u64)
            .wrapping_add(u64::from(has_foreground))
            .wrapping_add(label.first().copied().unwrap_or_default() as u64);
    }
    std::hint::black_box(&image);
    std::hint::black_box(&label);
    Ok(checksum)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedPatchPlan {
    records: Vec<PatchRecord>,
    case_indices: Vec<usize>,
    patch_size: [usize; 3],
}

fn read_patch_records(path: &PathBuf) -> Result<Vec<PatchRecord>> {
    let text = fs::read_to_string(path).map_err(|source| {
        BenchError::invalid_input(format!("failed to read {}: {source}", path.display()))
    })?;
    let mut records = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record = serde_json::from_str(line).map_err(|source| {
            BenchError::invalid_input(format!(
                "failed to parse {} line {}: {source}",
                path.display(),
                line_number + 1
            ))
        })?;
        records.push(record);
    }
    if records.is_empty() {
        return Err(BenchError::invalid_input(format!(
            "patch plan {} contains no records",
            path.display()
        )));
    }
    Ok(records)
}

fn resolve_patch_plan(
    records: &[PatchRecord],
    cases: &[LoadedCachedCase],
) -> Result<ResolvedPatchPlan> {
    if cases.is_empty() {
        return Err(BenchError::invalid_input("cache contains no cases"));
    }
    let patch_size = records[0].patch_size;
    if patch_size.contains(&0) {
        return Err(BenchError::invalid_input(format!(
            "patch size must be non-zero, got {patch_size:?}"
        )));
    }
    let case_indices = cases
        .iter()
        .enumerate()
        .map(|(index, case)| (case.metadata.case_id.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut resolved_indices = Vec::with_capacity(records.len());
    for record in records {
        if record.patch_size != patch_size {
            return Err(BenchError::invalid_input(format!(
                "mixed patch sizes are not supported: expected {patch_size:?}, got {:?}",
                record.patch_size
            )));
        }
        let case_index = case_indices
            .get(record.case_id.as_str())
            .copied()
            .ok_or_else(|| {
                BenchError::invalid_input(format!(
                    "patch record references missing cached case {}",
                    record.case_id
                ))
            })?;
        resolved_indices.push(case_index);
    }
    Ok(ResolvedPatchPlan {
        records: records.to_vec(),
        case_indices: resolved_indices,
        patch_size,
    })
}

fn centered_start(shape: [usize; 3], patch_size: [usize; 3]) -> Result<[usize; 3]> {
    let mut start = [0_usize; 3];
    for axis in 0..3 {
        if patch_size[axis] > shape[axis] {
            return Err(BenchError::invalid_input(format!(
                "patch size {patch_size:?} exceeds cached shape {shape:?}"
            )));
        }
        start[axis] = (shape[axis] - patch_size[axis]) / 2;
    }
    Ok(start)
}

fn patch_bytes(patch_size: [usize; 3]) -> usize {
    patch_voxels(patch_size) * (4 + 2)
}

fn patch_voxels(patch_size: [usize; 3]) -> usize {
    patch_size[0] * patch_size[1] * patch_size[2]
}
