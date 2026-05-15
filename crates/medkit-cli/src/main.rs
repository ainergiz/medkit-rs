#![forbid(unsafe_code)]

use std::{
    env,
    ffi::OsString,
    path::{Path, PathBuf},
    process,
};

use medkit_bench::{bench_cache, bench_patch_plan, BenchConfig, BenchReport, PlanBenchConfig};
use medkit_cache::{prepare_cache, CacheManifest, PrepareConfig};
use medkit_dataset::{
    validate_dataset, write_manifest_json, write_report, DatasetManifest, ValidationConfig,
};
use medkit_sampler::{sample_cache, SampleConfig, SampleSummary, SamplingStrategy};

fn main() {
    process::exit(match run(env::args_os()) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("{error}");
            2
        }
    });
}

fn run(args: impl IntoIterator<Item = OsString>) -> Result<(), CliError> {
    let command = parse_args(args)?;
    match command {
        Command::DatasetValidate {
            root,
            images_dir,
            labels_dir,
            manifest_path,
            report_path,
        } => {
            let config = ValidationConfig::new(&root)
                .images_dir(images_dir)
                .labels_dir(labels_dir);
            let manifest = validate_dataset(&config)?;
            write_manifest_json(&manifest, &manifest_path)?;
            write_report(&manifest, &report_path)?;
            print_summary(&manifest, &manifest_path, &report_path);
            Ok(())
        }
        Command::Prepare {
            root,
            manifest_path,
            plan_path,
            cache_dir,
            chunk_shape,
        } => {
            let manifest = prepare_cache(&PrepareConfig {
                dataset_root: root,
                manifest_path,
                plan_path,
                cache_dir,
                chunk_shape,
            })?;
            print_prepare_summary(&manifest);
            Ok(())
        }
        Command::Sample {
            cache_dir,
            patch_size,
            strategy,
            count,
            out_path,
        } => {
            let summary = sample_cache(&SampleConfig {
                cache_dir,
                patch_size,
                strategy,
                count,
                out_path: out_path.clone(),
                seed: 0,
                epoch: 0,
                worker: 0,
            })?;
            print_sample_summary(&summary, &out_path);
            Ok(())
        }
        Command::Bench {
            cache_dir,
            patch_size,
            workers,
            samples,
        } => {
            let report = bench_cache(&BenchConfig {
                cache_dir,
                patch_size,
                workers,
                samples,
            })?;
            print_bench_report(&report);
            Ok(())
        }
        Command::BenchPlan {
            cache_dir,
            patches_path,
            workers,
            samples,
        } => {
            let report = bench_patch_plan(&PlanBenchConfig {
                cache_dir,
                patches_path,
                workers,
                samples,
            })?;
            print_plan_bench_report(&report);
            Ok(())
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Command {
    DatasetValidate {
        root: PathBuf,
        images_dir: PathBuf,
        labels_dir: PathBuf,
        manifest_path: PathBuf,
        report_path: PathBuf,
    },
    Prepare {
        root: PathBuf,
        manifest_path: PathBuf,
        plan_path: PathBuf,
        cache_dir: PathBuf,
        chunk_shape: Option<[usize; 3]>,
    },
    Sample {
        cache_dir: PathBuf,
        patch_size: [usize; 3],
        strategy: SamplingStrategy,
        count: usize,
        out_path: PathBuf,
    },
    Bench {
        cache_dir: PathBuf,
        patch_size: [usize; 3],
        workers: usize,
        samples: usize,
    },
    BenchPlan {
        cache_dir: PathBuf,
        patches_path: PathBuf,
        workers: usize,
        samples: usize,
    },
}

fn parse_args(args: impl IntoIterator<Item = OsString>) -> Result<Command, CliError> {
    let mut args = args.into_iter();
    let _program = args.next();
    let Some(first) = args.next() else {
        return Err(CliError::usage());
    };
    if first == "dataset" {
        return parse_dataset_command(args);
    }
    if first == "prepare" {
        return parse_prepare_command(args);
    }
    if first == "sample" {
        return parse_sample_command(args);
    }
    if first == "bench" {
        return parse_bench_command(args);
    }
    if first == "bench-plan" {
        return parse_bench_plan_command(args);
    }
    Err(CliError::usage())
}

fn parse_dataset_command(mut args: impl Iterator<Item = OsString>) -> Result<Command, CliError> {
    let Some(action) = args.next() else {
        return Err(CliError::usage());
    };
    if action != "validate" {
        return Err(CliError::usage());
    }
    let Some(root) = args.next() else {
        return Err(CliError::usage());
    };
    let root = PathBuf::from(root);
    let mut images_dir = PathBuf::from("imagesTr");
    let mut labels_dir = PathBuf::from("labelsTr");
    let mut manifest_path = None;
    let mut report_path = None;

    let mut rest = args.peekable();
    while let Some(flag) = rest.next() {
        match flag.to_string_lossy().as_ref() {
            "--images" => images_dir = next_path(&mut rest, "--images")?,
            "--labels" => labels_dir = next_path(&mut rest, "--labels")?,
            "--out" => manifest_path = Some(next_path(&mut rest, "--out")?),
            "--report" => report_path = Some(next_path(&mut rest, "--report")?),
            "--help" | "-h" => return Err(CliError::usage()),
            other => {
                return Err(CliError::Message(format!(
                    "unknown argument: {other}\n\n{}",
                    usage()
                )))
            }
        }
    }

    let manifest_path = manifest_path.unwrap_or_else(|| root.join("manifest.json"));
    let report_path = report_path.unwrap_or_else(|| root.join("report.txt"));

    Ok(Command::DatasetValidate {
        root,
        images_dir,
        labels_dir,
        manifest_path,
        report_path,
    })
}

fn parse_prepare_command(mut args: impl Iterator<Item = OsString>) -> Result<Command, CliError> {
    let Some(root) = args.next() else {
        return Err(CliError::usage());
    };
    let root = PathBuf::from(root);
    let mut manifest_path = None;
    let mut plan_path = None;
    let mut cache_dir = None;
    let mut chunk_shape = None;
    let mut rest = args.peekable();
    while let Some(flag) = rest.next() {
        match flag.to_string_lossy().as_ref() {
            "--manifest" => manifest_path = Some(next_path(&mut rest, "--manifest")?),
            "--plan" => plan_path = Some(next_path(&mut rest, "--plan")?),
            "--cache" => cache_dir = Some(next_path(&mut rest, "--cache")?),
            "--chunk" => chunk_shape = Some(parse_patch(&next_string(&mut rest, "--chunk")?)?),
            "--help" | "-h" => return Err(CliError::usage()),
            other => {
                return Err(CliError::Message(format!(
                    "unknown argument: {other}\n\n{}",
                    usage()
                )))
            }
        }
    }
    Ok(Command::Prepare {
        root,
        manifest_path: manifest_path
            .ok_or_else(|| CliError::Message(format!("missing --manifest\n\n{}", usage())))?,
        plan_path: plan_path
            .ok_or_else(|| CliError::Message(format!("missing --plan\n\n{}", usage())))?,
        cache_dir: cache_dir
            .ok_or_else(|| CliError::Message(format!("missing --cache\n\n{}", usage())))?,
        chunk_shape,
    })
}

fn parse_sample_command(mut args: impl Iterator<Item = OsString>) -> Result<Command, CliError> {
    let Some(cache_dir) = args.next() else {
        return Err(CliError::usage());
    };
    let cache_dir = PathBuf::from(cache_dir);
    let mut patch_size = None;
    let mut strategy = None;
    let mut count = None;
    let mut out_path = None;
    let mut rest = args.peekable();
    while let Some(flag) = rest.next() {
        match flag.to_string_lossy().as_ref() {
            "--patch" => patch_size = Some(parse_patch(&next_string(&mut rest, "--patch")?)?),
            "--strategy" => {
                strategy = Some(parse_strategy(&next_string(&mut rest, "--strategy")?)?)
            }
            "--count" => count = Some(parse_usize(&next_string(&mut rest, "--count")?, "--count")?),
            "--out" => out_path = Some(next_path(&mut rest, "--out")?),
            "--help" | "-h" => return Err(CliError::usage()),
            other => {
                return Err(CliError::Message(format!(
                    "unknown argument: {other}\n\n{}",
                    usage()
                )))
            }
        }
    }
    Ok(Command::Sample {
        cache_dir,
        patch_size: patch_size
            .ok_or_else(|| CliError::Message(format!("missing --patch\n\n{}", usage())))?,
        strategy: strategy.unwrap_or(SamplingStrategy::ForegroundBalanced),
        count: count.ok_or_else(|| CliError::Message(format!("missing --count\n\n{}", usage())))?,
        out_path: out_path
            .ok_or_else(|| CliError::Message(format!("missing --out\n\n{}", usage())))?,
    })
}

fn parse_bench_command(mut args: impl Iterator<Item = OsString>) -> Result<Command, CliError> {
    let Some(cache_dir) = args.next() else {
        return Err(CliError::usage());
    };
    let cache_dir = PathBuf::from(cache_dir);
    let mut patch_size = None;
    let mut workers = None;
    let mut samples = None;
    let mut rest = args.peekable();
    while let Some(flag) = rest.next() {
        match flag.to_string_lossy().as_ref() {
            "--patch" => patch_size = Some(parse_patch(&next_string(&mut rest, "--patch")?)?),
            "--workers" => {
                workers = Some(parse_usize(
                    &next_string(&mut rest, "--workers")?,
                    "--workers",
                )?)
            }
            "--samples" => {
                samples = Some(parse_usize(
                    &next_string(&mut rest, "--samples")?,
                    "--samples",
                )?)
            }
            "--help" | "-h" => return Err(CliError::usage()),
            other => {
                return Err(CliError::Message(format!(
                    "unknown argument: {other}\n\n{}",
                    usage()
                )))
            }
        }
    }
    Ok(Command::Bench {
        cache_dir,
        patch_size: patch_size
            .ok_or_else(|| CliError::Message(format!("missing --patch\n\n{}", usage())))?,
        workers: workers.unwrap_or(1),
        samples: samples.unwrap_or_else(|| workers.unwrap_or(1) * 64),
    })
}

fn parse_bench_plan_command(mut args: impl Iterator<Item = OsString>) -> Result<Command, CliError> {
    let Some(cache_dir) = args.next() else {
        return Err(CliError::usage());
    };
    let cache_dir = PathBuf::from(cache_dir);
    let mut patches_path = None;
    let mut workers = None;
    let mut samples = None;
    let mut rest = args.peekable();
    while let Some(flag) = rest.next() {
        match flag.to_string_lossy().as_ref() {
            "--patches" => patches_path = Some(next_path(&mut rest, "--patches")?),
            "--workers" => {
                workers = Some(parse_usize(
                    &next_string(&mut rest, "--workers")?,
                    "--workers",
                )?)
            }
            "--samples" => {
                samples = Some(parse_usize(
                    &next_string(&mut rest, "--samples")?,
                    "--samples",
                )?)
            }
            "--help" | "-h" => return Err(CliError::usage()),
            other => {
                return Err(CliError::Message(format!(
                    "unknown argument: {other}\n\n{}",
                    usage()
                )))
            }
        }
    }
    Ok(Command::BenchPlan {
        cache_dir,
        patches_path: patches_path
            .ok_or_else(|| CliError::Message(format!("missing --patches\n\n{}", usage())))?,
        workers: workers.unwrap_or(1),
        samples: samples.unwrap_or_else(|| workers.unwrap_or(1) * 64),
    })
}

fn next_path(
    args: &mut std::iter::Peekable<impl Iterator<Item = OsString>>,
    flag: &str,
) -> Result<PathBuf, CliError> {
    args.next()
        .map(PathBuf::from)
        .ok_or_else(|| CliError::Message(format!("missing value for {flag}\n\n{}", usage())))
}

fn next_string(
    args: &mut std::iter::Peekable<impl Iterator<Item = OsString>>,
    flag: &str,
) -> Result<String, CliError> {
    args.next()
        .map(|value| value.to_string_lossy().into_owned())
        .ok_or_else(|| CliError::Message(format!("missing value for {flag}\n\n{}", usage())))
}

fn parse_patch(value: &str) -> Result<[usize; 3], CliError> {
    let parts = value.split(',').collect::<Vec<_>>();
    if parts.len() != 3 {
        return Err(CliError::Message(format!(
            "patch must be formatted as x,y,z, got {value}"
        )));
    }
    Ok([
        parse_usize(parts[0], "--patch")?,
        parse_usize(parts[1], "--patch")?,
        parse_usize(parts[2], "--patch")?,
    ])
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, CliError> {
    value
        .parse::<usize>()
        .map_err(|_| CliError::Message(format!("invalid integer for {flag}: {value}")))
}

fn parse_strategy(value: &str) -> Result<SamplingStrategy, CliError> {
    match value {
        "foreground-balanced" | "foreground_balanced" => Ok(SamplingStrategy::ForegroundBalanced),
        other => Err(CliError::Message(format!(
            "unsupported sampling strategy: {other}"
        ))),
    }
}

fn print_summary(manifest: &DatasetManifest, manifest_path: &Path, report_path: &Path) {
    println!("Dataset: {}", manifest.dataset_root);
    println!("Cases: {}", manifest.summary.total_cases);
    println!("Valid: {}", manifest.summary.valid_cases);
    println!("Invalid: {}", manifest.summary.invalid_cases);
    println!("Wrote manifest: {}", manifest_path.display());
    println!("Wrote report: {}", report_path.display());
}

fn print_prepare_summary(manifest: &CacheManifest) {
    println!("Cache: {}", manifest.cache_dir);
    println!("Input cases: {}", manifest.summary.input_cases);
    println!("Cached cases: {}", manifest.summary.cached_cases);
    println!("Failed cases: {}", manifest.summary.failed_cases);
    println!("Foreground voxels: {}", manifest.summary.foreground_voxels);
    println!("Transform plan hash: {}", manifest.transform_plan_hash);
}

fn print_sample_summary(summary: &SampleSummary, out_path: &Path) {
    println!("Samples: {}", summary.records);
    println!("Foreground records: {}", summary.foreground_records);
    println!("Background records: {}", summary.background_records);
    println!("Wrote samples: {}", out_path.display());
}

fn print_bench_report(report: &BenchReport) {
    println!("Cache: {}", report.cache_dir);
    println!("Patch: {:?}", report.patch_size);
    println!("Workers: {}", report.workers);
    println!("Samples: {}", report.samples);
    println!(
        "Cold: {:.2} samples/s, {:.2} MB/s, {:.2} ms",
        report.cold.samples_per_second, report.cold.mb_per_second, report.cold.elapsed_ms
    );
    println!(
        "Warm: {:.2} samples/s, {:.2} MB/s, {:.2} ms",
        report.warm.samples_per_second, report.warm.mb_per_second, report.warm.elapsed_ms
    );
    println!("Python/MONAI baseline: {}", report.python_monai_baseline);
}

fn print_plan_bench_report(report: &medkit_bench::PlanBenchReport) {
    println!("Cache: {}", report.cache_dir);
    println!("Patches: {}", report.patches_path);
    println!("Patch: {:?}", report.patch_size);
    println!("Workers: {}", report.workers);
    println!("Records: {}", report.records);
    println!("Samples: {}", report.samples);
    println!(
        "Plan cold: {:.2} samples/s, {:.2} MB/s, {:.2} ms",
        report.cold.samples_per_second, report.cold.mb_per_second, report.cold.elapsed_ms
    );
    println!(
        "Plan warm: {:.2} samples/s, {:.2} MB/s, {:.2} ms",
        report.warm.samples_per_second, report.warm.mb_per_second, report.warm.elapsed_ms
    );
}

#[derive(Debug)]
enum CliError {
    Message(String),
    Dataset(medkit_dataset::DatasetError),
    Cache(medkit_cache::CacheError),
    Sampler(medkit_sampler::SamplerError),
    Bench(medkit_bench::BenchError),
}

impl CliError {
    fn usage() -> Self {
        Self::Message(usage())
    }
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Message(message) => write!(f, "{message}"),
            Self::Dataset(error) => write!(f, "{error}"),
            Self::Cache(error) => write!(f, "{error}"),
            Self::Sampler(error) => write!(f, "{error}"),
            Self::Bench(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for CliError {}

impl From<medkit_dataset::DatasetError> for CliError {
    fn from(value: medkit_dataset::DatasetError) -> Self {
        Self::Dataset(value)
    }
}

impl From<medkit_cache::CacheError> for CliError {
    fn from(value: medkit_cache::CacheError) -> Self {
        Self::Cache(value)
    }
}

impl From<medkit_sampler::SamplerError> for CliError {
    fn from(value: medkit_sampler::SamplerError) -> Self {
        Self::Sampler(value)
    }
}

impl From<medkit_bench::BenchError> for CliError {
    fn from(value: medkit_bench::BenchError) -> Self {
        Self::Bench(value)
    }
}

fn usage() -> String {
    "Usage:\n  medkit dataset validate <root> [--images imagesTr] [--labels labelsTr] [--out manifest.json] [--report report.txt]\n  medkit prepare <root> --manifest manifest.json --plan ct-segmentation.toml --cache .medkit/cache [--chunk 96,96,96]\n  medkit sample <cache> --patch 96,96,96 --strategy foreground-balanced --count 10000 --out patches.jsonl\n  medkit bench <cache> --patch 96,96,96 --workers 8 [--samples 10000]\n  medkit bench-plan <cache> --patches patches.jsonl --workers 8 [--samples 10000]".to_string()
}
