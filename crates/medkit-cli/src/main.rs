#![forbid(unsafe_code)]

use std::{
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::{self, Command as ProcessCommand},
    time::Instant,
};

use medkit_bench::{bench_cache, bench_patch_plan, BenchConfig, BenchReport, PlanBenchConfig};
use medkit_cache::{prepare_cache, CacheManifest, PrepareConfig};
use medkit_cxr::{
    cache_cxr, index_cxr, ingest_cxr_dicom, split_cxr, validate_cache_cxr, validate_cxr,
    CacheConfig as CxrCacheConfig, CacheSummary as CxrCacheSummary,
    CacheValidationSummary as CxrCacheValidationSummary, IndexConfig as CxrIndexConfig,
    IndexSummary as CxrIndexSummary, IngestConfig as CxrIngestConfig,
    IngestSummary as CxrIngestSummary, SplitConfig as CxrSplitConfig,
    SplitSummary as CxrSplitSummary, ValidateCacheConfig as CxrValidateCacheConfig,
    ValidateConfig as CxrValidateConfig, ValidationSummary as CxrValidationSummary,
};
use medkit_dataset::{
    validate_dataset, write_manifest_json, write_report, DatasetManifest, ValidationConfig,
};
use medkit_dicom::{
    browse_dicom, explain_pixels, inspect_dicom_file, render_unicode, scan_dicom,
    scan_dicom_with_workers, write_scan_outputs, DicomBrowseConfig, DicomFileConfig,
    DicomGraphSummary, DicomScanConfig, DicomScanSummary, DicomViewConfig, RenderOptions,
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
        Command::CxrIndex {
            images_root,
            dicom_index_path,
            metadata_path,
            labels_path,
            reports_root,
            out_path,
        } => {
            let summary = index_cxr(&CxrIndexConfig {
                images_root,
                dicom_index_path,
                metadata_path,
                labels_path,
                reports_root,
                out_path: out_path.clone(),
            })?;
            print_cxr_index_summary(&summary);
            Ok(())
        }
        Command::CxrValidate {
            manifest_path,
            require_frontal,
            check_patient_leakage,
            check_duplicates,
            report_path,
        } => {
            let summary = validate_cxr(&CxrValidateConfig {
                manifest_path,
                require_frontal,
                check_patient_leakage,
                check_duplicates,
                report_path,
            })?;
            print_cxr_validation_summary(&summary);
            Ok(())
        }
        Command::CxrSplit {
            manifest_path,
            by,
            train,
            val,
            test,
            stratify,
            out_path,
            seed,
        } => {
            let summary = split_cxr(&CxrSplitConfig {
                manifest_path,
                by,
                train,
                val,
                test,
                stratify,
                out_path,
                seed,
            })?;
            print_cxr_split_summary(&summary);
            Ok(())
        }
        Command::CxrCache {
            manifest_path,
            splits_path,
            plan_path,
            cache_dir,
        } => {
            let summary = cache_cxr(&CxrCacheConfig {
                manifest_path,
                splits_path,
                plan_path,
                cache_dir,
            })?;
            print_cxr_cache_summary(&summary);
            Ok(())
        }
        Command::CxrValidateCache {
            cache_dir,
            split,
            expected_targets,
            expected_image_shape,
            plan_path,
            report_path,
            json_path,
        } => {
            let summary = validate_cache_cxr(&CxrValidateCacheConfig {
                cache_dir,
                split,
                expected_targets,
                expected_image_shape,
                plan_path,
                report_path,
                json_path,
            })?;
            print_cxr_cache_validation_summary(&summary);
            if summary.status != "ok" {
                return Err(CliError::Message(format!(
                    "CXR cache validation failed with {} errors",
                    summary.errors.len()
                )));
            }
            Ok(())
        }
        Command::CxrIngest {
            raw_root,
            recipe_path,
            labels_path,
            cache_dir,
            workdir,
            report_path,
            dry_run,
            workers,
        } => {
            let summary = ingest_cxr_dicom(&CxrIngestConfig {
                raw_root,
                recipe_path,
                labels_path,
                cache_dir,
                workdir,
                report_path,
                dry_run,
                workers,
            })?;
            print_cxr_ingest_summary(&summary);
            ensure_cxr_ingest_cache_validation_ok(&summary.cache_validation_status)
        }
        Command::CxrBenchmark(config) => run_cxr_benchmark_bridge(*config),
        Command::DicomScan {
            root,
            out_path,
            report_path,
            workers,
        } => {
            let config = DicomScanConfig {
                root,
                out_path: out_path.clone(),
                report_path: report_path.clone(),
            };
            let (summary, records) = if workers > 1 {
                scan_dicom_with_workers(&config, workers)?
            } else {
                scan_dicom(&config)?
            };
            write_scan_outputs(&summary, &records, &out_path, &report_path)?;
            print_dicom_scan_summary(&summary);
            Ok(())
        }
        Command::DicomBrowse {
            root,
            group,
            out_path,
            report_path,
            workers,
        } => {
            let summary = browse_dicom(&DicomBrowseConfig {
                root,
                group,
                out_path,
                report_path,
                workers,
            })?;
            print_dicom_graph_summary(&summary);
            Ok(())
        }
        Command::DicomInspect { path } => {
            let config = DicomFileConfig { path };
            let report = inspect_dicom_file(&config.path)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::DicomPixelsExplain { path } => {
            let config = DicomFileConfig { path };
            let explanation = explain_pixels(&config.path)?;
            println!("{}", serde_json::to_string_pretty(&explanation)?);
            Ok(())
        }
        Command::DicomView { path, width } => {
            let config = DicomViewConfig { path, width };
            let rendered = render_unicode(
                &config.path,
                &RenderOptions {
                    width: config.width,
                    include_metadata: true,
                },
            )?;
            println!("{rendered}");
            Ok(())
        }
    }
}

#[derive(Debug, Clone)]
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
    CxrIndex {
        images_root: PathBuf,
        dicom_index_path: Option<PathBuf>,
        metadata_path: Option<PathBuf>,
        labels_path: Option<PathBuf>,
        reports_root: Option<PathBuf>,
        out_path: PathBuf,
    },
    CxrValidate {
        manifest_path: PathBuf,
        require_frontal: bool,
        check_patient_leakage: bool,
        check_duplicates: bool,
        report_path: PathBuf,
    },
    CxrSplit {
        manifest_path: PathBuf,
        by: String,
        train: f64,
        val: f64,
        test: f64,
        stratify: Vec<String>,
        out_path: PathBuf,
        seed: u64,
    },
    CxrCache {
        manifest_path: PathBuf,
        splits_path: PathBuf,
        plan_path: PathBuf,
        cache_dir: PathBuf,
    },
    CxrValidateCache {
        cache_dir: PathBuf,
        split: Option<String>,
        expected_targets: Option<Vec<String>>,
        expected_image_shape: Option<[usize; 4]>,
        plan_path: Option<PathBuf>,
        report_path: Option<PathBuf>,
        json_path: Option<PathBuf>,
    },
    CxrIngest {
        raw_root: PathBuf,
        recipe_path: PathBuf,
        labels_path: PathBuf,
        cache_dir: PathBuf,
        workdir: PathBuf,
        report_path: PathBuf,
        dry_run: bool,
        workers: usize,
    },
    CxrBenchmark(Box<CxrBenchmarkBridgeConfig>),
    DicomScan {
        root: PathBuf,
        out_path: PathBuf,
        report_path: PathBuf,
        workers: usize,
    },
    DicomBrowse {
        root: PathBuf,
        group: Vec<String>,
        out_path: PathBuf,
        report_path: PathBuf,
        workers: usize,
    },
    DicomInspect {
        path: PathBuf,
    },
    DicomPixelsExplain {
        path: PathBuf,
    },
    DicomView {
        path: PathBuf,
        width: usize,
    },
}

#[derive(Debug, Clone)]
struct CxrBenchmarkBridgeConfig {
    manifest_path: Option<PathBuf>,
    splits_path: Option<PathBuf>,
    plan_path: Option<PathBuf>,
    targets: Option<String>,
    uncertain: Option<String>,
    baselines: Option<String>,
    batch_sizes: Option<String>,
    workers: Option<String>,
    device: Option<String>,
    out_path: Option<PathBuf>,
    python: String,
    script_path: PathBuf,
    work_dir: Option<PathBuf>,
    report_dir: Option<PathBuf>,
    run_id: Option<String>,
    max_samples: Option<usize>,
    max_train: Option<usize>,
    max_val: Option<usize>,
    max_test: Option<usize>,
    image_size: Option<usize>,
    epochs: Option<usize>,
    loader_batches: Option<usize>,
    warmup_batches: Option<usize>,
    smoke: bool,
    force_cache: bool,
    force_rematerialize: bool,
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
    if first == "cxr" {
        return parse_cxr_command(args);
    }
    if first == "dicom" {
        return parse_dicom_command(args);
    }
    Err(CliError::usage())
}

fn parse_dicom_command(mut args: impl Iterator<Item = OsString>) -> Result<Command, CliError> {
    let Some(action) = args.next() else {
        return Err(CliError::usage());
    };
    match action.to_string_lossy().as_ref() {
        "scan" => parse_dicom_scan_command(args),
        "browse" => parse_dicom_browse_command(args),
        "inspect" => parse_dicom_inspect_command(args),
        "pixels" => parse_dicom_pixels_command(args),
        "view" => parse_dicom_view_command(args),
        "--help" | "-h" => Err(CliError::usage()),
        other => Err(CliError::Message(format!(
            "unknown dicom command: {other}\n\n{}",
            usage()
        ))),
    }
}

fn parse_dicom_scan_command(mut args: impl Iterator<Item = OsString>) -> Result<Command, CliError> {
    let Some(root) = args.next() else {
        return Err(CliError::usage());
    };
    let root = PathBuf::from(root);
    let mut out_path = None;
    let mut report_path = None;
    let mut workers = 1usize;
    let mut rest = args.peekable();
    while let Some(flag) = rest.next() {
        match flag.to_string_lossy().as_ref() {
            "--out" => out_path = Some(next_path(&mut rest, "--out")?),
            "--report" => report_path = Some(next_path(&mut rest, "--report")?),
            "--workers" => {
                workers = parse_usize(&next_string(&mut rest, "--workers")?, "--workers")?
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
    Ok(Command::DicomScan {
        root,
        out_path: out_path
            .ok_or_else(|| CliError::Message(format!("missing --out\n\n{}", usage())))?,
        report_path: report_path
            .ok_or_else(|| CliError::Message(format!("missing --report\n\n{}", usage())))?,
        workers,
    })
}

fn parse_dicom_browse_command(
    mut args: impl Iterator<Item = OsString>,
) -> Result<Command, CliError> {
    let Some(root) = args.next() else {
        return Err(CliError::usage());
    };
    let root = PathBuf::from(root);
    let mut group = vec![
        "patient".to_string(),
        "study".to_string(),
        "series".to_string(),
    ];
    let mut out_path = None;
    let mut report_path = None;
    let mut workers = 1usize;
    let mut rest = args.peekable();
    while let Some(flag) = rest.next() {
        match flag.to_string_lossy().as_ref() {
            "--group" => group = parse_csv_list(&next_string(&mut rest, "--group")?),
            "--out" => out_path = Some(next_path(&mut rest, "--out")?),
            "--report" => report_path = Some(next_path(&mut rest, "--report")?),
            "--workers" => {
                workers = parse_usize(&next_string(&mut rest, "--workers")?, "--workers")?
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
    Ok(Command::DicomBrowse {
        root,
        group,
        out_path: out_path
            .ok_or_else(|| CliError::Message(format!("missing --out\n\n{}", usage())))?,
        report_path: report_path
            .ok_or_else(|| CliError::Message(format!("missing --report\n\n{}", usage())))?,
        workers,
    })
}

fn parse_dicom_inspect_command(
    mut args: impl Iterator<Item = OsString>,
) -> Result<Command, CliError> {
    let Some(path) = args.next() else {
        return Err(CliError::usage());
    };
    reject_trailing_args(args, "dicom inspect")?;
    Ok(Command::DicomInspect {
        path: PathBuf::from(path),
    })
}

fn parse_dicom_pixels_command(
    mut args: impl Iterator<Item = OsString>,
) -> Result<Command, CliError> {
    let Some(action) = args.next() else {
        return Err(CliError::usage());
    };
    if action != "--explain" {
        return Err(CliError::Message(format!(
            "unknown dicom pixels command: {}\n\n{}",
            action.to_string_lossy(),
            usage()
        )));
    }
    let Some(path) = args.next() else {
        return Err(CliError::usage());
    };
    reject_trailing_args(args, "dicom pixels --explain")?;
    Ok(Command::DicomPixelsExplain {
        path: PathBuf::from(path),
    })
}

fn parse_dicom_view_command(mut args: impl Iterator<Item = OsString>) -> Result<Command, CliError> {
    let Some(path) = args.next() else {
        return Err(CliError::usage());
    };
    let path = PathBuf::from(path);
    let mut width = 80usize;
    let mut rest = args.peekable();
    while let Some(flag) = rest.next() {
        match flag.to_string_lossy().as_ref() {
            "--width" => width = parse_usize(&next_string(&mut rest, "--width")?, "--width")?,
            "--help" | "-h" => return Err(CliError::usage()),
            other => {
                return Err(CliError::Message(format!(
                    "unknown argument: {other}\n\n{}",
                    usage()
                )))
            }
        }
    }
    Ok(Command::DicomView { path, width })
}

fn parse_cxr_command(mut args: impl Iterator<Item = OsString>) -> Result<Command, CliError> {
    let Some(action) = args.next() else {
        return Err(CliError::usage());
    };
    match action.to_string_lossy().as_ref() {
        "index" | "manifest" => parse_cxr_index_command(args),
        "validate" => parse_cxr_validate_command(args),
        "split" => parse_cxr_split_command(args),
        "cache" => parse_cxr_cache_command(args),
        "validate-cache" | "inspect-cache" => parse_cxr_validate_cache_command(args),
        "ingest" => parse_cxr_ingest_command(args),
        "benchmark" => parse_cxr_benchmark_command(args),
        "--help" | "-h" => Err(CliError::usage()),
        other => Err(CliError::Message(format!(
            "unknown cxr command: {other}\n\n{}",
            usage()
        ))),
    }
}

fn parse_cxr_benchmark_command(args: impl Iterator<Item = OsString>) -> Result<Command, CliError> {
    let mut manifest_path = None;
    let mut splits_path = None;
    let mut plan_path = None;
    let mut targets = None;
    let mut uncertain = None;
    let mut baselines = None;
    let mut batch_sizes = None;
    let mut workers = None;
    let mut device = None;
    let mut out_path = None;
    let mut python = "uv".to_string();
    let mut script_path =
        PathBuf::from("crates/medkit-benchmarks/scripts/cxr_classification_benchmark.py");
    let mut work_dir = None;
    let mut report_dir = None;
    let mut run_id = None;
    let mut max_samples = None;
    let mut max_train = None;
    let mut max_val = None;
    let mut max_test = None;
    let mut image_size = None;
    let mut epochs = None;
    let mut loader_batches = None;
    let mut warmup_batches = None;
    let mut smoke = false;
    let mut force_cache = false;
    let mut force_rematerialize = false;
    let mut rest = args.peekable();
    while let Some(flag) = rest.next() {
        match flag.to_string_lossy().as_ref() {
            "--manifest" => manifest_path = Some(next_path(&mut rest, "--manifest")?),
            "--splits" => splits_path = Some(next_path(&mut rest, "--splits")?),
            "--plan" => plan_path = Some(next_path(&mut rest, "--plan")?),
            "--targets" => targets = Some(next_string(&mut rest, "--targets")?),
            "--uncertain" => uncertain = Some(next_string(&mut rest, "--uncertain")?),
            "--baselines" => baselines = Some(next_string(&mut rest, "--baselines")?),
            "--batch-sizes" => batch_sizes = Some(next_string(&mut rest, "--batch-sizes")?),
            "--workers" => workers = Some(next_string(&mut rest, "--workers")?),
            "--device" => device = Some(next_string(&mut rest, "--device")?),
            "--out" => out_path = Some(next_path(&mut rest, "--out")?),
            "--python" => python = next_string(&mut rest, "--python")?,
            "--script" => script_path = next_path(&mut rest, "--script")?,
            "--work-dir" => work_dir = Some(next_path(&mut rest, "--work-dir")?),
            "--report-dir" => report_dir = Some(next_path(&mut rest, "--report-dir")?),
            "--run-id" => run_id = Some(next_string(&mut rest, "--run-id")?),
            "--max-samples" => {
                max_samples = Some(parse_usize(
                    &next_string(&mut rest, "--max-samples")?,
                    "--max-samples",
                )?)
            }
            "--max-train" => {
                max_train = Some(parse_usize(
                    &next_string(&mut rest, "--max-train")?,
                    "--max-train",
                )?)
            }
            "--max-val" => {
                max_val = Some(parse_usize(
                    &next_string(&mut rest, "--max-val")?,
                    "--max-val",
                )?)
            }
            "--max-test" => {
                max_test = Some(parse_usize(
                    &next_string(&mut rest, "--max-test")?,
                    "--max-test",
                )?)
            }
            "--image-size" => {
                image_size = Some(parse_usize(
                    &next_string(&mut rest, "--image-size")?,
                    "--image-size",
                )?)
            }
            "--epochs" => {
                epochs = Some(parse_usize(
                    &next_string(&mut rest, "--epochs")?,
                    "--epochs",
                )?)
            }
            "--loader-batches" => {
                loader_batches = Some(parse_usize(
                    &next_string(&mut rest, "--loader-batches")?,
                    "--loader-batches",
                )?)
            }
            "--warmup-batches" => {
                warmup_batches = Some(parse_usize(
                    &next_string(&mut rest, "--warmup-batches")?,
                    "--warmup-batches",
                )?)
            }
            "--smoke" => smoke = true,
            "--force-cache" => force_cache = true,
            "--force-rematerialize" => force_rematerialize = true,
            "--help" | "-h" => return Err(CliError::usage()),
            other => {
                return Err(CliError::Message(format!(
                    "unknown argument: {other}\n\n{}",
                    usage()
                )))
            }
        }
    }
    Ok(Command::CxrBenchmark(Box::new(CxrBenchmarkBridgeConfig {
        manifest_path,
        splits_path,
        plan_path,
        targets,
        uncertain,
        baselines,
        batch_sizes,
        workers,
        device,
        out_path,
        python,
        script_path,
        work_dir,
        report_dir,
        run_id,
        max_samples,
        max_train,
        max_val,
        max_test,
        image_size,
        epochs,
        loader_batches,
        warmup_batches,
        smoke,
        force_cache,
        force_rematerialize,
    })))
}

fn parse_cxr_index_command(args: impl Iterator<Item = OsString>) -> Result<Command, CliError> {
    let mut images_root = None;
    let mut dicom_index_path = None;
    let mut metadata_path = None;
    let mut labels_path = None;
    let mut reports_root = None;
    let mut out_path = None;
    let mut rest = args.peekable();
    while let Some(flag) = rest.next() {
        match flag.to_string_lossy().as_ref() {
            "--images" => images_root = Some(next_path(&mut rest, "--images")?),
            "--dicom-index" => dicom_index_path = Some(next_path(&mut rest, "--dicom-index")?),
            "--metadata" => metadata_path = Some(next_path(&mut rest, "--metadata")?),
            "--labels" => labels_path = Some(next_path(&mut rest, "--labels")?),
            "--reports" => reports_root = Some(next_path(&mut rest, "--reports")?),
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
    if images_root.is_some() && dicom_index_path.is_some() {
        return Err(CliError::Message(format!(
            "use either --images or --dicom-index, not both\n\n{}",
            usage()
        )));
    }
    let images_root = if let Some(root) = &images_root {
        root.clone()
    } else if let Some(path) = &dicom_index_path {
        path.parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    } else {
        return Err(CliError::Message(format!(
            "missing --images or --dicom-index\n\n{}",
            usage()
        )));
    };
    Ok(Command::CxrIndex {
        images_root,
        dicom_index_path,
        metadata_path,
        labels_path,
        reports_root,
        out_path: out_path
            .ok_or_else(|| CliError::Message(format!("missing --out\n\n{}", usage())))?,
    })
}

fn parse_cxr_validate_command(
    mut args: impl Iterator<Item = OsString>,
) -> Result<Command, CliError> {
    let Some(manifest_path) = args.next() else {
        return Err(CliError::usage());
    };
    let manifest_path = PathBuf::from(manifest_path);
    let mut require_frontal = false;
    let mut check_patient_leakage = false;
    let mut check_duplicates = false;
    let mut report_path = None;
    let mut rest = args.peekable();
    while let Some(flag) = rest.next() {
        match flag.to_string_lossy().as_ref() {
            "--require-frontal" => require_frontal = true,
            "--check-patient-leakage" => check_patient_leakage = true,
            "--check-duplicates" => check_duplicates = true,
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
    Ok(Command::CxrValidate {
        manifest_path,
        require_frontal,
        check_patient_leakage,
        check_duplicates,
        report_path: report_path.unwrap_or_else(|| PathBuf::from("validation.md")),
    })
}

fn parse_cxr_split_command(mut args: impl Iterator<Item = OsString>) -> Result<Command, CliError> {
    let Some(manifest_path) = args.next() else {
        return Err(CliError::usage());
    };
    let manifest_path = PathBuf::from(manifest_path);
    let mut by = "patient_id".to_string();
    let mut train = 0.8;
    let mut val = 0.1;
    let mut test = 0.1;
    let mut stratify = Vec::new();
    let mut out_path = None;
    let mut seed = 0u64;
    let mut rest = args.peekable();
    while let Some(flag) = rest.next() {
        match flag.to_string_lossy().as_ref() {
            "--by" => by = next_string(&mut rest, "--by")?,
            "--train" => train = parse_f64(&next_string(&mut rest, "--train")?, "--train")?,
            "--val" => val = parse_f64(&next_string(&mut rest, "--val")?, "--val")?,
            "--test" => test = parse_f64(&next_string(&mut rest, "--test")?, "--test")?,
            "--stratify" => stratify = parse_csv_list(&next_string(&mut rest, "--stratify")?),
            "--seed" => seed = parse_u64(&next_string(&mut rest, "--seed")?, "--seed")?,
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
    Ok(Command::CxrSplit {
        manifest_path,
        by,
        train,
        val,
        test,
        stratify,
        out_path: out_path
            .ok_or_else(|| CliError::Message(format!("missing --out\n\n{}", usage())))?,
        seed,
    })
}

fn parse_cxr_cache_command(mut args: impl Iterator<Item = OsString>) -> Result<Command, CliError> {
    let Some(manifest_path) = args.next() else {
        return Err(CliError::usage());
    };
    let manifest_path = PathBuf::from(manifest_path);
    let mut splits_path = None;
    let mut plan_path = None;
    let mut cache_dir = None;
    let mut rest = args.peekable();
    while let Some(flag) = rest.next() {
        match flag.to_string_lossy().as_ref() {
            "--splits" => splits_path = Some(next_path(&mut rest, "--splits")?),
            "--plan" => plan_path = Some(next_path(&mut rest, "--plan")?),
            "--cache" => cache_dir = Some(next_path(&mut rest, "--cache")?),
            "--help" | "-h" => return Err(CliError::usage()),
            other => {
                return Err(CliError::Message(format!(
                    "unknown argument: {other}\n\n{}",
                    usage()
                )))
            }
        }
    }
    Ok(Command::CxrCache {
        manifest_path,
        splits_path: splits_path
            .ok_or_else(|| CliError::Message(format!("missing --splits\n\n{}", usage())))?,
        plan_path: plan_path
            .ok_or_else(|| CliError::Message(format!("missing --plan\n\n{}", usage())))?,
        cache_dir: cache_dir
            .ok_or_else(|| CliError::Message(format!("missing --cache\n\n{}", usage())))?,
    })
}

fn parse_cxr_validate_cache_command(
    mut args: impl Iterator<Item = OsString>,
) -> Result<Command, CliError> {
    let Some(cache_dir) = args.next() else {
        return Err(CliError::usage());
    };
    let cache_dir = PathBuf::from(cache_dir);
    let mut split = None;
    let mut expected_targets = None;
    let mut expected_image_shape = None;
    let mut plan_path = None;
    let mut report_path = None;
    let mut json_path = None;
    let mut rest = args.peekable();
    while let Some(flag) = rest.next() {
        match flag.to_string_lossy().as_ref() {
            "--split" => split = Some(next_string(&mut rest, "--split")?),
            "--targets" => {
                expected_targets = Some(parse_csv_list(&next_string(&mut rest, "--targets")?))
            }
            "--image-shape" => {
                expected_image_shape = Some(parse_shape4(
                    &next_string(&mut rest, "--image-shape")?,
                    "--image-shape",
                )?)
            }
            "--plan" => plan_path = Some(next_path(&mut rest, "--plan")?),
            "--report" => report_path = Some(next_path(&mut rest, "--report")?),
            "--json" => json_path = Some(next_path(&mut rest, "--json")?),
            "--help" | "-h" => return Err(CliError::usage()),
            other => {
                return Err(CliError::Message(format!(
                    "unknown argument: {other}\n\n{}",
                    usage()
                )))
            }
        }
    }
    Ok(Command::CxrValidateCache {
        cache_dir,
        split,
        expected_targets,
        expected_image_shape,
        plan_path,
        report_path,
        json_path,
    })
}

fn parse_cxr_ingest_command(mut args: impl Iterator<Item = OsString>) -> Result<Command, CliError> {
    let Some(raw_root) = args.next() else {
        return Err(CliError::usage());
    };
    let raw_root = PathBuf::from(raw_root);
    let mut recipe_path = None;
    let mut labels_path = None;
    let mut cache_dir = None;
    let mut workdir = None;
    let mut report_path = None;
    let mut dry_run = false;
    let mut workers = 1usize;
    let mut rest = args.peekable();
    while let Some(flag) = rest.next() {
        match flag.to_string_lossy().as_ref() {
            "--recipe" => recipe_path = Some(next_path(&mut rest, "--recipe")?),
            "--labels" => labels_path = Some(next_path(&mut rest, "--labels")?),
            "--cache" => cache_dir = Some(next_path(&mut rest, "--cache")?),
            "--workdir" => workdir = Some(next_path(&mut rest, "--workdir")?),
            "--report" => report_path = Some(next_path(&mut rest, "--report")?),
            "--workers" => {
                workers = parse_usize(&next_string(&mut rest, "--workers")?, "--workers")?
            }
            "--dry-run" => dry_run = true,
            "--help" | "-h" => return Err(CliError::usage()),
            other => {
                return Err(CliError::Message(format!(
                    "unknown argument: {other}\n\n{}",
                    usage()
                )))
            }
        }
    }
    Ok(Command::CxrIngest {
        raw_root,
        recipe_path: recipe_path
            .ok_or_else(|| CliError::Message(format!("missing --recipe\n\n{}", usage())))?,
        labels_path: labels_path
            .ok_or_else(|| CliError::Message(format!("missing --labels\n\n{}", usage())))?,
        cache_dir: cache_dir
            .ok_or_else(|| CliError::Message(format!("missing --cache\n\n{}", usage())))?,
        workdir: workdir
            .ok_or_else(|| CliError::Message(format!("missing --workdir\n\n{}", usage())))?,
        report_path: report_path
            .ok_or_else(|| CliError::Message(format!("missing --report\n\n{}", usage())))?,
        dry_run,
        workers,
    })
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

fn reject_trailing_args(
    args: impl Iterator<Item = OsString>,
    command: &str,
) -> Result<(), CliError> {
    if let Some(extra) = args.into_iter().next() {
        return Err(CliError::Message(format!(
            "unexpected argument for {command}: {}\n\n{}",
            extra.to_string_lossy(),
            usage()
        )));
    }
    Ok(())
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

fn parse_u64(value: &str, flag: &str) -> Result<u64, CliError> {
    value
        .parse::<u64>()
        .map_err(|_| CliError::Message(format!("invalid integer for {flag}: {value}")))
}

fn parse_f64(value: &str, flag: &str) -> Result<f64, CliError> {
    value
        .parse::<f64>()
        .map_err(|_| CliError::Message(format!("invalid float for {flag}: {value}")))
}

fn parse_shape4(value: &str, flag: &str) -> Result<[usize; 4], CliError> {
    let parts = value.split(',').collect::<Vec<_>>();
    if parts.len() != 4 {
        return Err(CliError::Message(format!(
            "{flag} must be formatted as n,c,h,w, got {value}"
        )));
    }
    Ok([
        parse_usize(parts[0], flag)?,
        parse_usize(parts[1], flag)?,
        parse_usize(parts[2], flag)?,
        parse_usize(parts[3], flag)?,
    ])
}

fn parse_csv_list(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToOwned::to_owned)
        .collect()
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

fn print_cxr_index_summary(summary: &CxrIndexSummary) {
    println!("CXR records: {}", summary.records);
    println!("Patients: {}", summary.patients);
    println!("Studies: {}", summary.studies);
    println!("Wrote manifest: {}", summary.out_path);
}

fn print_cxr_validation_summary(summary: &CxrValidationSummary) {
    println!("Records: {}", summary.records);
    println!("Readable images: {}", summary.readable_images);
    println!("Unreadable images: {}", summary.unreadable_images);
    println!("Patient overlap: {}", summary.patient_overlap_count);
    println!(
        "Duplicate image hash overlap: {}",
        summary.duplicate_hash_overlap_count
    );
    println!("Wrote report: {}", summary.report_path);
}

fn print_cxr_split_summary(summary: &CxrSplitSummary) {
    println!("Split by: {}", summary.by);
    for (split, count) in &summary.counts {
        println!("{split}: {count}");
    }
    println!("Patient overlap: {}", summary.patient_overlap_count);
    println!("Wrote splits: {}", summary.out_path);
}

fn print_cxr_cache_summary(summary: &CxrCacheSummary) {
    println!("Cache: {}", summary.cache_dir);
    println!("Cache schema version: {}", summary.cache_schema_version);
    println!("Image size: {}", summary.image_size);
    println!("Targets: {}", summary.targets.join(","));
    println!(
        "Label policy: uncertain={}, missing={}",
        summary.label_policy.uncertain, summary.label_policy.missing
    );
    println!("Transform fingerprint: {}", summary.transform_fingerprint);
    println!(
        "Source manifest checksum: {}",
        summary.source_manifest_checksum
    );
    println!(
        "Normalization: mean {:.6}, std {:.6}",
        summary.normalization.mean, summary.normalization.std
    );
    for (split, split_summary) in &summary.splits {
        println!("{split}: {} samples", split_summary.samples);
    }
    println!("Failed samples: {}", summary.failed_samples.len());
    println!("Cache size bytes: {}", summary.cache_size_bytes);
}

fn print_cxr_cache_validation_summary(summary: &CxrCacheValidationSummary) {
    println!("Cache: {}", summary.cache_dir);
    println!("Status: {}", summary.status);
    println!(
        "Cache schema version: {} (expected {})",
        summary.cache_schema_version, summary.expected_cache_schema_version
    );
    println!("Checked splits: {}", summary.checked_splits.join(","));
    println!("Targets: {}", summary.targets.join(","));
    println!("Errors: {}", summary.errors.len());
    for error in &summary.errors {
        println!("  - {error}");
    }
    println!("Warnings: {}", summary.warnings.len());
    for warning in &summary.warnings {
        println!("  - {warning}");
    }
}

fn print_cxr_ingest_summary(summary: &CxrIngestSummary) {
    println!("CXR ingest status: {}", summary.status);
    println!("Dry run: {}", summary.dry_run);
    println!("Recipe: {}", summary.recipe_name);
    println!("Recipe hash: {}", summary.recipe_hash);
    println!(
        "DICOM records scanned: {}",
        summary.counts.dicom_records_scanned
    );
    println!("Manifest records: {}", summary.counts.manifest_records);
    println!(
        "Unsupported/skipped images: {}",
        summary.counts.unsupported_or_skipped_images
    );
    println!(
        "Cache transform fingerprint: {}",
        summary.cache_transform_fingerprint
    );
    println!("Cache validation: {}", summary.cache_validation_status);
    println!("Wrote report: {}", summary.paths.ingest_report);
    println!("Wrote summary: {}", summary.paths.ingest_summary_json);
}

fn print_dicom_scan_summary(summary: &DicomScanSummary) {
    println!("DICOM root: {}", summary.root);
    println!("Records: {}", summary.records);
    println!("Errors: {}", summary.errors.len());
    println!("Warnings: {}", summary.warnings);
    println!(
        "Duplicate SOP Instance UIDs: {}",
        summary.duplicate_sop_instance_uids
    );
    println!("Duplicate pixel hashes: {}", summary.duplicate_pixel_hashes);
    println!("Wrote inventory: {}", summary.out_path);
    println!("Wrote report: {}", summary.report_path);
}

fn print_dicom_graph_summary(summary: &DicomGraphSummary) {
    println!("DICOM graph root: {}", summary.root);
    println!("Patients: {}", summary.patients);
    println!("Studies: {}", summary.studies);
    println!("Series: {}", summary.series);
    println!("Instances: {}", summary.instances);
    println!(
        "Duplicate SOP Instance UIDs: {}",
        summary.duplicate_sop_instance_uids
    );
    println!("Duplicate pixel hashes: {}", summary.duplicate_pixel_hashes);
    println!("Warnings: {}", summary.warnings.len());
}

fn run_cxr_benchmark_bridge(config: CxrBenchmarkBridgeConfig) -> Result<(), CliError> {
    let started = Instant::now();
    let mut command = cxr_benchmark_harness_command(&config.python, &config.script_path);
    push_path_arg(&mut command, "--manifest", config.manifest_path.as_ref());
    push_path_arg(&mut command, "--splits", config.splits_path.as_ref());
    push_path_arg(&mut command, "--plan", config.plan_path.as_ref());
    push_path_arg(&mut command, "--out", config.out_path.as_ref());
    push_path_arg(&mut command, "--work-dir", config.work_dir.as_ref());
    push_path_arg(&mut command, "--report-dir", config.report_dir.as_ref());
    push_string_arg(&mut command, "--targets", config.targets.as_deref());
    push_string_arg(&mut command, "--uncertain", config.uncertain.as_deref());
    push_string_arg(&mut command, "--baselines", config.baselines.as_deref());
    push_string_arg(&mut command, "--device", config.device.as_deref());
    push_string_arg(&mut command, "--run-id", config.run_id.as_deref());
    if let Some(value) = config.batch_sizes.as_deref().and_then(first_csv_value) {
        command.arg("--batch-size").arg(value);
    }
    if let Some(value) = config.workers.as_deref().and_then(first_csv_value) {
        command.arg("--workers").arg(value);
    }
    push_usize_arg(&mut command, "--max-samples", config.max_samples);
    push_usize_arg(&mut command, "--max-train", config.max_train);
    push_usize_arg(&mut command, "--max-val", config.max_val);
    push_usize_arg(&mut command, "--max-test", config.max_test);
    push_usize_arg(&mut command, "--image-size", config.image_size);
    push_usize_arg(&mut command, "--epochs", config.epochs);
    push_usize_arg(&mut command, "--loader-batches", config.loader_batches);
    push_usize_arg(&mut command, "--warmup-batches", config.warmup_batches);
    if config.smoke {
        command.arg("--smoke");
    }
    if config.force_cache {
        command.arg("--force-cache");
    }
    if config.force_rematerialize {
        command.arg("--force-rematerialize");
    }

    let rendered_command = format_command(&command);
    let status = command.status().map_err(CliError::Io)?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    if let Some(out_path) = &config.out_path {
        let parent = out_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty());
        parent
            .map(fs::create_dir_all)
            .transpose()
            .map_err(CliError::Io)?;
        if !out_path.exists() {
            fs::write(
                out_path,
                format!(
                    "{{\n  \"status\": \"{}\",\n  \"exit_code\": {},\n  \"elapsed_ms\": {:.3},\n  \"bridge\": \"medkit cxr benchmark -> Python harness\",\n  \"command\": {:?}\n}}\n",
                    if status.success() { "ok" } else { "failed" },
                    status.code().unwrap_or(-1),
                    elapsed_ms,
                    rendered_command
                ),
            )
            .map_err(CliError::Io)?;
        }
    }
    if !status.success() {
        return Err(CliError::Message(format!(
            "cxr benchmark harness failed with status {status}"
        )));
    }
    println!("CXR benchmark command: {rendered_command}");
    println!("Elapsed: {:.2} ms", elapsed_ms);
    Ok(())
}

fn ensure_cxr_ingest_cache_validation_ok(status: &str) -> Result<(), CliError> {
    if status == "failed" {
        Err(CliError::Message(
            "CXR ingest completed with failed cache validation".to_string(),
        ))
    } else {
        Ok(())
    }
}

fn cxr_benchmark_harness_command(runner: &str, script_path: &Path) -> ProcessCommand {
    let mut command = ProcessCommand::new(runner);
    if runner == "uv" {
        command.arg("run");
    }
    command.arg(script_path);
    command
}

fn push_path_arg(command: &mut ProcessCommand, flag: &str, value: Option<&PathBuf>) {
    if let Some(value) = value {
        command.arg(flag).arg(value);
    }
}

fn push_string_arg(command: &mut ProcessCommand, flag: &str, value: Option<&str>) {
    if let Some(value) = value {
        command.arg(flag).arg(value);
    }
}

fn push_usize_arg(command: &mut ProcessCommand, flag: &str, value: Option<usize>) {
    if let Some(value) = value {
        command.arg(flag).arg(value.to_string());
    }
}

fn first_csv_value(value: &str) -> Option<&str> {
    value
        .split(',')
        .map(str::trim)
        .find(|item| !item.is_empty())
}

fn format_command(command: &ProcessCommand) -> String {
    let mut parts = Vec::new();
    parts.push(command.get_program().to_string_lossy().into_owned());
    parts.extend(
        command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned()),
    );
    parts.join(" ")
}

#[derive(Debug)]
enum CliError {
    Io(std::io::Error),
    Message(String),
    Dataset(medkit_dataset::DatasetError),
    Cache(medkit_cache::CacheError),
    Sampler(medkit_sampler::SamplerError),
    Bench(medkit_bench::BenchError),
    Cxr(medkit_cxr::CxrError),
    Dicom(medkit_dicom::DicomError),
    Json(serde_json::Error),
}

impl CliError {
    fn usage() -> Self {
        Self::Message(usage())
    }
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Message(message) => write!(f, "{message}"),
            Self::Dataset(error) => write!(f, "{error}"),
            Self::Cache(error) => write!(f, "{error}"),
            Self::Sampler(error) => write!(f, "{error}"),
            Self::Bench(error) => write!(f, "{error}"),
            Self::Cxr(error) => write!(f, "{error}"),
            Self::Dicom(error) => write!(f, "{error}"),
            Self::Json(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for CliError {}

impl From<medkit_dataset::DatasetError> for CliError {
    fn from(value: medkit_dataset::DatasetError) -> Self {
        Self::Dataset(value)
    }
}

impl From<std::io::Error> for CliError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
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

impl From<medkit_cxr::CxrError> for CliError {
    fn from(value: medkit_cxr::CxrError) -> Self {
        Self::Cxr(value)
    }
}

impl From<medkit_dicom::DicomError> for CliError {
    fn from(value: medkit_dicom::DicomError) -> Self {
        Self::Dicom(value)
    }
}

impl From<serde_json::Error> for CliError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

fn usage() -> String {
    "Usage:\n  medkit dataset validate <root> [--images imagesTr] [--labels labelsTr] [--out manifest.json] [--report report.txt]\n  medkit prepare <root> --manifest manifest.json --plan ct-segmentation.toml --cache .medkit/cache [--chunk 96,96,96]\n  medkit sample <cache> --patch 96,96,96 --strategy foreground-balanced --count 10000 --out patches.jsonl\n  medkit bench <cache> --patch 96,96,96 --workers 8 [--samples 10000]\n  medkit bench-plan <cache> --patches patches.jsonl --workers 8 [--samples 10000]\n  medkit cxr manifest --images <dir> [--metadata metadata.csv.gz] [--labels labels.csv.gz] [--reports reports] --out manifest.jsonl\n  medkit cxr manifest --dicom-index dicom-index.jsonl [--labels labels.csv.gz] [--reports reports] --out manifest.jsonl\n  medkit cxr index --images <dir> [--metadata metadata.csv.gz] [--labels labels.csv.gz] [--reports reports] --out manifest.jsonl\n  medkit cxr validate <manifest.jsonl> [--require-frontal] [--check-patient-leakage] [--check-duplicates] --report validation.md\n  medkit cxr split <manifest.jsonl> --by patient_id --train 0.8 --val 0.1 --test 0.1 [--stratify Pneumonia,view_position] [--seed 0] --out splits.json\n  medkit cxr cache <manifest.jsonl> --splits splits.json --plan cxr-512.toml --cache .medkit/cxr-cache\n  medkit cxr validate-cache <cache> [--split train] [--targets Pneumonia] [--image-shape n,c,h,w] [--plan cxr-512.toml] [--report cache-validation.md] [--json cache-validation.json]\n  medkit cxr ingest <raw-dicom> --recipe cxr-dicom-512.toml --labels labels.csv --cache .medkit/cxr-cache --workdir .medkit/cxr-ingest --report ingestion-report.md [--dry-run] [--workers 4]\n  medkit cxr benchmark [--manifest manifest.jsonl] [--splits splits.json] [--plan cxr-512.toml] [--targets Pneumonia] [--baselines pytorch_raw,monai_raw,medkit_cached_mmap] [--batch-sizes 64,128] [--workers 8,16] [--device cuda:0] [--out benchmark.json]\n  medkit dicom scan <root> --out inventory.jsonl --report dicom-report.md [--workers 4]\n  medkit dicom browse <root> --group patient,study,series --out graph.json --report graph-report.md [--workers 4]\n  medkit dicom inspect <file.dcm>\n  medkit dicom pixels --explain <file.dcm>\n  medkit dicom view <file.dcm> [--width 80]".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_error_from_io_preserves_display_message() {
        let error = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let cli_error = CliError::from(error);

        assert_eq!(cli_error.to_string(), "denied");
    }

    #[test]
    fn cxr_benchmark_bridge_uses_uv_run_for_default_runner() {
        let uv = cxr_benchmark_harness_command("uv", Path::new("bench.py"));
        assert_eq!(format_command(&uv), "uv run bench.py");

        let explicit = cxr_benchmark_harness_command("/bin/sh", Path::new("bench.sh"));
        assert_eq!(format_command(&explicit), "/bin/sh bench.sh");
    }

    #[test]
    fn dicom_cli_parser_covers_help_unknown_workers_and_trailing_errors() {
        assert!(parse(&["medkit", "dicom", "--help"])
            .unwrap_err()
            .to_string()
            .contains("Usage:"));
        assert!(parse(&["medkit", "dicom", "bogus"])
            .unwrap_err()
            .to_string()
            .contains("unknown dicom command"));
        assert!(parse(&["medkit", "dicom", "scan"])
            .unwrap_err()
            .to_string()
            .contains("Usage:"));
        assert!(parse(&["medkit", "dicom", "scan", "root", "--help"])
            .unwrap_err()
            .to_string()
            .contains("Usage:"));
        assert!(parse(&["medkit", "dicom", "scan", "root", "--unknown"])
            .unwrap_err()
            .to_string()
            .contains("unknown argument"));
        assert!(matches!(
            parse(&[
                "medkit",
                "dicom",
                "scan",
                "root",
                "--out",
                "index.jsonl",
                "--report",
                "report.md",
                "--workers",
                "2",
            ])
            .unwrap(),
            Command::DicomScan { workers: 2, .. }
        ));

        assert!(parse(&["medkit", "dicom", "browse", "root", "--help"])
            .unwrap_err()
            .to_string()
            .contains("Usage:"));
        assert!(parse(&["medkit", "dicom", "browse", "root", "--unknown"])
            .unwrap_err()
            .to_string()
            .contains("unknown argument"));
        assert!(matches!(
            parse(&[
                "medkit",
                "dicom",
                "browse",
                "root",
                "--group",
                "patient,study",
                "--out",
                "graph.json",
                "--report",
                "graph.md",
                "--workers",
                "3",
            ])
            .unwrap(),
            Command::DicomBrowse { workers: 3, .. }
        ));

        assert!(parse(&["medkit", "dicom", "inspect"])
            .unwrap_err()
            .to_string()
            .contains("Usage:"));
        assert!(parse(&["medkit", "dicom", "inspect", "file.dcm", "extra"])
            .unwrap_err()
            .to_string()
            .contains("unexpected argument"));
        assert!(parse(&["medkit", "dicom", "pixels"])
            .unwrap_err()
            .to_string()
            .contains("Usage:"));
        assert!(parse(&["medkit", "dicom", "pixels", "show"])
            .unwrap_err()
            .to_string()
            .contains("unknown dicom pixels command"));
        assert!(parse(&["medkit", "dicom", "pixels", "--explain"])
            .unwrap_err()
            .to_string()
            .contains("Usage:"));
        assert!(parse(&["medkit", "dicom", "view"])
            .unwrap_err()
            .to_string()
            .contains("Usage:"));
        assert!(parse(&["medkit", "dicom", "view", "file.dcm", "--help"])
            .unwrap_err()
            .to_string()
            .contains("Usage:"));
        assert!(parse(&["medkit", "dicom", "view", "file.dcm", "--unknown"])
            .unwrap_err()
            .to_string()
            .contains("unknown argument"));
    }

    #[test]
    fn cli_error_json_display_and_dicom_worker_run_error_are_covered() {
        let json_error = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
        let cli_error = CliError::from(json_error);
        assert!(cli_error.to_string().contains("EOF"));

        let missing = std::env::temp_dir().join(format!(
            "medkit-cli-missing-dicom-root-{}",
            std::process::id()
        ));
        let error = run([
            OsString::from("medkit"),
            OsString::from("dicom"),
            OsString::from("scan"),
            missing.into_os_string(),
            OsString::from("--out"),
            OsString::from("index.jsonl"),
            OsString::from("--report"),
            OsString::from("report.md"),
            OsString::from("--workers"),
            OsString::from("2"),
        ])
        .unwrap_err();
        assert!(!error.to_string().is_empty());

        assert!(ensure_cxr_ingest_cache_validation_ok("failed")
            .unwrap_err()
            .to_string()
            .contains("failed cache validation"));
        ensure_cxr_ingest_cache_validation_ok("ok").unwrap();
    }

    fn parse(args: &[&str]) -> Result<Command, CliError> {
        parse_args(args.iter().map(OsString::from))
    }
}
