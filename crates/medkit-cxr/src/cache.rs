use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
    thread,
    time::Instant,
};

use image::{imageops::FilterType, DynamicImage, GrayImage};
use serde_json;

use crate::{
    error::CxrError,
    manifest::{is_dicom_record, read_manifest},
    types::{
        CacheConfig, CacheSplitSummary, CacheSummary, CacheValidationSummary, CxrCacheBatch,
        CxrCacheReader, CxrIndexedReadMetrics, CxrRecord, ImageSizePolicy, LabelPolicy,
        Normalization, SplitFile, ValidateCacheConfig, CXR_CACHE_SCHEMA_VERSION,
        CXR_REPORT_SCHEMA_VERSION,
    },
    util::{collect_targets, directory_size, hash_file, resolve_cache_path, write_json},
};

struct CxrIndexedRun {
    start_sample: usize,
    out_indices: Vec<usize>,
}

#[derive(Debug)]
struct CxrIndexedRunRead {
    out_indices: Vec<usize>,
    images: Vec<f32>,
    labels: Vec<f32>,
    masks: Vec<f32>,
}

pub fn cache_cxr(config: &CacheConfig) -> Result<CacheSummary, CxrError> {
    let records = read_manifest(&config.manifest_path)?;
    let split_file = read_split_file(&config.splits_path)?;
    validate_split_membership(&records, &split_file)?;
    let image_size = image_size_from_plan(&config.plan_path)?;
    fs::create_dir_all(&config.cache_dir)?;
    let targets = collect_targets(&records);
    let transform_plan_hash = hash_file(&config.plan_path)?;
    let transform_description = if records.iter().any(is_dicom_record) {
        "medkit-dicom presentation to MONOCHROME2 u8, resize square, normalize dataset mean/std"
    } else {
        "decode grayscale, resize square, normalize dataset mean/std"
    };
    let train_records = records_for_split(&records, &split_file.train)?;
    let normalization = estimate_normalization(&train_records, image_size)?;
    let mut splits = BTreeMap::new();
    let mut failed_samples = Vec::new();

    for (name, ids) in [
        ("train", &split_file.train),
        ("val", &split_file.val),
        ("test", &split_file.test),
    ] {
        let split_records = records_for_split(&records, ids)?;
        let split_summary = write_cache_split(
            &config.cache_dir,
            name,
            &split_records,
            &targets,
            image_size,
            &normalization,
            &mut failed_samples,
        )?;
        splits.insert(name.to_string(), split_summary);
    }

    let split_names = splits.keys().cloned().collect::<Vec<_>>();
    let summary = CacheSummary {
        cache_schema_version: CXR_CACHE_SCHEMA_VERSION,
        report_schema_version: CXR_REPORT_SCHEMA_VERSION,
        cache_dir: config.cache_dir.display().to_string(),
        image_size,
        channels: 1,
        dtype: "float32".to_string(),
        targets,
        label_policy: LabelPolicy::default(),
        normalization,
        transform_fingerprint: transform_plan_hash.clone(),
        transform_plan_hash,
        source_manifest_checksum: hash_file(&config.manifest_path)?,
        split_names,
        image_size_policy: ImageSizePolicy {
            channels: 1,
            height: image_size,
            width: image_size,
            dtype: "float32".to_string(),
            transform: transform_description.to_string(),
        },
        splits,
        failed_samples,
        cache_size_bytes: directory_size(&config.cache_dir)?,
    };
    write_json(&config.cache_dir.join("cache-metadata.json"), &summary)?;
    Ok(summary)
}

pub fn read_cache_summary(cache_dir: &Path) -> Result<CacheSummary, CxrError> {
    let text = fs::read_to_string(cache_dir.join("cache-metadata.json"))?;
    Ok(serde_json::from_str(&text)?)
}

pub fn validate_cache_cxr(
    config: &ValidateCacheConfig,
) -> Result<CacheValidationSummary, CxrError> {
    let summary = read_cache_summary(&config.cache_dir)?;
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    if summary.cache_schema_version != CXR_CACHE_SCHEMA_VERSION {
        errors.push(format!(
            "unsupported CXR cache schema version {}; expected {}; rebuild with `medkit cxr cache`",
            summary.cache_schema_version, CXR_CACHE_SCHEMA_VERSION
        ));
    }
    if summary.report_schema_version != CXR_REPORT_SCHEMA_VERSION {
        warnings.push(format!(
            "CXR report schema version is {}; current writer is {}",
            summary.report_schema_version, CXR_REPORT_SCHEMA_VERSION
        ));
    }
    if let Some(expected_targets) = &config.expected_targets {
        if expected_targets != &summary.targets {
            errors.push(format!(
                "target-list mismatch: cache has [{}], expected [{}]",
                summary.targets.join(", "),
                expected_targets.join(", ")
            ));
        }
    }
    if let Some(plan_path) = &config.plan_path {
        let expected = hash_file(plan_path)?;
        if summary.transform_fingerprint != expected && summary.transform_plan_hash != expected {
            errors.push(format!(
                "stale transform fingerprint: cache has {}, expected {} from {}",
                if summary.transform_fingerprint.is_empty() {
                    &summary.transform_plan_hash
                } else {
                    &summary.transform_fingerprint
                },
                expected,
                plan_path.display()
            ));
        }
    }

    let checked_splits = match &config.split {
        Some(split) => {
            if summary.splits.contains_key(split) {
                vec![split.clone()]
            } else {
                errors.push(format!("cache split {split:?} not found"));
                Vec::new()
            }
        }
        None => summary.splits.keys().cloned().collect::<Vec<_>>(),
    };

    for split in &checked_splits {
        let split_summary = summary
            .splits
            .get(split)
            .expect("checked split names are drawn from cache metadata");
        let result = validate_cache_split_files(
            &config.cache_dir,
            split,
            split_summary,
            summary.targets.len(),
            config.expected_image_shape,
            &mut errors,
        );
        result?;
    }

    if !summary.failed_samples.is_empty() {
        errors.push(format!(
            "cache contains {} failed preprocessed samples; first failure: {}",
            summary.failed_samples.len(),
            summary.failed_samples[0]
        ));
    }

    let status = if errors.is_empty() { "ok" } else { "failed" }.to_string();
    let report = CacheValidationSummary {
        cache_dir: config.cache_dir.display().to_string(),
        cache_schema_version: summary.cache_schema_version,
        expected_cache_schema_version: CXR_CACHE_SCHEMA_VERSION,
        report_schema_version: summary.report_schema_version,
        status,
        errors,
        warnings,
        targets: summary.targets,
        label_policy: summary.label_policy,
        split_names: if summary.split_names.is_empty() {
            summary.splits.keys().cloned().collect()
        } else {
            summary.split_names
        },
        checked_splits,
        image_size_policy: summary.image_size_policy,
        transform_fingerprint: if summary.transform_fingerprint.is_empty() {
            summary.transform_plan_hash
        } else {
            summary.transform_fingerprint
        },
        source_manifest_checksum: summary.source_manifest_checksum,
        cache_size_bytes: directory_size(&config.cache_dir)?,
    };
    if let Some(path) = &config.json_path {
        write_json(path, &report)?;
    }
    if let Some(path) = &config.report_path {
        write_cache_validation_report(path, &report)?;
    }
    Ok(report)
}

impl CxrCacheReader {
    pub fn open(cache_dir: impl AsRef<Path>, split: impl Into<String>) -> Result<Self, CxrError> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        let split = split.into();
        let summary = read_cache_summary(&cache_dir)?;
        if summary.cache_schema_version != CXR_CACHE_SCHEMA_VERSION {
            return Err(CxrError::Message(format!(
                "unsupported CXR cache schema version {}; expected {}; rebuild with `medkit cxr cache`",
                summary.cache_schema_version, CXR_CACHE_SCHEMA_VERSION
            )));
        }
        let split_summary = summary
            .splits
            .get(&split)
            .cloned()
            .ok_or_else(|| CxrError::Message(format!("cache split {split:?} not found")))?;
        if split_summary.shape[1] != 1 {
            return Err(CxrError::Message(format!(
                "expected single-channel CXR cache, got shape {:?}",
                split_summary.shape
            )));
        }
        let mut file_errors = Vec::new();
        validate_cache_split_files(
            &cache_dir,
            &split,
            &split_summary,
            summary.targets.len(),
            None,
            &mut file_errors,
        )?;
        if !file_errors.is_empty() {
            return Err(CxrError::Message(file_errors.join("; ")));
        }
        let target_count = summary.targets.len();
        let image_values_per_sample = split_summary.shape[1]
            .checked_mul(split_summary.shape[2])
            .and_then(|value| value.checked_mul(split_summary.shape[3]))
            .ok_or_else(|| CxrError::Message("image shape overflow".to_string()))?;
        let metadata_path = resolve_cache_path(&cache_dir, &split_summary.metadata_path);
        let records = read_cache_metadata(&metadata_path)?;
        Ok(Self {
            cache_dir,
            split,
            summary,
            split_summary,
            records,
            image_values_per_sample,
            target_count,
        })
    }

    pub fn split(&self) -> &str {
        &self.split
    }

    pub fn samples(&self) -> usize {
        self.split_summary.samples
    }

    pub fn targets(&self) -> &[String] {
        &self.summary.targets
    }

    pub fn image_shape(&self) -> [usize; 4] {
        self.split_summary.shape
    }

    pub fn image_size(&self) -> usize {
        self.summary.image_size
    }

    pub fn cache_schema_version(&self) -> u32 {
        self.summary.cache_schema_version
    }

    pub fn label_policy(&self) -> &LabelPolicy {
        &self.summary.label_policy
    }

    pub fn records_for_range(
        &self,
        start_index: usize,
        samples: usize,
    ) -> Result<Vec<CxrRecord>, CxrError> {
        let end = start_index
            .checked_add(samples)
            .ok_or_else(|| CxrError::Message("metadata range overflow".to_string()))?;
        if end > self.records.len() {
            return Err(CxrError::Message(format!(
                "metadata range {start_index}..{end} out of bounds for {} samples",
                self.records.len()
            )));
        }
        Ok(self.records[start_index..end].to_vec())
    }

    pub fn records_for_indices(&self, indices: &[usize]) -> Result<Vec<CxrRecord>, CxrError> {
        indices
            .iter()
            .map(|index| {
                self.records.get(*index).cloned().ok_or_else(|| {
                    CxrError::Message(format!(
                        "metadata sample index {index} out of bounds for {} samples",
                        self.records.len()
                    ))
                })
            })
            .collect()
    }

    pub fn read_batch(
        &self,
        start_index: usize,
        batch_size: usize,
    ) -> Result<CxrCacheBatch, CxrError> {
        let samples = self.actual_batch_size(start_index, batch_size);
        let mut images = vec![0.0f32; samples * self.image_values_per_sample];
        let mut labels = vec![0.0f32; samples * self.target_count];
        let mut masks = vec![0.0f32; samples * self.target_count];
        let written = self.fill_batch(
            start_index,
            batch_size,
            &mut images,
            &mut labels,
            &mut masks,
        )?;
        images.truncate(written * self.image_values_per_sample);
        labels.truncate(written * self.target_count);
        masks.truncate(written * self.target_count);
        let records = if written == 0 {
            Vec::new()
        } else {
            self.records[start_index..start_index + written].to_vec()
        };
        Ok(CxrCacheBatch {
            samples: written,
            image_shape: [
                written,
                self.split_summary.shape[1],
                self.split_summary.shape[2],
                self.split_summary.shape[3],
            ],
            labels_shape: [written, self.target_count],
            images,
            labels,
            masks,
            records,
        })
    }

    pub fn fill_batch(
        &self,
        start_index: usize,
        batch_size: usize,
        image_out: &mut [f32],
        labels_out: &mut [f32],
        masks_out: &mut [f32],
    ) -> Result<usize, CxrError> {
        let samples = self.actual_batch_size(start_index, batch_size);
        let image_values = samples * self.image_values_per_sample;
        let label_values = samples * self.target_count;
        if image_out.len() < image_values {
            return Err(CxrError::Message(format!(
                "image output buffer too small: need {image_values}, got {}",
                image_out.len()
            )));
        }
        if labels_out.len() < label_values {
            return Err(CxrError::Message(format!(
                "label output buffer too small: need {label_values}, got {}",
                labels_out.len()
            )));
        }
        if masks_out.len() < label_values {
            return Err(CxrError::Message(format!(
                "mask output buffer too small: need {label_values}, got {}",
                masks_out.len()
            )));
        }
        if samples == 0 {
            return Ok(0);
        }

        let image_offset = start_index
            .checked_mul(self.image_values_per_sample)
            .ok_or_else(|| CxrError::Message("image offset overflow".to_string()))?;
        let label_offset = start_index
            .checked_mul(self.target_count)
            .ok_or_else(|| CxrError::Message("label offset overflow".to_string()))?;
        read_f32_range(
            &resolve_cache_path(&self.cache_dir, &self.split_summary.images_path),
            image_offset,
            &mut image_out[..image_values],
        )?;
        read_f32_range(
            &resolve_cache_path(&self.cache_dir, &self.split_summary.labels_path),
            label_offset,
            &mut labels_out[..label_values],
        )?;
        read_f32_range(
            &resolve_cache_path(&self.cache_dir, &self.split_summary.masks_path),
            label_offset,
            &mut masks_out[..label_values],
        )?;
        Ok(samples)
    }

    pub fn fill_indices(
        &self,
        indices: &[usize],
        image_out: &mut [f32],
        labels_out: &mut [f32],
        masks_out: &mut [f32],
    ) -> Result<usize, CxrError> {
        let samples = indices.len();
        let image_values = samples * self.image_values_per_sample;
        let label_values = samples * self.target_count;
        if image_out.len() < image_values {
            return Err(CxrError::Message(format!(
                "image output buffer too small: need {image_values}, got {}",
                image_out.len()
            )));
        }
        if labels_out.len() < label_values {
            return Err(CxrError::Message(format!(
                "label output buffer too small: need {label_values}, got {}",
                labels_out.len()
            )));
        }
        if masks_out.len() < label_values {
            return Err(CxrError::Message(format!(
                "mask output buffer too small: need {label_values}, got {}",
                masks_out.len()
            )));
        }
        if samples == 0 {
            return Ok(0);
        }

        let mut order: Vec<(usize, usize)> = indices
            .iter()
            .copied()
            .enumerate()
            .map(|(out_index, sample_index)| (sample_index, out_index))
            .collect();
        for (sample_index, _) in &order {
            if *sample_index >= self.samples() {
                return Err(CxrError::Message(format!(
                    "sample index {sample_index} out of bounds for {} samples",
                    self.samples()
                )));
            }
        }
        order.sort_unstable_by_key(|(sample_index, _)| *sample_index);

        let image_path = resolve_cache_path(&self.cache_dir, &self.split_summary.images_path);
        let labels_path = resolve_cache_path(&self.cache_dir, &self.split_summary.labels_path);
        let masks_path = resolve_cache_path(&self.cache_dir, &self.split_summary.masks_path);
        let mut images = File::open(image_path)?;
        let mut labels = File::open(labels_path)?;
        let mut masks = File::open(masks_path)?;

        let mut cursor = 0usize;
        while cursor < order.len() {
            let start_sample = order[cursor].0;
            let mut end = cursor + 1;
            while end < order.len() && order[end].0 == order[end - 1].0 + 1 {
                end += 1;
            }
            let run_samples = end - cursor;
            let image_offset = start_sample
                .checked_mul(self.image_values_per_sample)
                .ok_or_else(|| CxrError::Message("image offset overflow".to_string()))?;
            let label_offset = start_sample
                .checked_mul(self.target_count)
                .ok_or_else(|| CxrError::Message("label offset overflow".to_string()))?;
            let mut image_scratch = vec![0.0f32; run_samples * self.image_values_per_sample];
            let mut label_scratch = vec![0.0f32; run_samples * self.target_count];
            let mut mask_scratch = vec![0.0f32; run_samples * self.target_count];
            read_f32_range_from(&mut images, image_offset, &mut image_scratch)?;
            read_f32_range_from(&mut labels, label_offset, &mut label_scratch)?;
            read_f32_range_from(&mut masks, label_offset, &mut mask_scratch)?;

            for (run_index, (_sample_index, out_index)) in order[cursor..end].iter().enumerate() {
                let src_image_start = run_index * self.image_values_per_sample;
                let dst_image_start = *out_index * self.image_values_per_sample;
                image_out[dst_image_start..dst_image_start + self.image_values_per_sample]
                    .copy_from_slice(
                        &image_scratch
                            [src_image_start..src_image_start + self.image_values_per_sample],
                    );
                let src_label_start = run_index * self.target_count;
                let dst_label_start = *out_index * self.target_count;
                labels_out[dst_label_start..dst_label_start + self.target_count].copy_from_slice(
                    &label_scratch[src_label_start..src_label_start + self.target_count],
                );
                masks_out[dst_label_start..dst_label_start + self.target_count].copy_from_slice(
                    &mask_scratch[src_label_start..src_label_start + self.target_count],
                );
            }
            cursor = end;
        }

        Ok(samples)
    }

    pub fn fill_indices_parallel(
        &self,
        indices: &[usize],
        image_out: &mut [f32],
        labels_out: &mut [f32],
        masks_out: &mut [f32],
        workers: usize,
    ) -> Result<CxrIndexedReadMetrics, CxrError> {
        let samples = indices.len();
        let image_values = samples * self.image_values_per_sample;
        let label_values = samples * self.target_count;
        if image_out.len() < image_values {
            return Err(CxrError::Message(format!(
                "image output buffer too small: need {image_values}, got {}",
                image_out.len()
            )));
        }
        if labels_out.len() < label_values {
            return Err(CxrError::Message(format!(
                "label output buffer too small: need {label_values}, got {}",
                labels_out.len()
            )));
        }
        if masks_out.len() < label_values {
            return Err(CxrError::Message(format!(
                "mask output buffer too small: need {label_values}, got {}",
                masks_out.len()
            )));
        }
        if samples == 0 {
            return Ok(CxrIndexedReadMetrics::default());
        }

        let runs = self.indexed_runs(indices)?;
        let worker_count = workers.max(1).min(runs.len().max(1));
        let image_path = resolve_cache_path(&self.cache_dir, &self.split_summary.images_path);
        let labels_path = resolve_cache_path(&self.cache_dir, &self.split_summary.labels_path);
        let masks_path = resolve_cache_path(&self.cache_dir, &self.split_summary.masks_path);
        if worker_count == 1 {
            return self.fill_indexed_runs_streaming(
                &runs,
                &image_path,
                &labels_path,
                &masks_path,
                image_out,
                labels_out,
                masks_out,
            );
        }

        let read_start = Instant::now();
        let chunk_size = runs.len().div_ceil(worker_count);
        let run_reads = thread::scope(|scope| {
            let mut handles = Vec::new();
            for chunk in runs.chunks(chunk_size) {
                let image_path = &image_path;
                let labels_path = &labels_path;
                let masks_path = &masks_path;
                handles.push(scope.spawn(move || {
                    read_indexed_runs(
                        chunk,
                        image_path,
                        labels_path,
                        masks_path,
                        self.image_values_per_sample,
                        self.target_count,
                    )
                }));
            }

            let mut run_reads = Vec::with_capacity(runs.len());
            for handle in handles {
                let mut chunk_reads = handle.join().expect("indexed CXR read worker panicked")?;
                run_reads.append(&mut chunk_reads);
            }
            Ok::<_, CxrError>(run_reads)
        })?;
        let read_micros = read_start.elapsed().as_micros();

        let scatter_start = Instant::now();
        for run_read in &run_reads {
            for (run_index, out_index) in run_read.out_indices.iter().copied().enumerate() {
                let src_image_start = run_index * self.image_values_per_sample;
                let dst_image_start = out_index * self.image_values_per_sample;
                image_out[dst_image_start..dst_image_start + self.image_values_per_sample]
                    .copy_from_slice(
                        &run_read.images
                            [src_image_start..src_image_start + self.image_values_per_sample],
                    );
                let src_label_start = run_index * self.target_count;
                let dst_label_start = out_index * self.target_count;
                labels_out[dst_label_start..dst_label_start + self.target_count].copy_from_slice(
                    &run_read.labels[src_label_start..src_label_start + self.target_count],
                );
                masks_out[dst_label_start..dst_label_start + self.target_count].copy_from_slice(
                    &run_read.masks[src_label_start..src_label_start + self.target_count],
                );
            }
        }
        let scatter_micros = scatter_start.elapsed().as_micros();
        let bytes = (image_values + label_values + label_values) * std::mem::size_of::<f32>();
        Ok(CxrIndexedReadMetrics {
            samples,
            runs: runs.len(),
            workers: worker_count,
            read_bytes: bytes,
            scatter_bytes: bytes,
            read_micros,
            scatter_micros,
        })
    }

    fn fill_indexed_runs_streaming(
        &self,
        runs: &[CxrIndexedRun],
        image_path: &Path,
        labels_path: &Path,
        masks_path: &Path,
        image_out: &mut [f32],
        labels_out: &mut [f32],
        masks_out: &mut [f32],
    ) -> Result<CxrIndexedReadMetrics, CxrError> {
        let mut images = File::open(image_path)?;
        let mut labels = File::open(labels_path)?;
        let mut masks = File::open(masks_path)?;
        let mut samples = 0usize;
        let mut read_micros = 0u128;
        let mut scatter_micros = 0u128;
        for run in runs {
            samples += run.out_indices.len();
            let image_offset = run
                .start_sample
                .checked_mul(self.image_values_per_sample)
                .ok_or_else(|| CxrError::Message("image offset overflow".to_string()))?;
            let label_offset = run
                .start_sample
                .checked_mul(self.target_count)
                .ok_or_else(|| CxrError::Message("label offset overflow".to_string()))?;
            let mut image_scratch =
                vec![0.0f32; run.out_indices.len() * self.image_values_per_sample];
            let mut label_scratch = vec![0.0f32; run.out_indices.len() * self.target_count];
            let mut mask_scratch = vec![0.0f32; run.out_indices.len() * self.target_count];
            let read_start = Instant::now();
            read_f32_range_from(&mut images, image_offset, &mut image_scratch)?;
            read_f32_range_from(&mut labels, label_offset, &mut label_scratch)?;
            read_f32_range_from(&mut masks, label_offset, &mut mask_scratch)?;
            read_micros += read_start.elapsed().as_micros();

            let scatter_start = Instant::now();
            for (run_index, out_index) in run.out_indices.iter().copied().enumerate() {
                let src_image_start = run_index * self.image_values_per_sample;
                let dst_image_start = out_index * self.image_values_per_sample;
                image_out[dst_image_start..dst_image_start + self.image_values_per_sample]
                    .copy_from_slice(
                        &image_scratch
                            [src_image_start..src_image_start + self.image_values_per_sample],
                    );
                let src_label_start = run_index * self.target_count;
                let dst_label_start = out_index * self.target_count;
                labels_out[dst_label_start..dst_label_start + self.target_count].copy_from_slice(
                    &label_scratch[src_label_start..src_label_start + self.target_count],
                );
                masks_out[dst_label_start..dst_label_start + self.target_count].copy_from_slice(
                    &mask_scratch[src_label_start..src_label_start + self.target_count],
                );
            }
            scatter_micros += scatter_start.elapsed().as_micros();
        }
        let image_values = samples * self.image_values_per_sample;
        let label_values = samples * self.target_count;
        let bytes = (image_values + label_values + label_values) * std::mem::size_of::<f32>();
        Ok(CxrIndexedReadMetrics {
            samples,
            runs: runs.len(),
            workers: 1,
            read_bytes: bytes,
            scatter_bytes: bytes,
            read_micros,
            scatter_micros,
        })
    }

    fn actual_batch_size(&self, start_index: usize, batch_size: usize) -> usize {
        if start_index >= self.samples() {
            return 0;
        }
        batch_size.min(self.samples() - start_index)
    }

    fn indexed_runs(&self, indices: &[usize]) -> Result<Vec<CxrIndexedRun>, CxrError> {
        let mut order: Vec<(usize, usize)> = indices
            .iter()
            .copied()
            .enumerate()
            .map(|(out_index, sample_index)| (sample_index, out_index))
            .collect();
        for (sample_index, _) in &order {
            if *sample_index >= self.samples() {
                return Err(CxrError::Message(format!(
                    "sample index {sample_index} out of bounds for {} samples",
                    self.samples()
                )));
            }
        }
        order.sort_unstable_by_key(|(sample_index, _)| *sample_index);

        let mut runs = Vec::new();
        let mut cursor = 0usize;
        while cursor < order.len() {
            let start_sample = order[cursor].0;
            let mut end = cursor + 1;
            while end < order.len() && order[end].0 == order[end - 1].0 + 1 {
                end += 1;
            }
            runs.push(CxrIndexedRun {
                start_sample,
                out_indices: order[cursor..end]
                    .iter()
                    .map(|(_sample_index, out_index)| *out_index)
                    .collect(),
            });
            cursor = end;
        }
        Ok(runs)
    }
}
fn write_cache_split(
    cache_dir: &Path,
    split: &str,
    records: &[CxrRecord],
    targets: &[String],
    image_size: usize,
    normalization: &Normalization,
    failed_samples: &mut Vec<String>,
) -> Result<CacheSplitSummary, CxrError> {
    let images_name = format!("{split}-images.float32.dat");
    let labels_name = format!("{split}-labels.float32.dat");
    let masks_name = format!("{split}-masks.float32.dat");
    let metadata_name = format!("{split}-metadata.jsonl");
    let images_path = cache_dir.join(&images_name);
    let labels_path = cache_dir.join(&labels_name);
    let masks_path = cache_dir.join(&masks_name);
    let metadata_path = cache_dir.join(&metadata_name);
    let mut images = BufWriter::new(File::create(&images_path)?);
    let mut labels = BufWriter::new(File::create(&labels_path)?);
    let mut masks = BufWriter::new(File::create(&masks_path)?);
    let mut metadata = BufWriter::new(File::create(&metadata_path)?);

    for record in records {
        match preprocess_image(record, image_size, normalization) {
            Ok(values) => {
                for value in values {
                    images.write_all(&value.to_le_bytes())?;
                }
            }
            Err(error) => {
                failed_samples.push(format!("{}: {error}", record.sample_id));
                for _ in 0..(image_size * image_size) {
                    images.write_all(&0.0f32.to_le_bytes())?;
                }
            }
        }
        for target in targets {
            let value = record.labels.get(target).copied().flatten();
            let (label, mask) = match value {
                Some(1) => (1.0f32, 1.0f32),
                Some(0) => (0.0f32, 1.0f32),
                Some(-1) | None => (0.0f32, 0.0f32),
                Some(other) => (other as f32, 1.0f32),
            };
            labels.write_all(&label.to_le_bytes())?;
            masks.write_all(&mask.to_le_bytes())?;
        }
        serde_json::to_writer(&mut metadata, record)?;
        metadata.write_all(b"\n")?;
    }

    images.flush()?;
    labels.flush()?;
    masks.flush()?;
    metadata.flush()?;

    Ok(CacheSplitSummary {
        samples: records.len(),
        shape: [records.len(), 1, image_size, image_size],
        images_path: images_name,
        labels_path: labels_name,
        masks_path: masks_name,
        metadata_path: metadata_name,
    })
}

fn estimate_normalization(
    records: &[CxrRecord],
    image_size: usize,
) -> Result<Normalization, CxrError> {
    let mut sum = 0.0f64;
    let mut sq_sum = 0.0f64;
    let mut count = 0usize;
    let stride = (records.len() / 512).max(1);
    for record in records.iter().step_by(stride) {
        let gray = load_resized_luma(record, image_size)?;
        for value in gray {
            let scaled = value as f64 / 255.0;
            sum += scaled;
            sq_sum += scaled * scaled;
            count += 1;
        }
    }
    if count == 0 {
        return Ok(Normalization {
            mean: 0.5,
            std: 0.25,
        });
    }
    let mean = sum / count as f64;
    let variance = (sq_sum / count as f64 - mean * mean).max(1.0e-6);
    Ok(Normalization {
        mean: mean as f32,
        std: variance.sqrt().max(1.0e-3) as f32,
    })
}

fn preprocess_image(
    record: &CxrRecord,
    image_size: usize,
    normalization: &Normalization,
) -> Result<Vec<f32>, CxrError> {
    let gray = load_resized_luma(record, image_size)?;
    Ok(gray
        .into_iter()
        .map(|value| {
            let scaled = value as f32 / 255.0;
            (scaled - normalization.mean) / normalization.std
        })
        .collect())
}

fn load_resized_luma(record: &CxrRecord, image_size: usize) -> Result<Vec<u8>, CxrError> {
    if is_dicom_record(record) {
        load_resized_dicom_luma(&record.image_path, image_size)
    } else {
        load_resized_raster_luma(&record.image_path, image_size)
    }
}

fn load_resized_raster_luma(path: &str, image_size: usize) -> Result<Vec<u8>, CxrError> {
    let image = image::open(path)?;
    let gray = match image {
        DynamicImage::ImageLuma8(value) => value,
        other => other.to_luma8(),
    };
    let resized = image::imageops::resize(
        &gray,
        image_size as u32,
        image_size as u32,
        FilterType::Triangle,
    );
    Ok(resized.into_raw())
}

fn load_resized_dicom_luma(path: &str, image_size: usize) -> Result<Vec<u8>, CxrError> {
    let image = medkit_dicom::present_dicom_pixels(path)?;
    let gray = GrayImage::from_raw(image.width as u32, image.height as u32, image.pixels)
        .ok_or_else(|| CxrError::Message(format!("invalid DICOM raster shape for {path}")))?;
    let resized = image::imageops::resize(
        &gray,
        image_size as u32,
        image_size as u32,
        FilterType::Triangle,
    );
    Ok(resized.into_raw())
}

pub(crate) fn image_size_from_plan(plan_path: &Path) -> Result<usize, CxrError> {
    let text = fs::read_to_string(plan_path)?;
    let value = toml::from_str::<toml::Value>(&text)?;
    if let Some(size) = value
        .get("image")
        .and_then(|image| image.get("size"))
        .and_then(|size| size.as_array())
        .and_then(|items| items.first())
        .and_then(|item| item.as_integer())
    {
        return Ok(size as usize);
    }
    if let Some(ops) = value.get("operations").and_then(|ops| ops.as_array()) {
        for op in ops {
            if let Some(size) = op.get("size").and_then(|item| item.as_integer()) {
                return Ok(size as usize);
            }
        }
    }
    Err(CxrError::Message(format!(
        "could not determine image size from plan {}",
        plan_path.display()
    )))
}

pub(crate) fn records_for_split(
    records: &[CxrRecord],
    ids: &[String],
) -> Result<Vec<CxrRecord>, CxrError> {
    let by_id = records
        .iter()
        .map(|record| (record.sample_id.as_str(), record))
        .collect::<HashMap<_, _>>();
    ids.iter()
        .map(|id| {
            by_id
                .get(id.as_str())
                .map(|record| (*record).clone())
                .ok_or_else(|| {
                    CxrError::Message(format!(
                        "split artifact references unknown sample_id {id:?}"
                    ))
                })
        })
        .collect()
}

fn validate_split_membership(
    records: &[CxrRecord],
    split_file: &SplitFile,
) -> Result<(), CxrError> {
    let manifest_ids = records
        .iter()
        .map(|record| record.sample_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();
    let mut duplicates = BTreeSet::new();
    let mut unknown = BTreeSet::new();
    for id in split_file
        .train
        .iter()
        .chain(split_file.val.iter())
        .chain(split_file.test.iter())
    {
        if !manifest_ids.contains(id.as_str()) {
            unknown.insert(id.clone());
        }
        if !seen.insert(id.clone()) {
            duplicates.insert(id.clone());
        }
    }
    if !unknown.is_empty() {
        return Err(CxrError::Message(format!(
            "split artifact references unknown sample IDs: {}",
            unknown.into_iter().collect::<Vec<_>>().join(", ")
        )));
    }
    if !duplicates.is_empty() {
        return Err(CxrError::Message(format!(
            "split artifact assigns samples to multiple splits: {}",
            duplicates.into_iter().collect::<Vec<_>>().join(", ")
        )));
    }
    let missing = manifest_ids
        .into_iter()
        .filter(|id| !seen.contains(*id))
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(CxrError::Message(format!(
            "split artifact omits {} manifest samples; first omitted sample_id {:?}",
            missing.len(),
            missing[0]
        )));
    }
    Ok(())
}
pub(crate) fn read_split_file(path: &Path) -> Result<SplitFile, CxrError> {
    let file = File::open(path)?;
    Ok(serde_json::from_reader(file)?)
}
fn write_cache_validation_report(
    path: &Path,
    summary: &CacheValidationSummary,
) -> Result<(), CxrError> {
    let mut report = String::new();
    report.push_str("# CXR Cache Validation Report\n\n");
    report.push_str(&format!("- status: {}\n", summary.status));
    report.push_str(&format!(
        "- cache schema version: {} (expected {})\n",
        summary.cache_schema_version, summary.expected_cache_schema_version
    ));
    report.push_str(&format!(
        "- report schema version: {}\n",
        summary.report_schema_version
    ));
    report.push_str(&format!(
        "- transform fingerprint: {}\n",
        summary.transform_fingerprint
    ));
    report.push_str(&format!(
        "- source manifest checksum: {}\n",
        summary.source_manifest_checksum
    ));
    report.push_str(&format!(
        "- checked splits: {}\n",
        summary.checked_splits.join(", ")
    ));
    report.push_str(&format!("- targets: {}\n", summary.targets.join(", ")));
    report.push_str(&format!(
        "- label policy: uncertain={}, missing={}, loss_mask={}\n",
        summary.label_policy.uncertain,
        summary.label_policy.missing,
        summary.label_policy.loss_mask
    ));
    report.push_str("\n## Errors\n\n");
    if summary.errors.is_empty() {
        report.push_str("- none\n");
    } else {
        for error in &summary.errors {
            report.push_str(&format!("- {error}\n"));
        }
    }
    report.push_str("\n## Warnings\n\n");
    if summary.warnings.is_empty() {
        report.push_str("- none\n");
    } else {
        for warning in &summary.warnings {
            report.push_str(&format!("- {warning}\n"));
        }
    }
    fs::write(path, report)?;
    Ok(())
}

fn validate_cache_split_files(
    cache_dir: &Path,
    split: &str,
    split_summary: &CacheSplitSummary,
    target_count: usize,
    expected_image_shape: Option<[usize; 4]>,
    errors: &mut Vec<String>,
) -> Result<(), CxrError> {
    if let Some(expected) = expected_image_shape {
        if split_summary.shape != expected {
            errors.push(format!(
                "wrong image shape for split {split}: cache has {:?}, expected {:?}",
                split_summary.shape, expected
            ));
        }
    }
    if split_summary.shape[0] != split_summary.samples {
        errors.push(format!(
            "shape/sample mismatch for split {split}: shape[0]={}, samples={}",
            split_summary.shape[0], split_summary.samples
        ));
    }
    let image_values = split_summary
        .shape
        .iter()
        .try_fold(1usize, |acc, value| acc.checked_mul(*value))
        .ok_or_else(|| CxrError::Message(format!("image shape overflow for split {split}")))?;
    let label_values = split_summary
        .samples
        .checked_mul(target_count)
        .ok_or_else(|| CxrError::Message(format!("label shape overflow for split {split}")))?;
    check_file_size(
        &resolve_cache_path(cache_dir, &split_summary.images_path),
        image_values * std::mem::size_of::<f32>(),
        split,
        "images",
        errors,
    )?;
    let label_bytes = label_values * std::mem::size_of::<f32>();
    let labels_path = resolve_cache_path(cache_dir, &split_summary.labels_path);
    check_file_size(&labels_path, label_bytes, split, "labels", errors)?;
    let masks_path = resolve_cache_path(cache_dir, &split_summary.masks_path);
    check_file_size(&masks_path, label_bytes, split, "masks", errors)?;
    let metadata_path = resolve_cache_path(cache_dir, &split_summary.metadata_path);
    match count_jsonl_records(&metadata_path) {
        Ok(count) if count == split_summary.samples => {}
        Ok(count) => errors.push(format!(
            "metadata sample count mismatch for split {split}: metadata has {count}, summary has {}",
            split_summary.samples
        )),
        Err(error) => errors.push(format!(
            "missing or unreadable metadata for split {split}: {error}"
        )),
    }
    Ok(())
}

fn check_file_size(
    path: &Path,
    expected_bytes: usize,
    split: &str,
    kind: &str,
    errors: &mut Vec<String>,
) -> Result<(), CxrError> {
    match fs::metadata(path) {
        Ok(metadata) if metadata.len() == expected_bytes as u64 => {}
        Ok(metadata) => errors.push(format!(
            "wrong {kind} file size for split {split}: {} has {} bytes, expected {}",
            path.display(),
            metadata.len(),
            expected_bytes
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => errors.push(format!(
            "missing {kind} file for split {split}: {}",
            path.display()
        )),
        Err(error) => return Err(CxrError::Io(error)),
    }
    Ok(())
}

fn count_jsonl_records(path: &Path) -> Result<usize, CxrError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut count = 0usize;
    for line in reader.lines() {
        if !line?.trim().is_empty() {
            count += 1;
        }
    }
    Ok(count)
}
fn read_cache_metadata(path: &Path) -> Result<Vec<CxrRecord>, CxrError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        records.push(serde_json::from_str(&line)?);
    }
    Ok(records)
}
fn read_f32_range(path: &Path, start_value: usize, out: &mut [f32]) -> Result<(), CxrError> {
    let mut file = File::open(path)?;
    read_f32_range_from(&mut file, start_value, out)
}

fn read_indexed_runs(
    runs: &[CxrIndexedRun],
    image_path: &Path,
    labels_path: &Path,
    masks_path: &Path,
    image_values_per_sample: usize,
    target_count: usize,
) -> Result<Vec<CxrIndexedRunRead>, CxrError> {
    let mut images = File::open(image_path)?;
    let mut labels = File::open(labels_path)?;
    let mut masks = File::open(masks_path)?;
    let mut reads = Vec::with_capacity(runs.len());
    for run in runs {
        let run_len = run.out_indices.len();
        let image_offset = run
            .start_sample
            .checked_mul(image_values_per_sample)
            .ok_or_else(|| CxrError::Message("image offset overflow".to_string()))?;
        let label_offset = run
            .start_sample
            .checked_mul(target_count)
            .ok_or_else(|| CxrError::Message("label offset overflow".to_string()))?;
        let mut image_scratch = vec![0.0f32; run_len * image_values_per_sample];
        let mut label_scratch = vec![0.0f32; run_len * target_count];
        let mut mask_scratch = vec![0.0f32; run_len * target_count];
        read_f32_range_from(&mut images, image_offset, &mut image_scratch)?;
        read_f32_range_from(&mut labels, label_offset, &mut label_scratch)?;
        read_f32_range_from(&mut masks, label_offset, &mut mask_scratch)?;
        reads.push(CxrIndexedRunRead {
            out_indices: run.out_indices.clone(),
            images: image_scratch,
            labels: label_scratch,
            masks: mask_scratch,
        });
    }
    Ok(reads)
}

fn read_f32_range_from(
    file: &mut File,
    start_value: usize,
    out: &mut [f32],
) -> Result<(), CxrError> {
    file.seek(SeekFrom::Start(byte_offset(start_value)?))?;
    let mut bytes = vec![0u8; std::mem::size_of_val(out)];
    file.read_exact(&mut bytes)?;
    for (slot, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(())
}

fn byte_offset(value_index: usize) -> Result<u64, CxrError> {
    value_index
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| CxrError::Message("cache byte offset overflow".to_string()))
}
