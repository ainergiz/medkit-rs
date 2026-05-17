// PyO3's generated wrappers currently trigger this lint on valid `PyResult`
// method signatures.
#![allow(clippy::useless_conversion)]

use std::{
    path::PathBuf,
    slice,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc,
    },
    thread::{self, JoinHandle},
};

use medkit_cxr::{CxrCacheReader, CxrIndexedReadMetrics, CxrRecord};
use medkit_python_ffi::{DatasetHandle as NativeDatasetHandle, StorageMode};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyDict, PyModule, PySlice},
};

#[pyclass(module = "medkit_rs._native", name = "DatasetHandle")]
struct DatasetHandle {
    inner: NativeDatasetHandle,
}

#[pyclass(module = "medkit_rs._native", name = "BatchBuffer")]
struct BatchBuffer {
    image: Py<PyAny>,
    label: Py<PyAny>,
    image_ptr: usize,
    label_ptr: usize,
    batch_size: usize,
}

#[pyclass(module = "medkit_rs._native", name = "CxrCacheHandle")]
struct CxrCacheHandle {
    inner: CxrCacheReader,
}

#[pyclass(module = "medkit_rs._native", name = "CxrBatchBuffer")]
struct CxrBatchBuffer {
    image: Py<PyAny>,
    labels: Py<PyAny>,
    mask: Py<PyAny>,
    image_ptr: usize,
    labels_ptr: usize,
    mask_ptr: usize,
    batch_size: usize,
}

#[pyclass(module = "medkit_rs._native", name = "CxrPrefetcher")]
struct CxrPrefetcher {
    buffers: Vec<CxrBatchBuffer>,
    free_tx: Option<Sender<usize>>,
    ready_rx: Receiver<CxrPrefetchMessage>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    slot_leased: Vec<bool>,
    batch_size: usize,
    prefetch_depth: usize,
    read_workers: usize,
    stats: CxrPrefetchStats,
    closed: bool,
}

#[derive(Clone)]
struct CxrWorkerSlot {
    image_ptr: usize,
    labels_ptr: usize,
    mask_ptr: usize,
    image_values_per_sample: usize,
    target_count: usize,
}

enum CxrPrefetchMessage {
    Ready {
        slot_index: usize,
        samples: usize,
        records: Vec<CxrRecord>,
        metrics: Option<CxrIndexedReadMetrics>,
    },
    Done,
    Error(String),
}

#[derive(Debug, Clone, Default)]
struct CxrPrefetchStats {
    batches: usize,
    indexed_batches: usize,
    indexed_runs: usize,
    read_bytes: usize,
    scatter_bytes: usize,
    read_micros: u128,
    scatter_micros: u128,
}

#[pymethods]
impl DatasetHandle {
    #[new]
    #[pyo3(signature = (cache_dir, patches_path, storage = "resident"))]
    fn new(cache_dir: PathBuf, patches_path: PathBuf, storage: &str) -> PyResult<Self> {
        NativeDatasetHandle::open_with_storage(&cache_dir, &patches_path, parse_storage(storage)?)
            .map(|inner| Self { inner })
            .map_err(PyValueError::new_err)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn records(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn patch_x(&self) -> usize {
        self.inner.patch_size()[0]
    }

    #[getter]
    fn patch_y(&self) -> usize {
        self.inner.patch_size()[1]
    }

    #[getter]
    fn patch_z(&self) -> usize {
        self.inner.patch_size()[2]
    }

    #[getter]
    fn image_channels(&self) -> usize {
        self.inner.image_channel_count()
    }

    fn patch_size(&self) -> (usize, usize, usize) {
        let [x, y, z] = self.inner.patch_size();
        (x, y, z)
    }

    fn fill_batch_ptr(
        &self,
        py: Python<'_>,
        start_index: usize,
        batch_size: usize,
        image_ptr: usize,
        label_ptr: usize,
    ) -> PyResult<usize> {
        if image_ptr == 0 || label_ptr == 0 {
            return Err(PyValueError::new_err("null output buffer"));
        }
        py.allow_threads(|| unsafe {
            self.inner.fill_batch_f32_ptr(
                start_index,
                batch_size,
                image_ptr as *mut f32,
                label_ptr as *mut f32,
            )
        })
        .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (batch_size, pin_memory = false))]
    fn allocate_batch(
        &self,
        py: Python<'_>,
        batch_size: usize,
        pin_memory: bool,
    ) -> PyResult<BatchBuffer> {
        if batch_size == 0 {
            return Err(PyValueError::new_err(
                "batch_size must be greater than zero",
            ));
        }
        let [x, y, z] = self.inner.patch_size();
        let torch = PyModule::import_bound(py, "torch")?;
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("dtype", torch.getattr("float32")?)?;
        if pin_memory {
            kwargs.set_item("pin_memory", true)?;
        }
        let image_shape = (batch_size, self.inner.image_channel_count(), z, y, x);
        let label_shape = (batch_size, 1_usize, z, y, x);
        let image = torch.call_method("empty", (image_shape,), Some(&kwargs))?;
        let label = torch.call_method("empty", (label_shape,), Some(&kwargs))?;
        let image_ptr = image.call_method0("data_ptr")?.extract::<usize>()?;
        let label_ptr = label.call_method0("data_ptr")?.extract::<usize>()?;
        Ok(BatchBuffer {
            image: image.into(),
            label: label.into(),
            image_ptr,
            label_ptr,
            batch_size,
        })
    }

    fn fill_batch_buffer(
        &self,
        py: Python<'_>,
        buffer: &BatchBuffer,
        start_index: usize,
        batch_size: usize,
    ) -> PyResult<Py<PyAny>> {
        if batch_size == 0 {
            return Err(PyValueError::new_err(
                "batch_size must be greater than zero",
            ));
        }
        if batch_size > buffer.batch_size {
            return Err(PyValueError::new_err(format!(
                "batch_size {batch_size} exceeds buffer capacity {}",
                buffer.batch_size
            )));
        }
        py.allow_threads(|| unsafe {
            self.inner.fill_batch_f32_ptr(
                start_index,
                batch_size,
                buffer.image_ptr as *mut f32,
                buffer.label_ptr as *mut f32,
            )
        })
        .map_err(PyValueError::new_err)?;

        let out = PyDict::new_bound(py);
        if batch_size == buffer.batch_size {
            out.set_item("image", buffer.image.bind(py))?;
            out.set_item("label", buffer.label.bind(py))?;
        } else {
            let first_dim = PySlice::new_bound(py, 0, batch_size as isize, 1);
            out.set_item("image", buffer.image.bind(py).get_item(first_dim.clone())?)?;
            out.set_item("label", buffer.label.bind(py).get_item(first_dim)?)?;
        }
        Ok(out.into())
    }

    fn __repr__(&self) -> String {
        let [x, y, z] = self.inner.patch_size();
        format!(
            "DatasetHandle(records={}, patch_size=({}, {}, {}), image_channels={})",
            self.inner.len(),
            x,
            y,
            z,
            self.inner.image_channel_count()
        )
    }
}

#[pymethods]
impl CxrCacheHandle {
    #[new]
    #[pyo3(signature = (cache_dir, split = "train"))]
    fn new(cache_dir: PathBuf, split: &str) -> PyResult<Self> {
        CxrCacheReader::open(cache_dir, split)
            .map(|inner| Self { inner })
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn __len__(&self) -> usize {
        self.inner.samples()
    }

    #[getter]
    fn records(&self) -> usize {
        self.inner.samples()
    }

    #[getter]
    fn image_size(&self) -> usize {
        self.inner.image_size()
    }

    #[getter]
    fn target_count(&self) -> usize {
        self.inner.targets().len()
    }

    fn targets(&self) -> Vec<String> {
        self.inner.targets().to_vec()
    }

    fn image_shape(&self) -> (usize, usize, usize, usize) {
        let [batch, channels, height, width] = self.inner.image_shape();
        (batch, channels, height, width)
    }

    #[pyo3(signature = (batch_size, pin_memory = false))]
    fn allocate_cxr_batch(
        &self,
        py: Python<'_>,
        batch_size: usize,
        pin_memory: bool,
    ) -> PyResult<CxrBatchBuffer> {
        allocate_cxr_batch_for_reader(py, &self.inner, batch_size, pin_memory)
    }

    fn fill_cxr_batch_buffer(
        &self,
        py: Python<'_>,
        buffer: &CxrBatchBuffer,
        start_index: usize,
        batch_size: usize,
    ) -> PyResult<Py<PyAny>> {
        if batch_size == 0 {
            return Err(PyValueError::new_err(
                "batch_size must be greater than zero",
            ));
        }
        if batch_size > buffer.batch_size {
            return Err(PyValueError::new_err(format!(
                "batch_size {batch_size} exceeds buffer capacity {}",
                buffer.batch_size
            )));
        }
        let [_samples, channels, height, width] = self.inner.image_shape();
        let target_count = self.inner.targets().len();
        let image_values = batch_size
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| PyValueError::new_err("CXR image batch shape overflow"))?;
        let label_values = batch_size
            .checked_mul(target_count)
            .ok_or_else(|| PyValueError::new_err("CXR label batch shape overflow"))?;
        let written = py
            .allow_threads(|| unsafe {
                let image_out =
                    slice::from_raw_parts_mut(buffer.image_ptr as *mut f32, image_values);
                let labels_out =
                    slice::from_raw_parts_mut(buffer.labels_ptr as *mut f32, label_values);
                let masks_out =
                    slice::from_raw_parts_mut(buffer.mask_ptr as *mut f32, label_values);
                self.inner
                    .fill_batch(start_index, batch_size, image_out, labels_out, masks_out)
            })
            .map_err(|error| PyValueError::new_err(error.to_string()))?;

        let records = self
            .inner
            .records_for_range(start_index, written)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        cxr_batch_to_dict(py, buffer, written, &records)
    }

    fn fill_cxr_indices_buffer(
        &self,
        py: Python<'_>,
        buffer: &CxrBatchBuffer,
        indices: Vec<usize>,
    ) -> PyResult<Py<PyAny>> {
        let batch_size = indices.len();
        if batch_size == 0 {
            return Err(PyValueError::new_err(
                "indices must contain at least one sample",
            ));
        }
        if batch_size > buffer.batch_size {
            return Err(PyValueError::new_err(format!(
                "indices length {batch_size} exceeds buffer capacity {}",
                buffer.batch_size
            )));
        }
        let [_samples, channels, height, width] = self.inner.image_shape();
        let target_count = self.inner.targets().len();
        let image_values = batch_size
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| PyValueError::new_err("CXR image batch shape overflow"))?;
        let label_values = batch_size
            .checked_mul(target_count)
            .ok_or_else(|| PyValueError::new_err("CXR label batch shape overflow"))?;
        let written = py
            .allow_threads(|| unsafe {
                let image_out =
                    slice::from_raw_parts_mut(buffer.image_ptr as *mut f32, image_values);
                let labels_out =
                    slice::from_raw_parts_mut(buffer.labels_ptr as *mut f32, label_values);
                let masks_out =
                    slice::from_raw_parts_mut(buffer.mask_ptr as *mut f32, label_values);
                self.inner
                    .fill_indices(&indices, image_out, labels_out, masks_out)
            })
            .map_err(|error| PyValueError::new_err(error.to_string()))?;

        let records = self
            .inner
            .records_for_indices(&indices)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        cxr_batch_to_dict(py, buffer, written, &records)
    }

    #[pyo3(signature = (batch_size, batches, pin_memory = false, prefetch_depth = 3, read_workers = 1))]
    fn create_cxr_prefetcher(
        &self,
        py: Python<'_>,
        batch_size: usize,
        batches: Vec<Vec<usize>>,
        pin_memory: bool,
        prefetch_depth: usize,
        read_workers: usize,
    ) -> PyResult<CxrPrefetcher> {
        CxrPrefetcher::new(
            py,
            self.inner.clone(),
            batch_size,
            batches,
            pin_memory,
            prefetch_depth,
            read_workers,
        )
    }

    fn __repr__(&self) -> String {
        let [_samples, _channels, height, width] = self.inner.image_shape();
        format!(
            "CxrCacheHandle(split={}, records={}, image_size=({}, {}), targets={})",
            self.inner.split(),
            self.inner.samples(),
            height,
            width,
            self.inner.targets().len()
        )
    }
}

#[pymethods]
impl CxrPrefetcher {
    fn next(&mut self, py: Python<'_>) -> PyResult<Option<(usize, Py<PyAny>)>> {
        if self.closed {
            return Ok(None);
        }
        match self.ready_rx.recv() {
            Ok(CxrPrefetchMessage::Ready {
                slot_index,
                samples,
                records,
                metrics,
            }) => {
                if slot_index >= self.buffers.len() {
                    self.shutdown();
                    return Err(PyRuntimeError::new_err(format!(
                        "native CXR prefetcher returned invalid slot {slot_index}"
                    )));
                }
                if self.slot_leased[slot_index] {
                    self.shutdown();
                    return Err(PyRuntimeError::new_err(format!(
                        "native CXR prefetcher slot {slot_index} was yielded twice"
                    )));
                }
                self.slot_leased[slot_index] = true;
                self.record_ready_metrics(metrics);
                let batch = cxr_batch_to_dict(py, &self.buffers[slot_index], samples, &records)?;
                Ok(Some((slot_index, batch)))
            }
            Ok(CxrPrefetchMessage::Done) => {
                self.shutdown();
                Ok(None)
            }
            Ok(CxrPrefetchMessage::Error(error)) => {
                self.shutdown();
                Err(PyRuntimeError::new_err(error))
            }
            Err(_) => {
                self.shutdown();
                Ok(None)
            }
        }
    }

    fn release(&mut self, slot_index: usize) -> PyResult<()> {
        if slot_index >= self.slot_leased.len() {
            return Err(PyValueError::new_err(format!(
                "slot index {slot_index} out of bounds"
            )));
        }
        if !self.slot_leased[slot_index] {
            return Err(PyValueError::new_err(format!(
                "slot {slot_index} is not currently leased"
            )));
        }
        self.slot_leased[slot_index] = false;
        if let Some(free_tx) = &self.free_tx {
            let _ = free_tx.send(slot_index);
        }
        Ok(())
    }

    fn close(&mut self) {
        self.shutdown();
    }

    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let out = PyDict::new_bound(py);
        out.set_item("batches", self.stats.batches)?;
        out.set_item("indexed_batches", self.stats.indexed_batches)?;
        out.set_item("indexed_runs", self.stats.indexed_runs)?;
        out.set_item("read_bytes", self.stats.read_bytes)?;
        out.set_item("scatter_bytes", self.stats.scatter_bytes)?;
        out.set_item("read_micros", self.stats.read_micros)?;
        out.set_item("scatter_micros", self.stats.scatter_micros)?;
        out.set_item("read_workers", self.read_workers)?;
        Ok(out.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "CxrPrefetcher(batch_size={}, prefetch_depth={}, read_workers={}, slots={}, closed={})",
            self.batch_size,
            self.prefetch_depth,
            self.read_workers,
            self.buffers.len(),
            self.closed
        )
    }
}

impl CxrPrefetcher {
    fn new(
        py: Python<'_>,
        reader: CxrCacheReader,
        batch_size: usize,
        batches: Vec<Vec<usize>>,
        pin_memory: bool,
        prefetch_depth: usize,
        read_workers: usize,
    ) -> PyResult<Self> {
        if batch_size == 0 {
            return Err(PyValueError::new_err(
                "batch_size must be greater than zero",
            ));
        }
        for (batch_index, indices) in batches.iter().enumerate() {
            if indices.len() > batch_size {
                return Err(PyValueError::new_err(format!(
                    "prefetch batch {batch_index} has {} indices, exceeding batch_size {batch_size}",
                    indices.len()
                )));
            }
            for sample_index in indices {
                if *sample_index >= reader.samples() {
                    return Err(PyValueError::new_err(format!(
                        "prefetch batch {batch_index} contains sample index {sample_index}, \
                         out of bounds for {} samples",
                        reader.samples()
                    )));
                }
            }
        }
        let slot_count = prefetch_depth.max(1);
        let mut buffers = Vec::with_capacity(slot_count);
        let mut worker_slots = Vec::with_capacity(slot_count);
        let [_samples, channels, height, width] = reader.image_shape();
        let image_values_per_sample = channels
            .checked_mul(height)
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| PyValueError::new_err("CXR image shape overflow"))?;
        let target_count = reader.targets().len();
        for _ in 0..slot_count {
            let buffer = allocate_cxr_batch_for_reader(py, &reader, batch_size, pin_memory)?;
            worker_slots.push(CxrWorkerSlot {
                image_ptr: buffer.image_ptr,
                labels_ptr: buffer.labels_ptr,
                mask_ptr: buffer.mask_ptr,
                image_values_per_sample,
                target_count,
            });
            buffers.push(buffer);
        }

        let (free_tx, free_rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let worker = thread::spawn(move || {
            run_cxr_prefetch_worker(
                reader,
                worker_slots,
                batches,
                read_workers.max(1),
                free_rx,
                ready_tx,
                worker_stop,
            );
        });
        for slot_index in 0..slot_count {
            let _ = free_tx.send(slot_index);
        }

        Ok(Self {
            buffers,
            free_tx: Some(free_tx),
            ready_rx,
            stop,
            worker: Some(worker),
            slot_leased: vec![false; slot_count],
            batch_size,
            prefetch_depth: slot_count,
            read_workers: read_workers.max(1),
            stats: CxrPrefetchStats::default(),
            closed: false,
        })
    }

    fn record_ready_metrics(&mut self, metrics: Option<CxrIndexedReadMetrics>) {
        self.stats.batches += 1;
        if let Some(metrics) = metrics {
            self.stats.indexed_batches += 1;
            self.stats.indexed_runs += metrics.runs;
            self.stats.read_bytes += metrics.read_bytes;
            self.stats.scatter_bytes += metrics.scatter_bytes;
            self.stats.read_micros += metrics.read_micros;
            self.stats.scatter_micros += metrics.scatter_micros;
        }
    }

    fn shutdown(&mut self) {
        if self.closed && self.worker.is_none() {
            return;
        }
        self.stop.store(true, Ordering::Relaxed);
        if let Some(free_tx) = self.free_tx.take() {
            let _ = free_tx.send(0);
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        self.closed = true;
    }
}

impl Drop for CxrPrefetcher {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[pyfunction]
#[pyo3(signature = (cache_dir, patches_path, storage = "resident"))]
fn open_dataset(
    cache_dir: PathBuf,
    patches_path: PathBuf,
    storage: &str,
) -> PyResult<DatasetHandle> {
    DatasetHandle::new(cache_dir, patches_path, storage)
}

#[pyfunction]
#[pyo3(signature = (cache_dir, split = "train"))]
fn open_cxr_cache(cache_dir: PathBuf, split: &str) -> PyResult<CxrCacheHandle> {
    CxrCacheHandle::new(cache_dir, split)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DatasetHandle>()?;
    m.add_class::<BatchBuffer>()?;
    m.add_class::<CxrCacheHandle>()?;
    m.add_class::<CxrBatchBuffer>()?;
    m.add_class::<CxrPrefetcher>()?;
    m.add_function(wrap_pyfunction!(open_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(open_cxr_cache, m)?)?;
    Ok(())
}

fn allocate_cxr_batch_for_reader(
    py: Python<'_>,
    reader: &CxrCacheReader,
    batch_size: usize,
    pin_memory: bool,
) -> PyResult<CxrBatchBuffer> {
    if batch_size == 0 {
        return Err(PyValueError::new_err(
            "batch_size must be greater than zero",
        ));
    }
    let [_samples, channels, height, width] = reader.image_shape();
    let target_count = reader.targets().len();
    let torch = PyModule::import_bound(py, "torch")?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dtype", torch.getattr("float32")?)?;
    if pin_memory {
        kwargs.set_item("pin_memory", true)?;
    }
    let image = torch.call_method(
        "empty",
        ((batch_size, channels, height, width),),
        Some(&kwargs),
    )?;
    let labels = torch.call_method("empty", ((batch_size, target_count),), Some(&kwargs))?;
    let mask = torch.call_method("empty", ((batch_size, target_count),), Some(&kwargs))?;
    let image_ptr = image.call_method0("data_ptr")?.extract::<usize>()?;
    let labels_ptr = labels.call_method0("data_ptr")?.extract::<usize>()?;
    let mask_ptr = mask.call_method0("data_ptr")?.extract::<usize>()?;
    if image_ptr == 0 || labels_ptr == 0 || mask_ptr == 0 {
        return Err(PyRuntimeError::new_err(
            "Torch returned a null CXR tensor pointer",
        ));
    }
    Ok(CxrBatchBuffer {
        image: image.into(),
        labels: labels.into(),
        mask: mask.into(),
        image_ptr,
        labels_ptr,
        mask_ptr,
        batch_size,
    })
}

fn cxr_batch_to_dict(
    py: Python<'_>,
    buffer: &CxrBatchBuffer,
    written: usize,
    records: &[CxrRecord],
) -> PyResult<Py<PyAny>> {
    if written == 0 {
        return Err(PyRuntimeError::new_err(
            "native CXR prefetcher returned an empty batch",
        ));
    }
    if written > buffer.batch_size {
        return Err(PyRuntimeError::new_err(format!(
            "native CXR prefetcher wrote {written} samples into capacity {}",
            buffer.batch_size
        )));
    }
    if records.len() != written {
        return Err(PyRuntimeError::new_err(format!(
            "native CXR metadata sidecar has {} records for {written} samples",
            records.len()
        )));
    }
    let out = PyDict::new_bound(py);
    if written == buffer.batch_size {
        out.set_item("image", buffer.image.bind(py))?;
        out.set_item("labels", buffer.labels.bind(py))?;
        out.set_item("mask", buffer.mask.bind(py))?;
    } else {
        let first_dim = PySlice::new_bound(py, 0, written as isize, 1);
        out.set_item("image", buffer.image.bind(py).get_item(first_dim.clone())?)?;
        let labels = buffer.labels.bind(py).get_item(first_dim.clone())?;
        out.set_item("labels", labels)?;
        out.set_item("mask", buffer.mask.bind(py).get_item(first_dim)?)?;
    }
    add_cxr_metadata(py, &out, records)?;
    Ok(out.into())
}

fn add_cxr_metadata(
    py: Python<'_>,
    out: &Bound<'_, PyDict>,
    records: &[CxrRecord],
) -> PyResult<()> {
    let metadata = PyDict::new_bound(py);
    let sample_ids = records
        .iter()
        .map(|record| record.sample_id.clone())
        .collect::<Vec<_>>();
    let patient_ids = records
        .iter()
        .map(|record| record.patient_id.clone())
        .collect::<Vec<_>>();
    let study_ids = records
        .iter()
        .map(|record| record.study_id.clone())
        .collect::<Vec<_>>();
    let image_ids = records
        .iter()
        .map(|record| record.image_id.clone())
        .collect::<Vec<_>>();
    let image_paths = records
        .iter()
        .map(|record| record.image_path.clone())
        .collect::<Vec<_>>();
    metadata.set_item("sample_id", &sample_ids)?;
    metadata.set_item("patient_id", &patient_ids)?;
    metadata.set_item("study_id", &study_ids)?;
    metadata.set_item("image_id", &image_ids)?;
    metadata.set_item("image_path", &image_paths)?;
    out.set_item("metadata", metadata)?;
    out.set_item("sample_id", sample_ids)?;
    out.set_item("patient_id", patient_ids)?;
    out.set_item("study_id", study_ids)?;
    out.set_item("image_id", image_ids)?;
    out.set_item("image_path", image_paths)?;
    Ok(())
}

fn run_cxr_prefetch_worker(
    reader: CxrCacheReader,
    slots: Vec<CxrWorkerSlot>,
    batches: Vec<Vec<usize>>,
    read_workers: usize,
    free_rx: Receiver<usize>,
    ready_tx: Sender<CxrPrefetchMessage>,
    stop: Arc<AtomicBool>,
) {
    for indices in batches {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        if indices.is_empty() {
            continue;
        }
        let slot_index = match free_rx.recv() {
            Ok(value) => value,
            Err(_) => break,
        };
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let Some(slot) = slots.get(slot_index) else {
            let _ = ready_tx.send(CxrPrefetchMessage::Error(format!(
                "native CXR prefetcher received invalid free slot {slot_index}"
            )));
            break;
        };
        match fill_cxr_prefetch_slot(&reader, slot, &indices, read_workers) {
            Ok((samples, metrics)) => {
                let records = reader
                    .records_for_indices(&indices[..samples])
                    .expect("CXR prefetch fill returned only valid requested sample indices");
                if ready_tx
                    .send(CxrPrefetchMessage::Ready {
                        slot_index,
                        samples,
                        records,
                        metrics,
                    })
                    .is_err()
                {
                    break;
                }
            }
            Err(error) => {
                let _ = ready_tx.send(CxrPrefetchMessage::Error(error));
                break;
            }
        }
    }
    let _ = ready_tx.send(CxrPrefetchMessage::Done);
}

fn fill_cxr_prefetch_slot(
    reader: &CxrCacheReader,
    slot: &CxrWorkerSlot,
    indices: &[usize],
    read_workers: usize,
) -> Result<(usize, Option<CxrIndexedReadMetrics>), String> {
    let samples = indices.len();
    let image_values = samples
        .checked_mul(slot.image_values_per_sample)
        .ok_or_else(|| "CXR prefetch image shape overflow".to_string())?;
    let label_values = samples
        .checked_mul(slot.target_count)
        .ok_or_else(|| "CXR prefetch label shape overflow".to_string())?;
    let contiguous_start = contiguous_batch_start(indices);
    unsafe {
        let image_out = slice::from_raw_parts_mut(slot.image_ptr as *mut f32, image_values);
        let labels_out = slice::from_raw_parts_mut(slot.labels_ptr as *mut f32, label_values);
        let masks_out = slice::from_raw_parts_mut(slot.mask_ptr as *mut f32, label_values);
        match contiguous_start {
            Some(start_index) => reader
                .fill_batch(start_index, samples, image_out, labels_out, masks_out)
                .map(|written| (written, None))
                .map_err(|error| error.to_string()),
            None if read_workers <= 1 => reader
                .fill_indices(indices, image_out, labels_out, masks_out)
                .map(|written| (written, None))
                .map_err(|error| error.to_string()),
            None => reader
                .fill_indices_parallel(indices, image_out, labels_out, masks_out, read_workers)
                .map(|metrics| (metrics.samples, Some(metrics)))
                .map_err(|error| error.to_string()),
        }
    }
}

fn contiguous_batch_start(indices: &[usize]) -> Option<usize> {
    let (&start, rest) = indices.split_first()?;
    if rest
        .iter()
        .enumerate()
        .all(|(offset, value)| *value == start + offset + 1)
    {
        Some(start)
    } else {
        None
    }
}

fn parse_storage(value: &str) -> PyResult<StorageMode> {
    match value {
        "resident" => Ok(StorageMode::Resident),
        "chunked" => Ok(StorageMode::Chunked),
        other => Err(PyValueError::new_err(format!(
            "unsupported storage mode {other:?}; expected 'resident' or 'chunked'"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
        Arc,
    };

    fn bare_prefetcher(ready_rx: Receiver<CxrPrefetchMessage>) -> CxrPrefetcher {
        CxrPrefetcher {
            buffers: Vec::new(),
            free_tx: None,
            ready_rx,
            stop: Arc::new(AtomicBool::new(false)),
            worker: None,
            slot_leased: vec![false],
            batch_size: 2,
            prefetch_depth: 1,
            read_workers: 3,
            stats: CxrPrefetchStats::default(),
            closed: false,
        }
    }

    #[test]
    fn contiguous_batch_start_detects_only_strictly_contiguous_indices() {
        assert_eq!(contiguous_batch_start(&[]), None);
        assert_eq!(contiguous_batch_start(&[4]), Some(4));
        assert_eq!(contiguous_batch_start(&[4, 5, 6]), Some(4));
        assert_eq!(contiguous_batch_start(&[4, 6]), None);
        assert_eq!(contiguous_batch_start(&[6, 5]), None);
        assert_eq!(contiguous_batch_start(&[2, 3, 3]), None);
    }

    #[test]
    fn cxr_prefetcher_records_indexed_metrics() {
        let (_ready_tx, ready_rx) = mpsc::channel();
        let mut prefetcher = bare_prefetcher(ready_rx);

        prefetcher.record_ready_metrics(None);
        prefetcher.record_ready_metrics(Some(CxrIndexedReadMetrics {
            samples: 2,
            runs: 4,
            workers: 3,
            read_bytes: 128,
            scatter_bytes: 64,
            read_micros: 11,
            scatter_micros: 7,
        }));

        assert_eq!(prefetcher.stats.batches, 2);
        assert_eq!(prefetcher.stats.indexed_batches, 1);
        assert_eq!(prefetcher.stats.indexed_runs, 4);
        assert_eq!(prefetcher.stats.read_bytes, 128);
        assert_eq!(prefetcher.stats.scatter_bytes, 64);
        assert_eq!(prefetcher.stats.read_micros, 11);
        assert_eq!(prefetcher.stats.scatter_micros, 7);
    }

    #[test]
    fn cxr_prefetcher_shutdown_marks_closed_and_wakes_free_receiver() {
        let (free_tx, free_rx) = mpsc::channel();
        let (_ready_tx, ready_rx) = mpsc::channel();
        let mut prefetcher = bare_prefetcher(ready_rx);
        prefetcher.free_tx = Some(free_tx);

        prefetcher.shutdown();

        assert!(prefetcher.closed);
        assert!(prefetcher.free_tx.is_none());
        assert!(prefetcher.stop.load(Ordering::Relaxed));
        assert_eq!(free_rx.recv().unwrap(), 0);

        prefetcher.shutdown();
        assert!(prefetcher.closed);
    }

    #[test]
    fn cxr_prefetcher_repr_reflects_native_state() {
        let (_ready_tx, ready_rx) = mpsc::channel();
        let prefetcher = bare_prefetcher(ready_rx);

        assert_eq!(
            prefetcher.__repr__(),
            "CxrPrefetcher(batch_size=2, prefetch_depth=1, read_workers=3, slots=0, closed=false)"
        );
    }
}

// Embedded-Python tests need libpython symbols, while the default extension
// build uses host-interpreter dynamic lookup for extension-module packaging.
#[cfg(all(test, not(feature = "extension-module")))]
mod pyo3_tests {
    use super::*;

    use std::{
        collections::BTreeMap,
        fs,
        path::{Path, PathBuf},
        sync::{Mutex, Once},
        time::{SystemTime, UNIX_EPOCH},
    };

    use pyo3::types::{PyDict, PyList, PyTuple};
    use sha2::{Digest, Sha256};

    const CASE_ID: &str = "case_a";
    const SHAPE: [usize; 3] = [4, 3, 2];
    const PATCH_SIZE: [usize; 3] = [2, 2, 2];
    const FIRST_PATCH_START: [usize; 3] = [1, 1, 0];
    const SECOND_PATCH_START: [usize; 3] = [0, 0, 0];

    fn with_python<R>(test: impl FnOnce(Python<'_>) -> R) -> R {
        static INIT: Once = Once::new();
        INIT.call_once(pyo3::prepare_freethreaded_python);
        Python::with_gil(test)
    }

    fn with_fake_torch<R>(test: impl FnOnce(Python<'_>) -> R) -> R {
        static TORCH_LOCK: Mutex<()> = Mutex::new(());
        let _guard = TORCH_LOCK.lock().unwrap();
        with_python(|py| {
            install_fake_torch(py);
            test(py)
        })
    }

    fn install_fake_torch(py: Python<'_>) {
        let code = r#"
import ctypes
import sys

float32 = "float32"
calls = []

class Tensor:
    def __init__(self, shape, fill=0.0, array=None):
        self.shape = tuple(shape)
        size = 1
        for value in self.shape:
            size *= int(value)
        self._size = size
        self._array = array if array is not None else (ctypes.c_float * size)()
        if array is None and fill:
            for index in range(size):
                self._array[index] = fill
    def data_ptr(self):
        return ctypes.addressof(self._array)
    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.shape[0])
            if step != 1:
                raise ValueError("fake tensor only supports unit-step slices")
            return Tensor((max(0, stop - start),) + self.shape[1:], array=self._array)
        raise TypeError("fake tensor only supports first-dimension slices")
    def tolist(self):
        return [float(self._array[index]) for index in range(self._size)]

def empty(shape, **kwargs):
    calls.append(("empty", tuple(shape), dict(kwargs)))
    return Tensor(shape)

def empty_like(tensor):
    calls.append(("empty_like", tensor.shape, {}))
    return Tensor(tensor.shape)
"#;
        let module = PyModule::from_code_bound(py, code, "fake_torch.py", "torch").unwrap();
        let sys = PyModule::import_bound(py, "sys").unwrap();
        sys.getattr("modules")
            .unwrap()
            .set_item("torch", module)
            .unwrap();
    }

    fn install_null_torch(py: Python<'_>) {
        let code = r#"
float32 = "float32"
class NullTensor:
    def data_ptr(self):
        return 0
def empty(shape, **kwargs):
    return NullTensor()
def empty_like(tensor):
    return NullTensor()
"#;
        let module = PyModule::from_code_bound(py, code, "null_torch.py", "torch").unwrap();
        let sys = PyModule::import_bound(py, "sys").unwrap();
        sys.getattr("modules")
            .unwrap()
            .set_item("torch", module)
            .unwrap();
    }

    fn sample_record(suffix: &str) -> CxrRecord {
        CxrRecord {
            sample_id: format!("sample-{suffix}"),
            patient_id: format!("patient-{suffix}"),
            study_id: format!("study-{suffix}"),
            image_id: format!("image-{suffix}"),
            image_path: format!("/images/{suffix}.png"),
            source_format: "png".to_string(),
            modality: Some("CR".to_string()),
            view_position: Some("PA".to_string()),
            laterality: None,
            width: Some(256),
            height: Some(256),
            photometric_interpretation: Some("MONOCHROME2".to_string()),
            series_instance_uid: Some(format!("series-{suffix}")),
            sop_instance_uid: Some(format!("sop-{suffix}")),
            transfer_syntax_uid: Some("1.2.840.10008.1.2.1".to_string()),
            pixel_hash: Some(format!("pixel-{suffix}")),
            labels: BTreeMap::new(),
            label_source: Some("fixture".to_string()),
            report_path: Some(format!("/reports/{suffix}.txt")),
            split: Some("train".to_string()),
            sha256: Some(format!("sha256-{suffix}")),
        }
    }

    fn dummy_cxr_buffer(py: Python<'_>, batch_size: usize) -> CxrBatchBuffer {
        CxrBatchBuffer {
            image: vec![10_i32, 20, 30].into_py(py),
            labels: vec![1_i32, 2, 3].into_py(py),
            mask: vec![0_i32, 1, 0].into_py(py),
            image_ptr: 0,
            labels_ptr: 0,
            mask_ptr: 0,
            batch_size,
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
                "medkit-python-{name}-{}-{nanos}",
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

    struct TinyDatasetFixture {
        _root: TempDir,
        cache_dir: PathBuf,
        patches_path: PathBuf,
    }

    impl TinyDatasetFixture {
        fn new(name: &str) -> Self {
            let root = TempDir::new(name);
            let cache_dir = root.path().join("cache");
            let case_dir = cache_dir.join("case-a-key");
            fs::create_dir_all(&case_dir).unwrap();

            let image_path = case_dir.join("image.f32.raw");
            let label_path = case_dir.join("label.u16.raw");
            let image_values = image_values();
            let label_values = label_values();
            write_f32_raw(&image_path, &image_values);
            write_u16_raw(&label_path, &label_values);
            write_f32_raw(&case_dir.join("image.chunks.f32.raw"), &image_values);
            write_u16_raw(&case_dir.join("label.chunks.u16.raw"), &label_values);

            fs::write(
                cache_dir.join("cache_manifest.json"),
                format!(
                    r#"{{
  "version": 1,
  "cache_dir": "{cache_dir}",
  "dataset_manifest_path": "{manifest}",
  "transform_plan_hash": "test-plan-hash",
  "transform_plan": {{"name":"test-plan","operations":[],"image_interpolation":"linear","label_interpolation":"nearest"}},
  "summary": {{"input_cases":1,"cached_cases":1,"failed_cases":0,"foreground_voxels":24,"bytes_written":144}},
  "cases": [{{
    "case_id": "{CASE_ID}",
    "cache_key": "case-a-key",
    "source_metadata_hash": "source-hash",
    "transform_plan_hash": "test-plan-hash",
    "image_path": "{source_image}",
    "label_path": "{source_label}",
    "source_geometry": {geometry},
    "output_geometry": {geometry},
    "image_cache_path": "{image_path}",
    "label_cache_path": "{label_path}",
    "image_chunk_cache_path": "{image_chunk_path}",
    "label_chunk_cache_path": "{label_chunk_path}",
    "chunk_grid": [1,1,1],
    "shape": [4,3,2],
    "chunk_shape": [4,3,2],
    "crop_origin": [0,0,0],
    "applied_operations": [],
    "foreground_voxels": 24,
    "bytes_written": 144
  }}]
}}"#,
                    cache_dir = path_string(&cache_dir),
                    manifest = path_string(&root.path().join("manifest.json")),
                    source_image = path_string(&root.path().join("source-image.nii")),
                    source_label = path_string(&root.path().join("source-label.nii")),
                    geometry = geometry_json(),
                    image_path = path_string(&image_path),
                    label_path = path_string(&label_path),
                    image_chunk_path = path_string(&case_dir.join("image.chunks.f32.raw")),
                    label_chunk_path = path_string(&case_dir.join("label.chunks.u16.raw")),
                ),
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
                _root: root,
                cache_dir,
                patches_path,
            }
        }
    }

    struct TinyCxrFixture {
        _root: TempDir,
        cache_dir: PathBuf,
    }

    impl TinyCxrFixture {
        fn new(name: &str) -> Self {
            let root = TempDir::new(name);
            let cache_dir = root.path().join("cxr-cache");
            fs::create_dir_all(&cache_dir).unwrap();
            write_f32_raw(
                &cache_dir.join("train-images.float32.dat"),
                &[
                    1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0,
                ],
            );
            write_f32_raw(
                &cache_dir.join("train-labels.float32.dat"),
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            );
            write_f32_raw(
                &cache_dir.join("train-masks.float32.dat"),
                &[1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            );
            fs::write(
                cache_dir.join("train-metadata.jsonl"),
                [
                    cxr_record_json("sample-a", "patient-a", "study-a", "image-a"),
                    cxr_record_json("sample-b", "patient-b", "study-b", "image-b"),
                    cxr_record_json("sample-c", "patient-c", "study-c", "image-c"),
                ]
                .join("\n")
                    + "\n",
            )
            .unwrap();
            fs::write(
                cache_dir.join("cache-metadata.json"),
                format!(
                    r#"{{
  "cache_schema_version": 1,
  "report_schema_version": 1,
  "cache_dir": "{cache_dir}",
  "image_size": 2,
  "channels": 1,
  "dtype": "float32",
  "targets": ["No Finding", "Pneumonia"],
  "label_policy": {{"positive":"label=1 mask=1","negative":"label=0 mask=1","uncertain":"ignore","missing":"ignore","loss_mask":"uncertain and missing labels are masked from loss"}},
  "normalization": {{"mean": 0.0, "std": 1.0}},
  "transform_plan_hash": "test-plan-hash",
  "transform_fingerprint": "test-plan-hash",
  "source_manifest_checksum": "checksum",
  "split_names": ["train"],
  "image_size_policy": {{"channels":1,"height":2,"width":2,"dtype":"float32","transform":"fixture"}},
  "splits": {{
    "train": {{
      "samples": 3,
      "shape": [3, 1, 2, 2],
      "images_path": "{images}",
      "images_sha256": "{images_sha256}",
      "labels_path": "{labels}",
      "labels_sha256": "{labels_sha256}",
      "masks_path": "{masks}",
      "masks_sha256": "{masks_sha256}",
      "metadata_path": "{metadata}",
      "metadata_sha256": "{metadata_sha256}"
    }}
  }},
  "failed_samples": [],
  "cache_size_bytes": 0
}}"#,
                    cache_dir = path_string(&cache_dir),
                    images = path_string(&cache_dir.join("train-images.float32.dat")),
                    images_sha256 = sha256_file(&cache_dir.join("train-images.float32.dat")),
                    labels = path_string(&cache_dir.join("train-labels.float32.dat")),
                    labels_sha256 = sha256_file(&cache_dir.join("train-labels.float32.dat")),
                    masks = path_string(&cache_dir.join("train-masks.float32.dat")),
                    masks_sha256 = sha256_file(&cache_dir.join("train-masks.float32.dat")),
                    metadata = path_string(&cache_dir.join("train-metadata.jsonl")),
                    metadata_sha256 = sha256_file(&cache_dir.join("train-metadata.jsonl")),
                ),
            )
            .unwrap();

            Self {
                _root: root,
                cache_dir,
            }
        }
    }

    fn sha256_file(path: &Path) -> String {
        let bytes = fs::read(path).unwrap();
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        format!("{:x}", hasher.finalize())
    }

    fn assert_value_error(py: Python<'_>, error: PyErr, expected: &str) {
        assert!(error.is_instance_of::<PyValueError>(py));
        assert!(
            error.value_bound(py).to_string().contains(expected),
            "expected PyValueError containing {expected:?}, got {error}"
        );
    }

    fn assert_runtime_error(py: Python<'_>, error: PyErr, expected: &str) {
        assert!(error.is_instance_of::<PyRuntimeError>(py));
        assert!(
            error.value_bound(py).to_string().contains(expected),
            "expected PyRuntimeError containing {expected:?}, got {error}"
        );
    }

    fn patch_record(case_id: &str, patch_start: [usize; 3], patch_size: [usize; 3]) -> String {
        format!(
            r#"{{"case_id":"{case_id}","patch_start":[{},{},{}],"patch_size":[{},{},{}]}}"#,
            patch_start[0],
            patch_start[1],
            patch_start[2],
            patch_size[0],
            patch_size[1],
            patch_size[2]
        )
    }

    fn cxr_record_json(
        sample_id: &str,
        patient_id: &str,
        study_id: &str,
        image_id: &str,
    ) -> String {
        format!(
            r#"{{"sample_id":"{sample_id}","patient_id":"{patient_id}","study_id":"{study_id}","image_id":"{image_id}","image_path":"/images/{image_id}.png","source_format":"png","modality":"CR","view_position":"PA","laterality":null,"width":2,"height":2,"photometric_interpretation":"MONOCHROME2","labels":{{"No Finding":1,"Pneumonia":0}},"label_source":"fixture","report_path":"/reports/{study_id}.txt","split":"train","sha256":"sha256-{image_id}"}}"#
        )
    }

    fn expected_image_patch(start: [usize; 3]) -> Vec<f32> {
        expected_patch(start, |index| index as f32 + 0.25)
    }

    fn expected_label_patch_f32(start: [usize; 3]) -> Vec<f32> {
        expected_patch(start, |index| label_value(index) as f32)
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

    fn flat_index(x: usize, y: usize, z: usize, shape: [usize; 3]) -> usize {
        x + shape[0] * (y + shape[1] * z)
    }

    fn volume_len(shape: [usize; 3]) -> usize {
        shape[0] * shape[1] * shape[2]
    }

    fn geometry_json() -> &'static str {
        r#"{"shape":[4,3,2],"spacing":[1.0,1.0,1.0],"origin":[0.0,0.0,0.0],"direction":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]}"#
    }

    fn path_string(path: &Path) -> String {
        path.to_string_lossy().into_owned()
    }

    fn bare_prefetcher(ready_rx: Receiver<CxrPrefetchMessage>) -> CxrPrefetcher {
        CxrPrefetcher {
            buffers: Vec::new(),
            free_tx: None,
            ready_rx,
            stop: Arc::new(AtomicBool::new(false)),
            worker: None,
            slot_leased: vec![false],
            batch_size: 2,
            prefetch_depth: 1,
            read_workers: 3,
            stats: CxrPrefetchStats::default(),
            closed: false,
        }
    }

    #[test]
    fn parse_storage_accepts_supported_modes() {
        assert_eq!(parse_storage("resident").unwrap(), StorageMode::Resident);
        assert_eq!(parse_storage("chunked").unwrap(), StorageMode::Chunked);
    }

    #[test]
    fn parse_storage_rejects_unknown_modes_as_value_error() {
        with_python(|py| {
            let error = parse_storage("mmap").unwrap_err();
            assert_value_error(py, error, "unsupported storage mode \"mmap\"");
        });
    }

    #[test]
    fn dataset_handle_exposes_shape_repr_pointer_fill_and_buffer_fill() {
        let fixture = TinyDatasetFixture::new("dataset-handle-fill");
        with_fake_torch(|py| {
            let dataset = DatasetHandle::new(
                fixture.cache_dir.clone(),
                fixture.patches_path.clone(),
                "resident",
            )
            .unwrap();

            assert_eq!(dataset.__len__(), 2);
            assert_eq!(dataset.records(), 2);
            assert_eq!(dataset.patch_x(), 2);
            assert_eq!(dataset.patch_y(), 2);
            assert_eq!(dataset.patch_z(), 2);
            assert_eq!(dataset.image_channels(), 1);
            assert_eq!(dataset.patch_size(), (2, 2, 2));
            assert_eq!(
                dataset.__repr__(),
                "DatasetHandle(records=2, patch_size=(2, 2, 2), image_channels=1)"
            );

            let mut image = vec![0.0f32; volume_len(PATCH_SIZE)];
            let mut label = vec![0.0f32; volume_len(PATCH_SIZE)];
            let written = dataset
                .fill_batch_ptr(
                    py,
                    0,
                    1,
                    image.as_mut_ptr() as usize,
                    label.as_mut_ptr() as usize,
                )
                .unwrap();
            assert_eq!(written, 1);
            assert_eq!(image, expected_image_patch(FIRST_PATCH_START));
            assert_eq!(label, expected_label_patch_f32(FIRST_PATCH_START));

            let buffer = dataset.allocate_batch(py, 2, true).unwrap();
            let batch = dataset.fill_batch_buffer(py, &buffer, 1, 1).unwrap();
            let batch = batch.bind(py).downcast::<PyDict>().unwrap();
            let image = batch.get_item("image").unwrap().unwrap();
            let label = batch.get_item("label").unwrap().unwrap();
            assert_eq!(
                image
                    .getattr("shape")
                    .unwrap()
                    .extract::<(usize, usize, usize, usize, usize)>()
                    .unwrap(),
                (1, 1, 2, 2, 2)
            );
            assert_eq!(
                image
                    .call_method0("tolist")
                    .unwrap()
                    .extract::<Vec<f32>>()
                    .unwrap()[..volume_len(PATCH_SIZE)],
                expected_image_patch(SECOND_PATCH_START)
            );
            assert_eq!(
                label
                    .call_method0("tolist")
                    .unwrap()
                    .extract::<Vec<f32>>()
                    .unwrap()[..volume_len(PATCH_SIZE)],
                expected_label_patch_f32(SECOND_PATCH_START)
            );

            let torch = PyModule::import_bound(py, "torch").unwrap();
            let calls = torch
                .getattr("calls")
                .unwrap()
                .downcast_into::<PyList>()
                .unwrap();
            assert_eq!(calls.len(), 2);
            let first_call = calls
                .get_item(0)
                .unwrap()
                .downcast_into::<PyTuple>()
                .unwrap();
            assert_eq!(
                first_call.get_item(0).unwrap().extract::<String>().unwrap(),
                "empty"
            );
            let kwargs = first_call
                .get_item(2)
                .unwrap()
                .downcast_into::<PyDict>()
                .unwrap();
            assert!(kwargs
                .get_item("pin_memory")
                .unwrap()
                .unwrap()
                .extract::<bool>()
                .unwrap());

            let full = dataset.fill_batch_buffer(py, &buffer, 0, 2).unwrap();
            let full = full.bind(py).downcast::<PyDict>().unwrap();
            assert_eq!(
                full.get_item("image")
                    .unwrap()
                    .unwrap()
                    .getattr("shape")
                    .unwrap()
                    .extract::<(usize, usize, usize, usize, usize)>()
                    .unwrap(),
                (2, 1, 2, 2, 2)
            );
        });
    }

    #[test]
    fn dataset_handle_rejects_invalid_allocation_and_buffer_sizes() {
        let fixture = TinyDatasetFixture::new("dataset-handle-errors");
        with_fake_torch(|py| {
            let dataset = DatasetHandle::new(
                fixture.cache_dir.clone(),
                fixture.patches_path.clone(),
                "resident",
            )
            .unwrap();

            let error = dataset
                .allocate_batch(py, 0, false)
                .err()
                .expect("allocate_batch should reject zero batch size");
            assert_value_error(py, error, "batch_size must be greater than zero");

            let error = dataset.fill_batch_ptr(py, 0, 1, 0, 1).unwrap_err();
            assert_value_error(py, error, "null output buffer");

            let buffer = dataset.allocate_batch(py, 1, false).unwrap();
            let error = dataset.fill_batch_buffer(py, &buffer, 0, 0).unwrap_err();
            assert_value_error(py, error, "batch_size must be greater than zero");
            let error = dataset.fill_batch_buffer(py, &buffer, 0, 2).unwrap_err();
            assert_value_error(py, error, "exceeds buffer capacity");
        });
    }

    #[test]
    fn module_functions_open_dataset_and_cxr_cache() {
        let dataset_fixture = TinyDatasetFixture::new("module-open-dataset");
        let cxr_fixture = TinyCxrFixture::new("module-open-cxr");

        let dataset = open_dataset(
            dataset_fixture.cache_dir.clone(),
            dataset_fixture.patches_path.clone(),
            "resident",
        )
        .unwrap();
        assert_eq!(dataset.__len__(), 2);

        let cache = open_cxr_cache(cxr_fixture.cache_dir.clone(), "train").unwrap();
        assert_eq!(cache.__len__(), 3);
        assert_eq!(cache.image_shape(), (3, 1, 2, 2));
    }

    #[test]
    fn native_module_registers_exported_classes_and_functions() {
        with_python(|py| {
            let module = PyModule::new_bound(py, "_native").unwrap();
            _native(&module).unwrap();

            assert!(module.getattr("DatasetHandle").is_ok());
            assert!(module.getattr("BatchBuffer").is_ok());
            assert!(module.getattr("CxrCacheHandle").is_ok());
            assert!(module.getattr("CxrBatchBuffer").is_ok());
            assert!(module.getattr("CxrPrefetcher").is_ok());
            assert!(module.getattr("open_dataset").is_ok());
            assert!(module.getattr("open_cxr_cache").is_ok());
        });
    }

    #[test]
    fn native_module_python_constructors_open_handles() {
        let dataset_fixture = TinyDatasetFixture::new("module-python-dataset");
        let cxr_fixture = TinyCxrFixture::new("module-python-cxr");
        with_fake_torch(|py| {
            let module = PyModule::new_bound(py, "_native").unwrap();
            _native(&module).unwrap();

            let dataset = module
                .getattr("DatasetHandle")
                .unwrap()
                .call1((
                    path_string(&dataset_fixture.cache_dir),
                    path_string(&dataset_fixture.patches_path),
                    "resident",
                ))
                .unwrap();
            assert_eq!(
                dataset
                    .call_method0("__len__")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                2
            );
            assert_eq!(
                dataset
                    .call_method0("patch_size")
                    .unwrap()
                    .extract::<(usize, usize, usize)>()
                    .unwrap(),
                (2, 2, 2)
            );

            let cache = module
                .getattr("CxrCacheHandle")
                .unwrap()
                .call1((path_string(&cxr_fixture.cache_dir), "train"))
                .unwrap();
            assert_eq!(
                cache
                    .call_method0("__len__")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                3
            );
            assert_eq!(
                cache
                    .call_method0("image_shape")
                    .unwrap()
                    .extract::<(usize, usize, usize, usize)>()
                    .unwrap(),
                (3, 1, 2, 2)
            );
        });
    }

    #[test]
    fn contiguous_batch_start_detects_only_strictly_contiguous_indices() {
        assert_eq!(contiguous_batch_start(&[]), None);
        assert_eq!(contiguous_batch_start(&[4]), Some(4));
        assert_eq!(contiguous_batch_start(&[4, 5, 6]), Some(4));
        assert_eq!(contiguous_batch_start(&[4, 6]), None);
        assert_eq!(contiguous_batch_start(&[6, 5]), None);
        assert_eq!(contiguous_batch_start(&[2, 3, 3]), None);
    }

    #[test]
    fn add_cxr_metadata_populates_nested_and_flat_fields() {
        with_python(|py| {
            let out = PyDict::new_bound(py);
            let records = vec![sample_record("a"), sample_record("b")];

            add_cxr_metadata(py, &out, &records).unwrap();

            assert_eq!(
                out.get_item("sample_id")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<String>>()
                    .unwrap(),
                vec!["sample-a".to_string(), "sample-b".to_string()]
            );
            assert_eq!(
                out.get_item("image_path")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<String>>()
                    .unwrap(),
                vec!["/images/a.png".to_string(), "/images/b.png".to_string()]
            );

            let metadata = out
                .get_item("metadata")
                .unwrap()
                .unwrap()
                .downcast_into::<PyDict>()
                .unwrap();
            assert_eq!(
                metadata
                    .get_item("patient_id")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<String>>()
                    .unwrap(),
                vec!["patient-a".to_string(), "patient-b".to_string()]
            );
            assert_eq!(
                metadata
                    .get_item("study_id")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<String>>()
                    .unwrap(),
                vec!["study-a".to_string(), "study-b".to_string()]
            );
        });
    }

    #[test]
    fn cxr_cache_handle_allocates_and_fills_contiguous_and_indexed_batches() {
        let fixture = TinyCxrFixture::new("cxr-cache-handle-fill");
        with_fake_torch(|py| {
            let cache = CxrCacheHandle::new(fixture.cache_dir.clone(), "train").unwrap();

            assert_eq!(cache.__len__(), 3);
            assert_eq!(cache.records(), 3);
            assert_eq!(cache.image_size(), 2);
            assert_eq!(cache.target_count(), 2);
            assert_eq!(
                cache.targets(),
                vec!["No Finding".to_string(), "Pneumonia".to_string()]
            );
            assert_eq!(cache.image_shape(), (3, 1, 2, 2));
            assert_eq!(
                cache.__repr__(),
                "CxrCacheHandle(split=train, records=3, image_size=(2, 2), targets=2)"
            );

            let buffer = cache.allocate_cxr_batch(py, 2, true).unwrap();
            let batch = cache.fill_cxr_batch_buffer(py, &buffer, 0, 2).unwrap();
            let batch = batch.bind(py).downcast::<PyDict>().unwrap();
            assert_eq!(
                batch
                    .get_item("sample_id")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<String>>()
                    .unwrap(),
                vec!["sample-a".to_string(), "sample-b".to_string()]
            );
            assert_eq!(
                batch
                    .get_item("image")
                    .unwrap()
                    .unwrap()
                    .call_method0("tolist")
                    .unwrap()
                    .extract::<Vec<f32>>()
                    .unwrap(),
                vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]
            );
            assert_eq!(
                batch
                    .get_item("labels")
                    .unwrap()
                    .unwrap()
                    .call_method0("tolist")
                    .unwrap()
                    .extract::<Vec<f32>>()
                    .unwrap(),
                vec![1.0, 0.0, 0.0, 1.0]
            );

            let indexed = cache
                .fill_cxr_indices_buffer(py, &buffer, vec![2, 0])
                .unwrap();
            let indexed = indexed.bind(py).downcast::<PyDict>().unwrap();
            assert_eq!(
                indexed
                    .get_item("sample_id")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<String>>()
                    .unwrap(),
                vec!["sample-c".to_string(), "sample-a".to_string()]
            );
            assert_eq!(
                indexed
                    .get_item("image")
                    .unwrap()
                    .unwrap()
                    .call_method0("tolist")
                    .unwrap()
                    .extract::<Vec<f32>>()
                    .unwrap(),
                vec![100.0, 200.0, 300.0, 400.0, 1.0, 2.0, 3.0, 4.0]
            );

            let partial = cache.fill_cxr_batch_buffer(py, &buffer, 1, 1).unwrap();
            let partial = partial.bind(py).downcast::<PyDict>().unwrap();
            assert_eq!(
                partial
                    .get_item("labels")
                    .unwrap()
                    .unwrap()
                    .call_method0("tolist")
                    .unwrap()
                    .extract::<Vec<f32>>()
                    .unwrap(),
                vec![0.0, 1.0]
            );
        });
    }

    #[test]
    fn cxr_allocation_rejects_null_torch_pointers() {
        let fixture = TinyCxrFixture::new("cxr-cache-null-torch");
        with_fake_torch(|py| {
            install_null_torch(py);
            let cache = CxrCacheHandle::new(fixture.cache_dir.clone(), "train").unwrap();

            let error = cache
                .allocate_cxr_batch(py, 1, false)
                .err()
                .expect("null torch pointers should be rejected");
            assert_runtime_error(py, error, "Torch returned a null CXR tensor pointer");
        });
    }

    #[test]
    fn cxr_cache_handle_rejects_bad_batch_requests() {
        let fixture = TinyCxrFixture::new("cxr-cache-handle-errors");
        with_fake_torch(|py| {
            let cache = CxrCacheHandle::new(fixture.cache_dir.clone(), "train").unwrap();

            let error = cache
                .allocate_cxr_batch(py, 0, false)
                .err()
                .expect("allocate_cxr_batch should reject zero batch size");
            assert_value_error(py, error, "batch_size must be greater than zero");

            let buffer = cache.allocate_cxr_batch(py, 1, false).unwrap();
            let error = cache.fill_cxr_batch_buffer(py, &buffer, 0, 0).unwrap_err();
            assert_value_error(py, error, "batch_size must be greater than zero");
            let error = cache.fill_cxr_batch_buffer(py, &buffer, 0, 2).unwrap_err();
            assert_value_error(py, error, "exceeds buffer capacity");
            let error = cache
                .fill_cxr_indices_buffer(py, &buffer, Vec::new())
                .unwrap_err();
            assert_value_error(py, error, "indices must contain at least one sample");
            let error = cache
                .fill_cxr_indices_buffer(py, &buffer, vec![0, 1])
                .unwrap_err();
            assert_value_error(py, error, "exceeds buffer capacity");
        });
    }

    #[test]
    fn cxr_batch_to_dict_rejects_invalid_written_counts_before_touching_torch_objects() {
        with_python(|py| {
            let buffer = dummy_cxr_buffer(py, 2);

            let error = cxr_batch_to_dict(py, &buffer, 0, &[]).unwrap_err();
            assert_runtime_error(py, error, "empty batch");

            let records = vec![sample_record("a"), sample_record("b"), sample_record("c")];
            let error = cxr_batch_to_dict(py, &buffer, 3, &records).unwrap_err();
            assert_runtime_error(py, error, "wrote 3 samples into capacity 2");

            let records = vec![sample_record("a")];
            let error = cxr_batch_to_dict(py, &buffer, 2, &records).unwrap_err();
            assert_runtime_error(py, error, "metadata sidecar has 1 records for 2 samples");
        });
    }

    #[test]
    fn cxr_batch_to_dict_returns_full_batch_without_torch() {
        with_python(|py| {
            let buffer = dummy_cxr_buffer(py, 2);
            let records = vec![sample_record("a"), sample_record("b")];

            let batch = cxr_batch_to_dict(py, &buffer, 2, &records).unwrap();
            let batch = batch.bind(py).downcast::<PyDict>().unwrap();

            assert_eq!(
                batch
                    .get_item("image")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<i32>>()
                    .unwrap(),
                vec![10, 20, 30]
            );
            assert_eq!(
                batch
                    .get_item("sample_id")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<String>>()
                    .unwrap(),
                vec!["sample-a".to_string(), "sample-b".to_string()]
            );
        });
    }

    #[test]
    fn cxr_batch_to_dict_slices_partial_batches_without_torch() {
        with_python(|py| {
            let buffer = dummy_cxr_buffer(py, 3);
            let records = vec![sample_record("a"), sample_record("b")];

            let batch = cxr_batch_to_dict(py, &buffer, 2, &records).unwrap();
            let batch = batch.bind(py).downcast::<PyDict>().unwrap();

            assert_eq!(
                batch
                    .get_item("image")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<i32>>()
                    .unwrap(),
                vec![10, 20]
            );
            assert_eq!(
                batch
                    .get_item("labels")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<i32>>()
                    .unwrap(),
                vec![1, 2]
            );
            assert_eq!(
                batch
                    .get_item("mask")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<i32>>()
                    .unwrap(),
                vec![0, 1]
            );
        });
    }

    #[test]
    fn cxr_prefetcher_release_validates_slot_state() {
        with_python(|py| {
            let (_ready_tx, ready_rx) = mpsc::channel();
            let mut prefetcher = bare_prefetcher(ready_rx);

            let error = prefetcher.release(1).unwrap_err();
            assert_value_error(py, error, "slot index 1 out of bounds");

            let error = prefetcher.release(0).unwrap_err();
            assert_value_error(py, error, "slot 0 is not currently leased");

            prefetcher.slot_leased[0] = true;
            prefetcher.release(0).unwrap();
            assert!(!prefetcher.slot_leased[0]);
        });
    }

    #[test]
    fn cxr_prefetcher_release_returns_leased_slot_to_worker_queue() {
        let (free_tx, free_rx) = mpsc::channel();
        let (_ready_tx, ready_rx) = mpsc::channel();
        let mut prefetcher = bare_prefetcher(ready_rx);
        prefetcher.free_tx = Some(free_tx);
        prefetcher.slot_leased[0] = true;

        prefetcher.release(0).unwrap();

        assert!(!prefetcher.slot_leased[0]);
        assert_eq!(free_rx.recv().unwrap(), 0);
    }

    #[test]
    fn cxr_prefetcher_new_validates_requests_before_spawning_worker() {
        let fixture = TinyCxrFixture::new("cxr-prefetcher-validation");
        with_fake_torch(|py| {
            let reader = CxrCacheReader::open(&fixture.cache_dir, "train").unwrap();

            let error = CxrPrefetcher::new(py, reader.clone(), 0, vec![vec![0]], false, 1, 1)
                .err()
                .expect("prefetcher should reject zero batch size");
            assert_value_error(py, error, "batch_size must be greater than zero");

            let error = CxrPrefetcher::new(py, reader.clone(), 1, vec![vec![0, 1]], false, 1, 1)
                .err()
                .expect("prefetcher should reject oversized batches");
            assert_value_error(py, error, "exceeding batch_size 1");

            let error = CxrPrefetcher::new(py, reader, 1, vec![vec![3]], false, 1, 1)
                .err()
                .expect("prefetcher should reject out of bounds sample");
            assert_value_error(py, error, "out of bounds for 3 samples");
        });
    }

    #[test]
    fn cxr_prefetcher_worker_yields_contiguous_and_parallel_indexed_batches() {
        let fixture = TinyCxrFixture::new("cxr-prefetcher-worker");
        with_fake_torch(|py| {
            let cache = CxrCacheHandle::new(fixture.cache_dir.clone(), "train").unwrap();
            let mut prefetcher = cache
                .create_cxr_prefetcher(py, 2, vec![vec![], vec![0, 1], vec![2, 0]], false, 0, 2)
                .unwrap();
            assert_eq!(
                prefetcher.__repr__(),
                "CxrPrefetcher(batch_size=2, prefetch_depth=1, read_workers=2, slots=1, closed=false)"
            );

            let (slot, batch) = prefetcher.next(py).unwrap().unwrap();
            let batch = batch.bind(py).downcast::<PyDict>().unwrap();
            assert_eq!(slot, 0);
            assert_eq!(
                batch
                    .get_item("sample_id")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<String>>()
                    .unwrap(),
                vec!["sample-a".to_string(), "sample-b".to_string()]
            );
            prefetcher.release(slot).unwrap();

            let (slot, batch) = prefetcher.next(py).unwrap().unwrap();
            let batch = batch.bind(py).downcast::<PyDict>().unwrap();
            assert_eq!(slot, 0);
            assert_eq!(
                batch
                    .get_item("sample_id")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<String>>()
                    .unwrap(),
                vec!["sample-c".to_string(), "sample-a".to_string()]
            );
            assert_eq!(prefetcher.stats.batches, 2);
            assert_eq!(prefetcher.stats.indexed_batches, 1);
            assert!(prefetcher.stats.indexed_runs >= 2);
            prefetcher.release(slot).unwrap();

            assert!(prefetcher.next(py).unwrap().is_none());
            assert!(prefetcher.closed);
        });
    }

    #[test]
    fn cxr_prefetcher_worker_reports_invalid_free_slots() {
        let fixture = TinyCxrFixture::new("cxr-prefetcher-invalid-free-slot");
        let reader = CxrCacheReader::open(&fixture.cache_dir, "train").unwrap();
        let slot = CxrWorkerSlot {
            image_ptr: 1,
            labels_ptr: 1,
            mask_ptr: 1,
            image_values_per_sample: 4,
            target_count: 2,
        };
        let (free_tx, free_rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        free_tx.send(3).unwrap();
        drop(free_tx);

        run_cxr_prefetch_worker(
            reader,
            vec![slot],
            vec![vec![0]],
            1,
            free_rx,
            ready_tx,
            stop,
        );

        let message = ready_rx.recv().unwrap();
        assert!(matches!(message, CxrPrefetchMessage::Error(_)));
    }

    #[test]
    fn cxr_prefetcher_stats_reports_recorded_metrics() {
        with_python(|py| {
            let (_ready_tx, ready_rx) = mpsc::channel();
            let mut prefetcher = bare_prefetcher(ready_rx);
            prefetcher.record_ready_metrics(None);
            prefetcher.record_ready_metrics(Some(CxrIndexedReadMetrics {
                samples: 2,
                runs: 4,
                workers: 3,
                read_bytes: 128,
                scatter_bytes: 64,
                read_micros: 11,
                scatter_micros: 7,
            }));

            let stats = prefetcher.stats(py).unwrap();
            let stats = stats.bind(py).downcast::<PyDict>().unwrap();
            assert_eq!(
                stats
                    .get_item("batches")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                2
            );
            assert_eq!(
                stats
                    .get_item("indexed_batches")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                1
            );
            assert_eq!(
                stats
                    .get_item("indexed_runs")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                4
            );
            assert_eq!(
                stats
                    .get_item("read_bytes")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                128
            );
            assert_eq!(
                stats
                    .get_item("scatter_bytes")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                64
            );
            assert_eq!(
                stats
                    .get_item("read_micros")
                    .unwrap()
                    .unwrap()
                    .extract::<u128>()
                    .unwrap(),
                11
            );
            assert_eq!(
                stats
                    .get_item("scatter_micros")
                    .unwrap()
                    .unwrap()
                    .extract::<u128>()
                    .unwrap(),
                7
            );
            assert_eq!(
                stats
                    .get_item("read_workers")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                3
            );
        });
    }

    #[test]
    fn cxr_prefetcher_next_rejects_invalid_ready_slots() {
        with_python(|py| {
            let (ready_tx, ready_rx) = mpsc::channel();
            ready_tx
                .send(CxrPrefetchMessage::Ready {
                    slot_index: 0,
                    samples: 1,
                    records: vec![sample_record("a")],
                    metrics: None,
                })
                .unwrap();
            let mut prefetcher = bare_prefetcher(ready_rx);

            let error = prefetcher.next(py).unwrap_err();
            assert_runtime_error(py, error, "invalid slot 0");
            assert!(prefetcher.closed);
        });
    }

    #[test]
    fn cxr_prefetcher_next_yields_ready_batch_and_records_metrics() {
        with_python(|py| {
            let (ready_tx, ready_rx) = mpsc::channel();
            ready_tx
                .send(CxrPrefetchMessage::Ready {
                    slot_index: 0,
                    samples: 1,
                    records: vec![sample_record("a")],
                    metrics: Some(CxrIndexedReadMetrics {
                        samples: 1,
                        runs: 2,
                        workers: 3,
                        read_bytes: 40,
                        scatter_bytes: 20,
                        read_micros: 5,
                        scatter_micros: 2,
                    }),
                })
                .unwrap();
            let mut prefetcher = bare_prefetcher(ready_rx);
            prefetcher.buffers.push(dummy_cxr_buffer(py, 2));

            let (slot_index, batch) = prefetcher.next(py).unwrap().unwrap();

            assert_eq!(slot_index, 0);
            assert!(prefetcher.slot_leased[0]);
            assert_eq!(prefetcher.stats.batches, 1);
            assert_eq!(prefetcher.stats.indexed_batches, 1);
            assert_eq!(prefetcher.stats.indexed_runs, 2);
            assert_eq!(prefetcher.stats.read_bytes, 40);

            let batch = batch.bind(py).downcast::<PyDict>().unwrap();
            assert_eq!(
                batch
                    .get_item("sample_id")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<String>>()
                    .unwrap(),
                vec!["sample-a".to_string()]
            );
        });
    }

    #[test]
    fn cxr_prefetcher_next_rejects_duplicate_ready_slot() {
        with_python(|py| {
            let (ready_tx, ready_rx) = mpsc::channel();
            for suffix in ["a", "b"] {
                ready_tx
                    .send(CxrPrefetchMessage::Ready {
                        slot_index: 0,
                        samples: 1,
                        records: vec![sample_record(suffix)],
                        metrics: None,
                    })
                    .unwrap();
            }
            let mut prefetcher = bare_prefetcher(ready_rx);
            prefetcher.buffers.push(dummy_cxr_buffer(py, 2));

            assert!(prefetcher.next(py).unwrap().is_some());
            let error = prefetcher.next(py).unwrap_err();

            assert_runtime_error(py, error, "slot 0 was yielded twice");
            assert!(prefetcher.closed);
        });
    }

    #[test]
    fn cxr_prefetcher_next_handles_terminal_worker_messages() {
        with_python(|py| {
            let (ready_tx, ready_rx) = mpsc::channel();
            ready_tx.send(CxrPrefetchMessage::Done).unwrap();
            let mut prefetcher = bare_prefetcher(ready_rx);

            assert!(prefetcher.next(py).unwrap().is_none());
            assert!(prefetcher.closed);

            let (ready_tx, ready_rx) = mpsc::channel();
            ready_tx
                .send(CxrPrefetchMessage::Error("worker failed".to_string()))
                .unwrap();
            let mut prefetcher = bare_prefetcher(ready_rx);

            let error = prefetcher.next(py).unwrap_err();
            assert_runtime_error(py, error, "worker failed");
            assert!(prefetcher.closed);

            let (ready_tx, ready_rx) = mpsc::channel();
            drop(ready_tx);
            let mut prefetcher = bare_prefetcher(ready_rx);

            assert!(prefetcher.next(py).unwrap().is_none());
            assert!(prefetcher.closed);
            assert!(prefetcher.next(py).unwrap().is_none());
            prefetcher.close();
            assert!(prefetcher.closed);
        });
    }

    #[test]
    fn cxr_prefetch_worker_handles_stop_and_channel_edges() {
        let fixture = TinyCxrFixture::new("cxr-prefetch-worker-edges");
        let reader = CxrCacheReader::open(&fixture.cache_dir, "train").unwrap();
        let slot = CxrWorkerSlot {
            image_ptr: 1,
            labels_ptr: 1,
            mask_ptr: 1,
            image_values_per_sample: 4,
            target_count: 2,
        };

        let (_free_tx, free_rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(true));
        run_cxr_prefetch_worker(
            reader.clone(),
            vec![slot.clone()],
            vec![vec![0]],
            1,
            free_rx,
            ready_tx,
            stop,
        );
        assert!(matches!(ready_rx.recv().unwrap(), CxrPrefetchMessage::Done));

        let (free_tx, free_rx) = mpsc::channel();
        drop(free_tx);
        let (ready_tx, ready_rx) = mpsc::channel();
        run_cxr_prefetch_worker(
            reader.clone(),
            vec![slot.clone()],
            vec![vec![0]],
            1,
            free_rx,
            ready_tx,
            Arc::new(AtomicBool::new(false)),
        );
        assert!(matches!(ready_rx.recv().unwrap(), CxrPrefetchMessage::Done));

        let (free_tx, free_rx) = mpsc::channel();
        free_tx.send(0).unwrap();
        let (ready_tx, ready_rx) = mpsc::channel();
        run_cxr_prefetch_worker(
            reader,
            vec![slot],
            vec![vec![0]],
            1,
            free_rx,
            ready_tx,
            Arc::new(AtomicBool::new(true)),
        );
        assert!(matches!(ready_rx.recv().unwrap(), CxrPrefetchMessage::Done));

        let reader = CxrCacheReader::open(&fixture.cache_dir, "train").unwrap();
        let mut images = vec![0.0f32; 4];
        let mut labels = vec![0.0f32; 2];
        let mut masks = vec![0.0f32; 2];
        let slot = CxrWorkerSlot {
            image_ptr: images.as_mut_ptr() as usize,
            labels_ptr: labels.as_mut_ptr() as usize,
            mask_ptr: masks.as_mut_ptr() as usize,
            image_values_per_sample: 4,
            target_count: 2,
        };
        let (free_tx, free_rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let worker = std::thread::spawn(move || {
            run_cxr_prefetch_worker(
                reader,
                vec![slot],
                vec![vec![0]],
                1,
                free_rx,
                ready_tx,
                worker_stop,
            );
        });
        std::thread::sleep(std::time::Duration::from_millis(10));
        stop.store(true, Ordering::Relaxed);
        free_tx.send(0).unwrap();
        worker.join().unwrap();
        assert!(matches!(ready_rx.recv().unwrap(), CxrPrefetchMessage::Done));
    }

    #[test]
    fn cxr_prefetch_worker_reports_fill_errors_and_dropped_ready_receiver() {
        let fixture = TinyCxrFixture::new("cxr-prefetch-worker-fill-errors");
        let reader = CxrCacheReader::open(&fixture.cache_dir, "train").unwrap();
        let mut images = vec![0.0f32; 8];
        let mut labels = vec![0.0f32; 4];
        let mut masks = vec![0.0f32; 4];
        let slot = CxrWorkerSlot {
            image_ptr: images.as_mut_ptr() as usize,
            labels_ptr: labels.as_mut_ptr() as usize,
            mask_ptr: masks.as_mut_ptr() as usize,
            image_values_per_sample: 4,
            target_count: 2,
        };

        let (free_tx, free_rx) = mpsc::channel();
        free_tx.send(0).unwrap();
        let (ready_tx, ready_rx) = mpsc::channel();
        run_cxr_prefetch_worker(
            reader.clone(),
            vec![slot.clone()],
            vec![vec![0, 3]],
            1,
            free_rx,
            ready_tx,
            Arc::new(AtomicBool::new(false)),
        );
        let message = ready_rx.recv().unwrap();
        assert!(matches!(message, CxrPrefetchMessage::Error(_)));

        let (free_tx, free_rx) = mpsc::channel();
        free_tx.send(0).unwrap();
        let (ready_tx, ready_rx) = mpsc::channel();
        drop(ready_rx);
        run_cxr_prefetch_worker(
            reader,
            vec![slot],
            vec![vec![2, 0]],
            1,
            free_rx,
            ready_tx,
            Arc::new(AtomicBool::new(false)),
        );
    }
}
