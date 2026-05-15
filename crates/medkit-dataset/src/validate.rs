use std::{
    collections::{btree_map::Entry, BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use medkit_core::{GeometryCompatibility, ImageModality};
use medkit_io::{ImageMetadataReader, NiftiMetadataReader};

use crate::{
    error::DatasetError,
    manifest::{path_string, summarize},
    pairing::{case_id_from_image_path, case_id_from_label_path, is_nifti_path},
    render_report, CaseManifest, DatasetManifest, ImageRecord, Problem, ProblemCode, Result,
};

/// Configuration for dataset validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationConfig {
    root: PathBuf,
    images_dir: PathBuf,
    labels_dir: PathBuf,
}

impl ValidationConfig {
    /// Creates a validation config with default `imagesTr` and `labelsTr` directories.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            images_dir: PathBuf::from("imagesTr"),
            labels_dir: PathBuf::from("labelsTr"),
        }
    }

    /// Sets the image directory, relative to root unless absolute.
    pub fn images_dir(mut self, images_dir: impl Into<PathBuf>) -> Self {
        self.images_dir = images_dir.into();
        self
    }

    /// Sets the label directory, relative to root unless absolute.
    pub fn labels_dir(mut self, labels_dir: impl Into<PathBuf>) -> Self {
        self.labels_dir = labels_dir.into();
        self
    }

    /// Returns the dataset root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns the configured image directory.
    pub fn images_dir_path(&self) -> &Path {
        &self.images_dir
    }

    /// Returns the configured label directory.
    pub fn labels_dir_path(&self) -> &Path {
        &self.labels_dir
    }

    fn resolved_images_dir(&self) -> PathBuf {
        resolve_under_root(&self.root, &self.images_dir)
    }

    fn resolved_labels_dir(&self) -> PathBuf {
        resolve_under_root(&self.root, &self.labels_dir)
    }
}

/// Scans, pairs, validates, and manifests a dataset.
pub fn validate_dataset(config: &ValidationConfig) -> Result<DatasetManifest> {
    let images_dir = config.resolved_images_dir();
    let labels_dir = config.resolved_labels_dir();
    ensure_dir(config.root(), "dataset root")?;
    ensure_dir(&images_dir, "image directory")?;
    ensure_dir(&labels_dir, "label directory")?;

    let image_index = collect_index(&images_dir, case_id_from_image_path)?;
    let label_index = collect_index(&labels_dir, case_id_from_label_path)?;
    let mut case_ids = BTreeSet::new();
    case_ids.extend(image_index.keys().cloned());
    case_ids.extend(label_index.keys().cloned());

    let image_reader = NiftiMetadataReader::with_default_modality(ImageModality::CT);
    let label_reader = NiftiMetadataReader::with_default_modality(ImageModality::Segmentation);

    let mut cases = Vec::with_capacity(case_ids.len());
    for case_id in case_ids {
        let image_entry = image_index.get(&case_id);
        let label_entry = label_index.get(&case_id);
        cases.push(validate_case(
            case_id,
            image_entry,
            label_entry,
            &image_reader,
            &label_reader,
        ));
    }

    let summary = summarize(&cases);
    Ok(DatasetManifest {
        dataset_root: path_string(config.root()),
        images_dir: path_string(&images_dir),
        labels_dir: path_string(&labels_dir),
        summary,
        cases,
    })
}

/// Writes a pretty JSON dataset manifest.
pub fn write_manifest_json(manifest: &DatasetManifest, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|source| DatasetError::io(parent, source))?;
        }
    }
    let json = serde_json::to_string_pretty(manifest)
        .map_err(|source| DatasetError::json(path, source))?;
    fs::write(path, json).map_err(|source| DatasetError::io(path, source))
}

/// Writes a human-readable validation report.
pub fn write_report(manifest: &DatasetManifest, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|source| DatasetError::io(parent, source))?;
        }
    }
    fs::write(path, render_report(manifest)).map_err(|source| DatasetError::io(path, source))
}

fn validate_case(
    case_id: String,
    image_entry: Option<&IndexedPaths>,
    label_entry: Option<&IndexedPaths>,
    image_reader: &NiftiMetadataReader,
    label_reader: &NiftiMetadataReader,
) -> CaseManifest {
    let image_path = image_entry.and_then(IndexedPaths::first);
    let label_path = label_entry.and_then(IndexedPaths::first);
    let mut problems = Vec::new();

    if image_path.is_none() {
        problems.push(Problem::new(
            ProblemCode::MissingImage,
            "case has no image file",
        ));
    }
    if label_path.is_none() {
        problems.push(Problem::new(
            ProblemCode::MissingLabel,
            "case has no label file",
        ));
    }
    if image_entry.is_some_and(IndexedPaths::has_duplicates) {
        problems.push(Problem::new(
            ProblemCode::DuplicateImage,
            "multiple image files map to the same case id",
        ));
    }
    if label_entry.is_some_and(IndexedPaths::has_duplicates) {
        problems.push(Problem::new(
            ProblemCode::DuplicateLabel,
            "multiple label files map to the same case id",
        ));
    }

    let image_spec = image_path.and_then(|path| match image_reader.read_spec(path) {
        Ok(spec) => Some(spec),
        Err(error) => {
            problems.push(Problem::new(
                ProblemCode::ImageReadError,
                format!("failed to read image metadata: {error}"),
            ));
            None
        }
    });
    let label_spec = label_path.and_then(|path| match label_reader.read_spec(path) {
        Ok(spec) => Some(spec),
        Err(error) => {
            problems.push(Problem::new(
                ProblemCode::LabelReadError,
                format!("failed to read label metadata: {error}"),
            ));
            None
        }
    });

    if let (Some(image_spec), Some(label_spec)) = (&image_spec, &label_spec) {
        let report = image_spec
            .geometry()
            .compatibility_with(label_spec.geometry(), Default::default());
        problems.extend(report.mismatches().iter().map(Problem::geometry));
    }

    CaseManifest::new(
        case_id,
        image_path.map(|path| path_string(path)),
        label_path.map(|path| path_string(path)),
        image_spec.as_ref().map(ImageRecord::from_spec),
        label_spec.as_ref().map(ImageRecord::from_spec),
        problems,
    )
}

fn collect_index(
    dir: &Path,
    case_id_from_path: fn(&Path) -> Option<String>,
) -> Result<BTreeMap<String, IndexedPaths>> {
    let mut index = BTreeMap::new();
    let mut paths = Vec::new();
    collect_nifti_paths(dir, &mut paths)?;
    paths.sort();
    for path in paths {
        let Some(case_id) = case_id_from_path(&path) else {
            continue;
        };
        match index.entry(case_id) {
            Entry::Vacant(entry) => {
                entry.insert(IndexedPaths { paths: vec![path] });
            }
            Entry::Occupied(mut entry) => entry.get_mut().paths.push(path),
        }
    }
    Ok(index)
}

fn collect_nifti_paths(dir: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).map_err(|source| DatasetError::io(dir, source))? {
        let entry = entry.map_err(|source| DatasetError::io(dir, source))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|source| DatasetError::io(&path, source))?;
        if file_type.is_dir() {
            collect_nifti_paths(&path, paths)?;
        } else if file_type.is_file() && is_nifti_path(&path) {
            paths.push(path);
        }
    }
    Ok(())
}

fn ensure_dir(path: &Path, name: &str) -> Result<()> {
    let metadata = fs::metadata(path).map_err(|source| DatasetError::io(path, source))?;
    if !metadata.is_dir() {
        return Err(DatasetError::invalid_input(format!(
            "{name} is not a directory: {}",
            path.display()
        )));
    }
    Ok(())
}

fn resolve_under_root(root: &Path, child: &Path) -> PathBuf {
    if child.is_absolute() {
        child.to_path_buf()
    } else {
        root.join(child)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IndexedPaths {
    paths: Vec<PathBuf>,
}

impl IndexedPaths {
    fn first(&self) -> Option<&PathBuf> {
        self.paths.first()
    }

    fn has_duplicates(&self) -> bool {
        self.paths.len() > 1
    }
}
