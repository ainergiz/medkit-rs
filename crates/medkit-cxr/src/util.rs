use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufWriter, Read},
    path::{Path, PathBuf},
};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{
    error::CxrError,
    types::{CxrRecord, LabelCount},
};

pub(crate) fn collect_targets(records: &[CxrRecord]) -> Vec<String> {
    let mut targets = BTreeSet::new();
    for record in records {
        targets.extend(record.labels.keys().cloned());
    }
    targets.into_iter().collect()
}
pub(crate) fn add_label_count(count: &mut LabelCount, value: Option<i8>) {
    match value {
        Some(1) => count.positive += 1,
        Some(0) => count.negative += 1,
        Some(-1) => count.uncertain += 1,
        _ => count.missing += 1,
    }
}
pub(crate) fn overlap_count(values_by_split: &BTreeMap<String, BTreeSet<String>>) -> usize {
    let entries = values_by_split.iter().collect::<Vec<_>>();
    let mut overlap = BTreeSet::new();
    for left_index in 0..entries.len() {
        for right_index in (left_index + 1)..entries.len() {
            overlap.extend(
                entries[left_index]
                    .1
                    .intersection(entries[right_index].1)
                    .cloned(),
            );
        }
    }
    overlap.len()
}

pub(crate) fn stable_bucket(value: &str, seed: u64) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(seed.to_le_bytes());
    hasher.update(value.as_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes([
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
    ])
}
pub(crate) fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), CxrError> {
    let writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(writer, value)?;
    Ok(())
}

pub(crate) fn resolve_cache_path(cache_dir: &Path, path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() || path.starts_with(cache_dir) {
        path
    } else {
        cache_dir.join(path)
    }
}
pub(crate) fn hash_file(path: &Path) -> Result<String, CxrError> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 64];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub(crate) fn directory_size(path: &Path) -> Result<u64, CxrError> {
    let mut total = 0u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            total += directory_size(&path)?;
        } else {
            total += metadata.len();
        }
    }
    Ok(total)
}
