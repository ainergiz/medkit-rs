use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use flate2::{write::GzEncoder, Compression};

const HEADER_LEN: usize = 348;

#[derive(Debug, Clone)]
pub struct NiftiFixture {
    bytes: [u8; HEADER_LEN],
}

impl NiftiFixture {
    pub fn new(dims: &[i16], datatype: i16, pixdim: &[f32]) -> Self {
        let mut fixture = Self {
            bytes: [0; HEADER_LEN],
        };
        fixture.put_i32(0, 348);
        fixture.put_i16(40, i16::try_from(dims.len()).unwrap());
        for (index, dim) in dims.iter().enumerate() {
            fixture.put_i16(42 + index * 2, *dim);
        }
        fixture.put_i16(70, datatype);
        fixture.put_i16(72, bitpix_for(datatype));
        fixture.put_f32(76, 1.0);
        for (index, spacing) in pixdim.iter().enumerate() {
            fixture.put_f32(80 + index * 4, *spacing);
        }
        fixture.put_f32(108, 352.0);
        fixture.bytes[344..348].copy_from_slice(b"n+1\0");
        fixture
    }

    pub fn write_nii(&self, path: &Path) {
        let mut bytes = self.bytes.to_vec();
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        fs::write(path, bytes).unwrap();
    }

    pub fn write_hdr_gz(&self, path: &Path) {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(&self.bytes).unwrap();
        fs::write(path, encoder.finish().unwrap()).unwrap();
    }

    fn put_i32(&mut self, offset: usize, value: i32) {
        self.bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_i16(&mut self, offset: usize, value: i16) {
        self.bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_f32(&mut self, offset: usize, value: f32) {
        self.bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
}

pub fn temp_case_dir(case: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "medkit-dataset-{case}-{}-{nanos}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

pub fn create_dataset_dirs(root: &Path) -> (PathBuf, PathBuf) {
    let images = root.join("imagesTr");
    let labels = root.join("labelsTr");
    fs::create_dir_all(&images).unwrap();
    fs::create_dir_all(&labels).unwrap();
    (images, labels)
}

pub fn write_case(
    images: &Path,
    labels: &Path,
    case_id: &str,
    image: Option<NiftiFixture>,
    label: Option<NiftiFixture>,
) {
    if let Some(image) = image {
        image.write_nii(&images.join(format!("{case_id}.nii")));
    }
    if let Some(label) = label {
        label.write_nii(&labels.join(format!("{case_id}.nii")));
    }
}

fn bitpix_for(datatype: i16) -> i16 {
    match datatype {
        1 => 1,
        2 | 256 => 8,
        4 | 512 => 16,
        8 | 16 | 768 => 32,
        64 => 64,
        _ => 0,
    }
}
