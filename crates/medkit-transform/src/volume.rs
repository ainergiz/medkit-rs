use serde::{Deserialize, Serialize};

use crate::{Result, TransformError};

/// Dense row-major 3D volume in x, y, z order with x as the fastest axis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Volume3D<T> {
    /// Shape in x, y, z order.
    pub shape: [usize; 3],
    /// Contiguous data with index `x + sx * (y + sy * z)`.
    pub data: Vec<T>,
}

impl<T: Clone> Volume3D<T> {
    /// Creates a volume and validates shape/data length.
    pub fn new(shape: [usize; 3], data: Vec<T>) -> Result<Self> {
        if shape.contains(&0) {
            return Err(TransformError::InvalidSize { size: shape });
        }
        let expected = shape[0] * shape[1] * shape[2];
        if expected != data.len() {
            return Err(TransformError::InvalidVolume {
                shape,
                len: data.len(),
            });
        }
        Ok(Self { shape, data })
    }

    /// Creates a filled volume.
    pub fn filled(shape: [usize; 3], value: T) -> Result<Self> {
        if shape.contains(&0) {
            return Err(TransformError::InvalidSize { size: shape });
        }
        Ok(Self {
            shape,
            data: vec![value; shape[0] * shape[1] * shape[2]],
        })
    }

    /// Returns a flat index.
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        x + self.shape[0] * (y + self.shape[1] * z)
    }

    /// Returns a shared voxel reference.
    pub fn get(&self, x: usize, y: usize, z: usize) -> &T {
        &self.data[self.index(x, y, z)]
    }

    /// Returns a mutable voxel reference.
    pub fn get_mut(&mut self, x: usize, y: usize, z: usize) -> &mut T {
        let index = self.index(x, y, z);
        &mut self.data[index]
    }

    /// Crops the volume to a bounding box.
    pub fn crop(&self, bbox: BoundingBox3) -> Result<Self> {
        let out_shape = bbox.size();
        let mut data = Vec::with_capacity(out_shape[0] * out_shape[1] * out_shape[2]);
        for z in bbox.start[2]..bbox.end[2] {
            for y in bbox.start[1]..bbox.end[1] {
                for x in bbox.start[0]..bbox.end[0] {
                    data.push(self.get(x, y, z).clone());
                }
            }
        }
        Self::new(out_shape, data)
    }

    /// Center pads or crops a volume to `target`.
    pub fn pad_crop_center(&self, target: [usize; 3], fill: T) -> Result<Self> {
        if target.contains(&0) {
            return Err(TransformError::InvalidSize { size: target });
        }
        let mut out = Self::filled(target, fill)?;
        let mut src_start = [0_usize; 3];
        let mut dst_start = [0_usize; 3];
        let mut copy = [0_usize; 3];
        for axis in 0..3 {
            if self.shape[axis] > target[axis] {
                src_start[axis] = (self.shape[axis] - target[axis]) / 2;
                copy[axis] = target[axis];
            } else {
                dst_start[axis] = (target[axis] - self.shape[axis]) / 2;
                copy[axis] = self.shape[axis];
            }
        }
        for z in 0..copy[2] {
            for y in 0..copy[1] {
                for x in 0..copy[0] {
                    let value = self
                        .get(src_start[0] + x, src_start[1] + y, src_start[2] + z)
                        .clone();
                    *out.get_mut(dst_start[0] + x, dst_start[1] + y, dst_start[2] + z) = value;
                }
            }
        }
        Ok(out)
    }
}

/// Half-open 3D bounding box.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundingBox3 {
    /// Inclusive start in x, y, z order.
    pub start: [usize; 3],
    /// Exclusive end in x, y, z order.
    pub end: [usize; 3],
}

impl BoundingBox3 {
    /// Creates a bounding box.
    pub fn new(start: [usize; 3], end: [usize; 3]) -> Self {
        Self { start, end }
    }

    /// Returns the size of the bounding box.
    pub fn size(&self) -> [usize; 3] {
        [
            self.end[0] - self.start[0],
            self.end[1] - self.start[1],
            self.end[2] - self.start[2],
        ]
    }
}
