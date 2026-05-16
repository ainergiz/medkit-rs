use crate::{
    geometry::{mat_mul, mat_vec_mul},
    Interpolation, Result, Volume3D, VolumeGeometry,
};

const EDGE_EPSILON: f64 = 1e-9;

pub(crate) fn resample_f32(
    volume: &Volume3D<f32>,
    source: &VolumeGeometry,
    target: &VolumeGeometry,
    interpolation: Interpolation,
) -> Result<Volume3D<f32>> {
    let transform = IndexTransform::new(source, target)?;
    let mut data = Vec::with_capacity(target.shape[0] * target.shape[1] * target.shape[2]);
    for z in 0..target.shape[2] {
        for y in 0..target.shape[1] {
            for x in 0..target.shape[0] {
                let source_index = transform.source_index(x, y, z);
                let value = match interpolation {
                    Interpolation::Nearest => sample_nearest_f32(volume, source_index, 0.0),
                    Interpolation::Linear => sample_linear_f32(volume, source_index, 0.0),
                };
                data.push(value);
            }
        }
    }
    Volume3D::new(target.shape, data)
}

pub(crate) fn resample_u16(
    volume: &Volume3D<u16>,
    source: &VolumeGeometry,
    target: &VolumeGeometry,
    interpolation: Interpolation,
) -> Result<Volume3D<u16>> {
    let transform = IndexTransform::new(source, target)?;
    let mut data = Vec::with_capacity(target.shape[0] * target.shape[1] * target.shape[2]);
    for z in 0..target.shape[2] {
        for y in 0..target.shape[1] {
            for x in 0..target.shape[0] {
                let source_index = transform.source_index(x, y, z);
                let value = match interpolation {
                    Interpolation::Nearest => sample_nearest_u16(volume, source_index, 0),
                    Interpolation::Linear => sample_linear_u16(volume, source_index, 0),
                };
                data.push(value);
            }
        }
    }
    Volume3D::new(target.shape, data)
}

#[derive(Debug, Clone, Copy)]
struct IndexTransform {
    offset: [f64; 3],
    step: [[f64; 3]; 3],
}

impl IndexTransform {
    fn new(source: &VolumeGeometry, target: &VolumeGeometry) -> Result<Self> {
        let inverse_source = source.inverse_voxel_matrix()?;
        let step = mat_mul(inverse_source, target.voxel_matrix());
        let delta_origin = [
            target.origin[0] - source.origin[0],
            target.origin[1] - source.origin[1],
            target.origin[2] - source.origin[2],
        ];
        Ok(Self {
            offset: mat_vec_mul(inverse_source, delta_origin),
            step,
        })
    }

    fn source_index(&self, x: usize, y: usize, z: usize) -> [f64; 3] {
        [
            self.offset[0]
                + self.step[0][0] * x as f64
                + self.step[0][1] * y as f64
                + self.step[0][2] * z as f64,
            self.offset[1]
                + self.step[1][0] * x as f64
                + self.step[1][1] * y as f64
                + self.step[1][2] * z as f64,
            self.offset[2]
                + self.step[2][0] * x as f64
                + self.step[2][1] * y as f64
                + self.step[2][2] * z as f64,
        ]
    }
}

fn sample_nearest_f32(volume: &Volume3D<f32>, index: [f64; 3], fill: f32) -> f32 {
    let Some([x, y, z]) = nearest_index(index, volume.shape) else {
        return fill;
    };
    *volume.get(x, y, z)
}

fn sample_nearest_u16(volume: &Volume3D<u16>, index: [f64; 3], fill: u16) -> u16 {
    let Some([x, y, z]) = nearest_index(index, volume.shape) else {
        return fill;
    };
    *volume.get(x, y, z)
}

fn sample_linear_f32(volume: &Volume3D<f32>, index: [f64; 3], fill: f32) -> f32 {
    let Some(index) = normalized_index(index, volume.shape) else {
        return fill;
    };
    let (base, weight) = linear_base_and_weight(index);
    let mut out = 0.0_f64;
    for dz in 0..=1 {
        for dy in 0..=1 {
            for dx in 0..=1 {
                let factor = axis_weight(dx, weight[0])
                    * axis_weight(dy, weight[1])
                    * axis_weight(dz, weight[2]);
                let value =
                    value_or_fill_f32(volume, [base[0] + dx, base[1] + dy, base[2] + dz], fill)
                        as f64;
                out += value * factor;
            }
        }
    }
    out as f32
}

fn sample_linear_u16(volume: &Volume3D<u16>, index: [f64; 3], fill: u16) -> u16 {
    let Some(index) = normalized_index(index, volume.shape) else {
        return fill;
    };
    let (base, weight) = linear_base_and_weight(index);
    let mut out = 0.0_f64;
    for dz in 0..=1 {
        for dy in 0..=1 {
            for dx in 0..=1 {
                let factor = axis_weight(dx, weight[0])
                    * axis_weight(dy, weight[1])
                    * axis_weight(dz, weight[2]);
                let value =
                    value_or_fill_u16(volume, [base[0] + dx, base[1] + dy, base[2] + dz], fill)
                        as f64;
                out += value * factor;
            }
        }
    }
    out.round().clamp(0.0, u16::MAX as f64) as u16
}

fn nearest_index(index: [f64; 3], shape: [usize; 3]) -> Option<[usize; 3]> {
    let index = normalized_index(index, shape)?;
    let mut out = [0_usize; 3];
    for axis in 0..3 {
        out[axis] = index[axis].round() as usize;
    }
    Some(out)
}

fn normalized_index(mut index: [f64; 3], shape: [usize; 3]) -> Option<[f64; 3]> {
    for axis in 0..3 {
        if !index[axis].is_finite() {
            return None;
        }
        let max = (shape[axis] - 1) as f64;
        if index[axis].abs() <= EDGE_EPSILON {
            index[axis] = 0.0;
        } else if (index[axis] - max).abs() <= EDGE_EPSILON {
            index[axis] = max;
        }
        if index[axis] < 0.0 || index[axis] > max {
            return None;
        }
    }
    Some(index)
}

fn linear_base_and_weight(index: [f64; 3]) -> ([usize; 3], [f64; 3]) {
    let base = [
        index[0].floor() as usize,
        index[1].floor() as usize,
        index[2].floor() as usize,
    ];
    let weight = [
        index[0] - base[0] as f64,
        index[1] - base[1] as f64,
        index[2] - base[2] as f64,
    ];
    (base, weight)
}

fn axis_weight(offset: usize, weight: f64) -> f64 {
    if offset == 0 {
        1.0 - weight
    } else {
        weight
    }
}

fn value_or_fill_f32(volume: &Volume3D<f32>, index: [usize; 3], fill: f32) -> f32 {
    if index
        .iter()
        .zip(volume.shape)
        .any(|(index, shape)| *index >= shape)
    {
        return fill;
    }
    *volume.get(index[0], index[1], index[2])
}

fn value_or_fill_u16(volume: &Volume3D<u16>, index: [usize; 3], fill: u16) -> u16 {
    if index
        .iter()
        .zip(volume.shape)
        .any(|(index, shape)| *index >= shape)
    {
        return fill;
    }
    *volume.get(index[0], index[1], index[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_volume() -> Volume3D<f32> {
        Volume3D::new(
            [2, 2, 2],
            vec![0.0, 1.0, 10.0, 11.0, 100.0, 101.0, 110.0, 111.0],
        )
        .unwrap()
    }

    #[test]
    fn nearest_sampling_clamps_tiny_edge_drift_and_rejects_invalid_coordinates() {
        let volume = f32_volume();

        assert_eq!(
            sample_nearest_f32(&volume, [-EDGE_EPSILON / 2.0, 0.0, 0.0], -1.0),
            0.0
        );
        assert_eq!(
            sample_nearest_f32(&volume, [1.0 + EDGE_EPSILON / 2.0, 0.0, 0.0], -1.0),
            1.0
        );
        assert_eq!(
            sample_nearest_f32(&volume, [-EDGE_EPSILON * 2.0, 0.0, 0.0], -1.0),
            -1.0
        );
        assert_eq!(
            sample_nearest_f32(&volume, [f64::INFINITY, 0.0, 0.0], -1.0),
            -1.0
        );
    }

    #[test]
    fn linear_sampling_interpolates_rounds_u16_and_uses_fill_for_missing_neighbors() {
        let image = Volume3D::new([2, 1, 1], vec![0.0, 4.0]).unwrap();
        let label = Volume3D::new([2, 1, 1], vec![0_u16, 3]).unwrap();

        assert_eq!(sample_linear_f32(&image, [0.25, 0.0, 0.0], -9.0), 1.0);
        assert_eq!(sample_linear_u16(&label, [0.5, 0.0, 0.0], 0), 2);
        assert_eq!(sample_linear_u16(&label, [2.0, 0.0, 0.0], 7), 7);
        assert_eq!(sample_linear_f32(&image, [2.0, 0.0, 0.0], -9.0), -9.0);
        assert_eq!(value_or_fill_f32(&image, [2, 0, 0], -9.0), -9.0);
        assert_eq!(value_or_fill_u16(&label, [0, 1, 0], 7), 7);
    }

    #[test]
    fn u16_nearest_sampling_uses_fill_for_out_of_bounds_indices() {
        let label = Volume3D::new([2, 1, 1], vec![1_u16, 2]).unwrap();

        assert_eq!(sample_nearest_u16(&label, [f64::NAN, 0.0, 0.0], 9), 9);
        assert_eq!(sample_nearest_u16(&label, [2.0, 0.0, 0.0], 9), 9);
    }

    #[test]
    fn resample_uses_target_origin_offset_for_fill_and_source_coordinates() {
        let volume = Volume3D::new([2, 1, 1], vec![5.0, 9.0]).unwrap();
        let source = VolumeGeometry::identity([2, 1, 1], [1.0, 1.0, 1.0]).unwrap();
        let target = VolumeGeometry::new(
            [3, 1, 1],
            [1.0, 1.0, 1.0],
            [-1.0, 0.0, 0.0],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        .unwrap();

        let resampled = resample_f32(&volume, &source, &target, Interpolation::Nearest).unwrap();

        assert_eq!(resampled.shape, [3, 1, 1]);
        assert_eq!(resampled.data, vec![0.0, 5.0, 9.0]);
    }
}
