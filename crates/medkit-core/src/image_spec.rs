use crate::{Axis, DType, ImageModality, MedkitCoreError, Provenance, Result, SpatialGeometry};

/// Complete non-pixel description of an image object.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageSpec {
    id: String,
    dtype: DType,
    axes: Vec<Axis>,
    geometry: SpatialGeometry,
    modality: ImageModality,
    provenance: Provenance,
}

impl ImageSpec {
    /// Starts building an image spec.
    pub fn builder(
        id: impl Into<String>,
        dtype: DType,
        geometry: SpatialGeometry,
        modality: ImageModality,
        provenance: Provenance,
    ) -> ImageSpecBuilder {
        ImageSpecBuilder {
            id: id.into(),
            dtype,
            axes: None,
            geometry,
            modality,
            provenance,
        }
    }

    /// Returns the image id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the scalar data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns axes in storage order.
    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }

    /// Returns spatial geometry.
    pub fn geometry(&self) -> &SpatialGeometry {
        &self.geometry
    }

    /// Returns modality.
    pub fn modality(&self) -> &ImageModality {
        &self.modality
    }

    /// Returns provenance.
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
}

/// Builder for [`ImageSpec`].
#[derive(Debug, Clone)]
pub struct ImageSpecBuilder {
    id: String,
    dtype: DType,
    axes: Option<Vec<Axis>>,
    geometry: SpatialGeometry,
    modality: ImageModality,
    provenance: Provenance,
}

impl ImageSpecBuilder {
    /// Sets storage axes.
    pub fn axes(mut self, axes: impl Into<Vec<Axis>>) -> Self {
        self.axes = Some(axes.into());
        self
    }

    /// Builds the image spec.
    pub fn build(self) -> Result<ImageSpec> {
        if self.id.is_empty() {
            return Err(MedkitCoreError::EmptyImageId);
        }
        let axes = match self.axes {
            Some(axes) => axes,
            None => default_axes(self.geometry.shape().rank()),
        };
        let rank = self.geometry.shape().rank();
        if axes.len() != rank {
            return Err(MedkitCoreError::AxisRankMismatch {
                axes: axes.len(),
                rank,
            });
        }
        Ok(ImageSpec {
            id: self.id,
            dtype: self.dtype,
            axes,
            geometry: self.geometry,
            modality: self.modality,
            provenance: self.provenance,
        })
    }
}

fn default_axes(rank: usize) -> Vec<Axis> {
    let mut axes = Vec::with_capacity(rank);
    axes.push(Axis::x());
    if rank >= 2 {
        axes.push(Axis::y());
    }
    if rank >= 3 {
        axes.push(Axis::z());
    }
    for index in 3..rank {
        axes.push(
            Axis::new(
                crate::AxisKind::Other(format!("dim{index}")),
                format!("dim{index}"),
            )
            .expect("generated axis labels are non-empty"),
        );
    }
    axes
}
