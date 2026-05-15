#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Lazy transform plans and deterministic 3D preprocessing kernels.

mod error;
mod geometry;
mod graph;
mod plan;
mod resample;
mod volume;

pub use error::{Result, TransformError};
pub use geometry::VolumeGeometry;
pub use graph::{Interpolation, LazyTransformGraph, TransformOp};
pub use plan::{PreparedPair, TransformPlan};
pub use volume::{BoundingBox3, Volume3D};
