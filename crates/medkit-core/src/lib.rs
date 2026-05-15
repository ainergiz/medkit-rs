#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Core spatial image contracts for `medkit-rs`.
//!
//! This crate intentionally does not load image pixels. It defines the shared
//! types that IO, transforms, validators, samplers, caches, and bindings can
//! agree on before an image is materialized as a training tensor.

mod axis;
mod dtype;
mod error;
mod geometry;
mod image_spec;
mod metadata;
mod shape;
mod validation;

pub use axis::{Axis, AxisKind};
pub use dtype::DType;
pub use error::{MedkitCoreError, Result};
pub use geometry::{CoordinateSystem, SpatialGeometry};
pub use image_spec::{ImageSpec, ImageSpecBuilder};
pub use metadata::{ImageModality, Provenance, SourceKind, SourceRef};
pub use shape::Shape;
pub use validation::{
    GeometryCompatibility, GeometryCompatibilityReport, GeometryMismatch, GeometryTolerance,
};
