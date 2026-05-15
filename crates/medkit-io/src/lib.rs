#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Medical image metadata readers for `medkit-rs`.
//!
//! The first goal of this crate is fast metadata extraction into
//! `medkit-core` contracts without loading pixel arrays.

mod error;
mod nifti;
mod reader;

pub use error::{MedkitIoError, Result};
pub use nifti::NiftiMetadataReader;
pub use reader::ImageMetadataReader;
