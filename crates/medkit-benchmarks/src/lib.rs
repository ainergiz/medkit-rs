#![forbid(unsafe_code)]

//! Benchmark fixtures, microbenchmark helpers, and CLI macrobenchmark runners.

pub mod fixtures;
pub mod macrobench;

/// Result type used by the benchmark harness.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
