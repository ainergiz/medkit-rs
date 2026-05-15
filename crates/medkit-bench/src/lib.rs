#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Benchmark cached training data loading and patch extraction throughput.

mod bench;
mod error;

pub use bench::{
    bench_cache, bench_patch_plan, BenchConfig, BenchMetrics, BenchReport, PlanBenchConfig,
    PlanBenchReport,
};
pub use error::{BenchError, Result};
