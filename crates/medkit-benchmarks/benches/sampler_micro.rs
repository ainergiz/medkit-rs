use criterion::{black_box, criterion_group, criterion_main, Criterion};
use medkit_benchmarks::fixtures::synthetic_loaded_case;
use medkit_sampler::{extract_patch_pair, plan_batches, PatchRecord, SamplingStrategy};

fn sampler_extract_patch(c: &mut Criterion) {
    let case = synthetic_loaded_case([96, 96, 96]).expect("synthetic loaded case");
    c.bench_function("sampler/extract_patch_pair/32cubed", |b| {
        b.iter(|| {
            extract_patch_pair(
                black_box(&case),
                black_box([32, 32, 32]),
                black_box([32, 32, 32]),
            )
            .expect("patch extraction should succeed")
        })
    });
}

fn sampler_plan_batches(c: &mut Criterion) {
    let records = (0..4096)
        .map(|index| PatchRecord {
            index,
            case_id: format!("case_{:04}", index % 16),
            patch_start: [index % 32, 0, 0],
            patch_size: [32, 32, 32],
            has_foreground: index % 2 == 0,
            strategy: SamplingStrategy::ForegroundBalanced,
            epoch: 0,
            worker: 0,
        })
        .collect::<Vec<_>>();

    c.bench_function("sampler/plan_batches/4096_records", |b| {
        b.iter(|| plan_batches(black_box(records.clone()), black_box(8)).expect("batch planning"))
    });
}

criterion_group!(benches, sampler_extract_patch, sampler_plan_batches);
criterion_main!(benches);
