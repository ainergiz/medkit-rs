use criterion::{black_box, criterion_group, criterion_main, Criterion};
use medkit_benchmarks::fixtures::{synthetic_volume_pair, transform_plan, SyntheticFixtureConfig};

fn transform_plan_apply_pair(c: &mut Criterion) {
    let mut config = SyntheticFixtureConfig::new("criterion-transform");
    config.shape = [64, 64, 64];
    config.cache_shape = [64, 64, 64];
    config.resample_spacing = [1.0, 1.0, 1.0];
    let plan = transform_plan(&config).expect("benchmark transform plan should parse");
    let (image, label, geometry) =
        synthetic_volume_pair(config.shape, [1.0, 1.0, 1.0], 0).expect("synthetic pair");

    c.bench_function("transform/apply_pair_with_geometry/64cubed", |b| {
        b.iter(|| {
            plan.apply_pair_with_geometry(
                black_box(image.clone()),
                black_box(label.clone()),
                black_box(geometry),
            )
            .expect("transform should succeed")
        })
    });
}

criterion_group!(benches, transform_plan_apply_pair);
criterion_main!(benches);
