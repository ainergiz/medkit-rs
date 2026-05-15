use criterion::{black_box, criterion_group, criterion_main, Criterion};
use medkit_bench::{bench_cache, BenchConfig};
use medkit_benchmarks::fixtures::{
    build_cached_fixture, temp_fixture_root, SyntheticFixtureConfig,
};
use medkit_sampler::load_cached_cases;

fn cache_load_and_extract(c: &mut Criterion) {
    let mut config = SyntheticFixtureConfig::new(temp_fixture_root("criterion-cache"));
    config.cases = 2;
    config.shape = [48, 48, 48];
    config.cache_shape = [48, 48, 48];
    let fixture = build_cached_fixture(&config).expect("cached benchmark fixture");
    let cache_dir = fixture.fixture.cache_dir.clone();

    c.bench_function("cache/load_cached_cases/2x48cubed", |b| {
        b.iter(|| load_cached_cases(black_box(&cache_dir)).expect("load cached cases"))
    });

    c.bench_function("cache/bench_cache/patch24/workers2", |b| {
        b.iter(|| {
            bench_cache(black_box(&BenchConfig {
                cache_dir: cache_dir.clone(),
                patch_size: [24, 24, 24],
                workers: 2,
                samples: 128,
            }))
            .expect("bench cache")
        })
    });
}

criterion_group!(benches, cache_load_and_extract);
criterion_main!(benches);
