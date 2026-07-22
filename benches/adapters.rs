//! Benchmarks for PEFT adapters
//!
//! **Stub (PEFT-P0-05 / PEFT-P2-02):** empty criterion group on purpose.
//! Real wall-time / RSS numbers belong in METRICS.md once instrumented.
//! Do not interpret a successful `cargo bench` as performance parity.

#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_adapters(c: &mut Criterion) {
    // Intentionally empty harness — adapters exist, measured benches do not yet.
    let group = c.benchmark_group("adapters");
    group.finish();
}

criterion_group!(benches, benchmark_adapters);
criterion_main!(benches);
