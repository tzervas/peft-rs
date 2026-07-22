//! Criterion benches for PEFT adapters (PEFT-P2-02).
//!
//! Non-empty `LoRA` forward / merge wall-time measurements on CPU F32.
//! Record numbers into METRICS.md after a full (non-`--quick`) run.
//!
//! ```bash
//! cargo bench --bench adapters
//! cargo bench --bench adapters -- --quick
//! ```

#![allow(missing_docs)]

use candle_core::{Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use peft_rs::{Adapter, LoraConfig, LoraLayer, Mergeable};

fn make_lora(in_f: usize, out_f: usize, r: usize, device: &Device) -> LoraLayer {
    let cfg = LoraConfig {
        r,
        alpha: 2 * r,
        dropout: 0.0,
        ..Default::default()
    };
    // Non-zero A/B so matmuls do real work (zeros-B would under-measure).
    let a = Tensor::randn(0f32, 0.02, (r, in_f), device).expect("A");
    let b = Tensor::randn(0f32, 0.02, (out_f, r), device).expect("B");
    LoraLayer::from_weights(a, b, cfg).expect("lora")
}

fn benchmark_lora_forward(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("lora_forward");

    // (batch, seq, hidden, rank)
    let shapes = [
        (1usize, 128usize, 768usize, 8usize),
        (4, 128, 768, 8),
        (1, 128, 768, 64),
    ];

    for &(batch, seq, hidden, r) in &shapes {
        let layer = make_lora(hidden, hidden, r, &device);
        let input = Tensor::randn(0f32, 1f32, (batch, seq, hidden), &device).expect("input");
        let id = format!("b{batch}_s{seq}_h{hidden}_r{r}");
        group.bench_with_input(BenchmarkId::new("fwd", &id), &input, |bencher, x| {
            bencher.iter(|| {
                let y = layer.forward(black_box(x), None).expect("fwd");
                black_box(y)
            });
        });
    }
    group.finish();
}

fn benchmark_lora_merge(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("lora_merge");
    for &(hidden, r) in &[(768usize, 8usize), (768, 64), (2048, 8)] {
        let layer = make_lora(hidden, hidden, r, &device);
        let base = Tensor::randn(0f32, 0.02, (hidden, hidden), &device).expect("W");
        let id = format!("h{hidden}_r{r}");
        group.bench_with_input(BenchmarkId::new("merge", &id), &base, |bencher, w| {
            bencher.iter(|| {
                let m = layer.merge(black_box(w)).expect("merge");
                black_box(m)
            });
        });
    }
    group.finish();
}

fn benchmark_lora_fwd_with_base(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("lora_forward_with_base");
    let hidden = 768usize;
    let r = 8usize;
    let layer = make_lora(hidden, hidden, r, &device);
    let input = Tensor::randn(0f32, 1f32, (4, 128, hidden), &device).expect("x");
    let base_out = Tensor::randn(0f32, 1f32, (4, 128, hidden), &device).expect("base");
    group.bench_function("b4_s128_h768_r8", |bencher| {
        bencher.iter(|| {
            let y = layer
                .forward(black_box(&input), Some(black_box(&base_out)))
                .expect("fwd");
            black_box(y)
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    benchmark_lora_forward,
    benchmark_lora_merge,
    benchmark_lora_fwd_with_base
);
criterion_main!(benches);
