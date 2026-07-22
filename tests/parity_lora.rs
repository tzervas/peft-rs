//! PR-042: LoRA forward/merge parity against checked-in golden fixtures.
//!
//! Fixtures encode the same math as HuggingFace PEFT Linear LoRA:
//! `y = x @ W.T + (x @ A.T @ B.T) * (alpha / r)`
//! `W_merged = W + (B @ A) * (alpha / r)`
//!
//! Python is **not** required in CI. See `scripts/gen_lora_parity_fixture.py`
//! and `tests/parity/README.md` for offline regeneration / optional peft verify.
//!
//! Tolerances (documented in fixture `meta` and METRICS.md):
//! - `atol = 1e-5`
//! - `rtol = 1e-5`

use std::path::PathBuf;

use candle_core::{DType, Device, Module, Tensor};
use peft_rs::{Adapter, LinearWithLora, LoraConfig, LoraLayer, Mergeable};
use serde::Deserialize;

const ATOL: f32 = 1e-5;
const RTOL: f32 = 1e-5;

#[derive(Debug, Deserialize)]
struct Meta {
    #[allow(dead_code)]
    in_features: usize,
    out_features: usize,
    r: usize,
    alpha: usize,
    batch: usize,
    seq: usize,
    atol: f32,
    rtol: f32,
}

#[derive(Debug, Deserialize)]
struct NamedTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    meta: Meta,
    #[serde(rename = "lora_A")]
    lora_a: NamedTensor,
    #[serde(rename = "lora_B")]
    lora_b: NamedTensor,
    base_weight: NamedTensor,
    input: NamedTensor,
    forward_output: NamedTensor,
    merged_weight: NamedTensor,
    merged_forward_output: NamedTensor,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/parity/fixtures/lora_fwd_merge.json")
}

fn load_fixture() -> anyhow::Result<Fixture> {
    let text = std::fs::read_to_string(fixture_path())?;
    Ok(serde_json::from_str(&text)?)
}

fn tensor_from(named: &NamedTensor, device: &Device) -> candle_core::Result<Tensor> {
    Tensor::from_vec(named.data.clone(), named.shape.as_slice(), device)
}

fn allclose(actual: &Tensor, expected: &Tensor, atol: f32, rtol: f32) -> anyhow::Result<()> {
    assert_eq!(
        actual.dims(),
        expected.dims(),
        "shape mismatch {:?} vs {:?}",
        actual.dims(),
        expected.dims()
    );
    let diff = (actual - expected)?.abs()?;
    let tol = (expected.abs()? * f64::from(rtol))?.broadcast_add(&Tensor::new(atol, actual.device())?)?;
    // element-wise: diff <= atol + rtol * |expected|
    let over = diff.gt(&tol)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
    if over > 0.0 {
        let max_diff = diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        anyhow::bail!("allclose failed: {over} elements over tol (max_diff={max_diff}, atol={atol}, rtol={rtol})");
    }
    Ok(())
}

#[test]
fn parity_lora_forward_matches_golden() -> anyhow::Result<()> {
    let fix = load_fixture()?;
    assert!((fix.meta.atol - ATOL).abs() < 1e-12);
    assert!((fix.meta.rtol - RTOL).abs() < 1e-12);

    let device = Device::Cpu;
    let cfg = LoraConfig {
        r: fix.meta.r,
        alpha: fix.meta.alpha,
        dropout: 0.0,
        ..Default::default()
    };

    let a = tensor_from(&fix.lora_a, &device)?;
    let b = tensor_from(&fix.lora_b, &device)?;
    let w = tensor_from(&fix.base_weight, &device)?;
    let x = tensor_from(&fix.input, &device)?;
    let y_ref = tensor_from(&fix.forward_output, &device)?;

    let lora = LoraLayer::from_weights(a, b, cfg)?;
    let base = candle_nn::Linear::new(w, None);
    let layer = LinearWithLora::from_parts("parity", base, lora)?;

    let y = layer.forward(&x)?;
    assert_eq!(
        y.dims(),
        &[fix.meta.batch, fix.meta.seq, fix.meta.out_features]
    );
    allclose(&y, &y_ref, ATOL, RTOL)?;
    Ok(())
}

#[test]
fn parity_lora_merge_matches_golden() -> anyhow::Result<()> {
    let fix = load_fixture()?;
    let device = Device::Cpu;
    let cfg = LoraConfig {
        r: fix.meta.r,
        alpha: fix.meta.alpha,
        dropout: 0.0,
        ..Default::default()
    };

    let a = tensor_from(&fix.lora_a, &device)?;
    let b = tensor_from(&fix.lora_b, &device)?;
    let w = tensor_from(&fix.base_weight, &device)?;
    let x = tensor_from(&fix.input, &device)?;
    let w_ref = tensor_from(&fix.merged_weight, &device)?;
    let y_ref = tensor_from(&fix.merged_forward_output, &device)?;

    let lora = LoraLayer::from_weights(a, b, cfg)?;
    let merged = lora.merge(&w)?;
    allclose(&merged, &w_ref, ATOL, RTOL)?;

    // Merged linear forward should match residual forward
    let base = candle_nn::Linear::new(w.copy()?, None);
    let layer = LinearWithLora::from_parts("parity", base, lora)?;
    let y_residual = layer.forward(&x)?;
    let merged_linear = candle_nn::Linear::new(merged, None);
    let y_merged = merged_linear.forward(&x)?;
    allclose(&y_residual, &y_merged, ATOL, RTOL)?;
    allclose(&y_merged, &y_ref, ATOL, RTOL)?;
    Ok(())
}

#[test]
fn parity_lora_adapter_only_path() -> anyhow::Result<()> {
    // Adapter-only (no base) matches golden residual component
    let fix = load_fixture()?;
    let device = Device::Cpu;
    let cfg = LoraConfig {
        r: fix.meta.r,
        alpha: fix.meta.alpha,
        ..Default::default()
    };
    let a = tensor_from(&fix.lora_a, &device)?;
    let b = tensor_from(&fix.lora_b, &device)?;
    let w = tensor_from(&fix.base_weight, &device)?;
    let x = tensor_from(&fix.input, &device)?;
    let y_full = tensor_from(&fix.forward_output, &device)?;

    let lora = LoraLayer::from_weights(a, b, cfg)?;
    let base_out = candle_nn::Linear::new(w, None).forward(&x)?;
    let residual = lora.forward(&x, None)?;
    let y = (base_out + residual)?;
    allclose(&y, &y_full, ATOL, RTOL)?;
    Ok(())
}
