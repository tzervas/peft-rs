//! Tiny multi-layer MLP with `LoRA` inject + candle-nn `AdamW` (PR-041).
//!
//! Demonstrates:
//! - `get_peft_model` wrapping named `Linear` modules
//! - Real stacked forward
//! - Adapter params change under `AdamW` while base weights stay frozen

#![allow(
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::items_after_statements
)]
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{linear_no_bias, AdamW, Linear, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use peft_rs::{get_peft_model, LoraConfig};

fn main() -> Result<()> {
    println!("=== LoRA inject + AdamW (tiny MLP) ===\n");
    let device = Device::Cpu;

    // Base host weights (copied out of host VarMap → not trained)
    let host_vm = VarMap::new();
    let host_vb = VarBuilder::from_varmap(&host_vm, DType::F32, &device);
    let h0 = linear_no_bias(16, 16, host_vb.pp("fc1"))?;
    let h1 = linear_no_bias(16, 16, host_vb.pp("fc2"))?;
    let base_modules = vec![
        ("mlp.fc1".into(), Linear::new(h0.weight().copy()?, None)),
        ("mlp.fc2".into(), Linear::new(h1.weight().copy()?, None)),
    ];

    let adapter_vm = VarMap::new();
    let adapter_vb = VarBuilder::from_varmap(&adapter_vm, DType::F32, &device);
    let config = LoraConfig {
        r: 4,
        alpha: 8,
        dropout: 0.0,
        ..Default::default()
    };
    let model = get_peft_model(base_modules, "mlp.*", config, "default", adapter_vb)?;
    println!(
        "Injected {} modules, {} adapter params",
        model.num_modules(),
        model.num_adapter_parameters()
    );

    let mut opt = AdamW::new(
        adapter_vm.all_vars(),
        ParamsAdamW {
            lr: 1e-2,
            ..Default::default()
        },
    )?;

    let x = Tensor::randn(0f32, 1f32, (8, 16), &device)?;
    let mut last_loss = 0.0f32;
    for step in 0..20 {
        let y = model.forward(&x)?;
        let loss = y.sqr()?.mean_all()?;
        last_loss = loss.to_scalar::<f32>()?;
        opt.backward_step(&loss)?;
        if step % 5 == 0 {
            println!("step {step:02}  loss={last_loss:.6}");
        }
    }
    println!("\nFinal loss={last_loss:.6}");
    println!("Base weights were not in the optimizer (adapter-only training).");
    Ok(())
}
