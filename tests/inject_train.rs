//! PR-041: Linear+LoRA inject path with candle-nn AdamW adapter updates.

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear_no_bias, AdamW, Linear, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use peft_rs::{get_peft_model, LinearWithLora, LoraConfig, PeftLinearModel};

/// Snapshot adapter weight sum for change detection.
fn adapter_weight_l1(model: &PeftLinearModel) -> candle_core::Result<f32> {
    let mut total = 0.0f32;
    for m in model.iter() {
        let (a, b) = m.lora().weights();
        total += a.abs()?.sum_all()?.to_scalar::<f32>()?;
        total += b.abs()?.sum_all()?.to_scalar::<f32>()?;
    }
    Ok(total)
}

fn base_weight_l1(model: &PeftLinearModel) -> candle_core::Result<f32> {
    let mut total = 0.0f32;
    for m in model.iter() {
        total += m.base().weight().abs()?.sum_all()?.to_scalar::<f32>()?;
    }
    Ok(total)
}

#[test]
fn linear_with_lora_forward_adds_residual() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let base_w = Tensor::eye(4, DType::F32, &device)?;
    let base = Linear::new(base_w, None);
    let cfg = LoraConfig {
        r: 2,
        alpha: 4, // scaling = 2
        ..Default::default()
    };
    // Non-zero B so residual is non-zero
    let a = Tensor::ones((2, 4), DType::F32, &device)?;
    let b = Tensor::ones((4, 2), DType::F32, &device)?;
    let lora = peft_rs::LoraLayer::from_weights(a, b, cfg)?;
    let layer = LinearWithLora::from_parts("fc", base, lora)?;

    let x = Tensor::ones((1, 1, 4), DType::F32, &device)?;
    // base: identity → ones
    // lora: x@A.T = [4,4] ones-row @ ones → large; then @ B.T * 2
    let y = layer.forward(&x)?;
    assert_eq!(y.dims(), &[1, 1, 4]);
    // Residual path must move output away from pure identity
    let base_only = layer.base().forward(&x)?;
    let diff = y
        .sub(&base_only)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;
    assert!(diff > 1e-3, "expected non-zero LoRA residual, got {diff}");
    Ok(())
}

#[test]
fn get_peft_model_mlp_adamw_updates_adapters_not_base() -> anyhow::Result<()> {
    let device = Device::Cpu;

    // Host "model" linears constructed once; weights copied out so they are not Vars in adapter map.
    let host_vm = VarMap::new();
    let host_vb = VarBuilder::from_varmap(&host_vm, DType::F32, &device);
    let h0 = linear_no_bias(8, 8, host_vb.pp("fc1"))?;
    let h1 = linear_no_bias(8, 8, host_vb.pp("fc2"))?;
    let base_modules = vec![
        (
            "mlp.fc1".to_string(),
            Linear::new(h0.weight().copy()?, None),
        ),
        (
            "mlp.fc2".to_string(),
            Linear::new(h1.weight().copy()?, None),
        ),
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
    assert_eq!(model.num_modules(), 2);

    let before_adapter = adapter_weight_l1(&model)?;
    let before_base = base_weight_l1(&model)?;

    let vars = adapter_vm.all_vars();
    assert!(
        !vars.is_empty(),
        "adapter VarMap must own trainable LoRA params"
    );
    let mut opt = AdamW::new(
        vars,
        ParamsAdamW {
            lr: 1e-2,
            ..Default::default()
        },
    )?;

    // Toy regression: push outputs toward zeros
    let x = Tensor::randn(0f32, 1f32, (4, 8), &device)?;
    for _ in 0..15 {
        let y = model.forward(&x)?;
        let loss = y.sqr()?.mean_all()?;
        opt.backward_step(&loss)?;
    }

    let after_adapter = adapter_weight_l1(&model)?;
    let after_base = base_weight_l1(&model)?;

    let adapter_delta = (after_adapter - before_adapter).abs();
    assert!(
        adapter_delta > 1e-4,
        "adapter weights should change under AdamW (delta={adapter_delta}, before={before_adapter}, after={after_adapter})"
    );
    let base_delta = (after_base - before_base).abs();
    assert!(
        base_delta < 1e-5,
        "base Linear weights must stay frozen (delta={base_delta})"
    );

    Ok(())
}

#[test]
fn peft_linear_model_module_forward() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
    let base = Linear::new(Tensor::randn(0f32, 0.1, (16, 16), &device)?, None);
    let modules = vec![("layer.0".into(), base)];
    let model = PeftLinearModel::from_linears(
        modules,
        LoraConfig {
            r: 2,
            alpha: 4,
            ..Default::default()
        },
        "default",
        vb,
    )?;
    let x = Tensor::zeros(&[2, 3, 16], DType::F32, &device)?;
    let y = model.forward_module("layer.0", &x)?;
    assert_eq!(y.dims(), &[2, 3, 16]);
    Ok(())
}
