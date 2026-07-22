//! PR-040: HuggingFace adapter_config + LoRA key round-trip fixtures.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use peft_rs::{
    extract_lora_ab, load_pretrained_hf, pack_lora_state_dict, save_pretrained_hf, Adapter,
    HfLoraConfig, LoraConfig, LoraKeyStyle, LoraLayer, SaveLoad, ADAPTER_CONFIG_FILENAME,
    ADAPTER_WEIGHTS_FILENAME, DEFAULT_ADAPTER_NAME,
};
use tempfile::TempDir;

#[test]
fn hf_adapter_config_fields_roundtrip() -> anyhow::Result<()> {
    let cfg = HfLoraConfig {
        peft_type: "LORA".into(),
        r: 8,
        lora_alpha: 16,
        target_modules: vec!["q_proj".into(), "v_proj".into()],
        base_model_name_or_path: Some("org/model".into()),
        task_type: Some("CAUSAL_LM".into()),
        lora_dropout: 0.1,
        modules_to_save: Some(vec!["embed_tokens".into()]),
        ..Default::default()
    };
    let json = serde_json::to_string_pretty(&cfg)?;
    for needle in [
        "peft_type",
        "lora_alpha",
        "target_modules",
        "base_model_name_or_path",
        "task_type",
        "modules_to_save",
    ] {
        assert!(json.contains(needle), "missing {needle} in {json}");
    }
    let back: HfLoraConfig = serde_json::from_str(&json)?;
    assert_eq!(back, cfg);
    let native = back.to_lora_config();
    assert_eq!(native.r, 8);
    assert_eq!(native.alpha, 16);
    Ok(())
}

#[test]
fn hf_weight_keys_module_prefix_roundtrip() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let lora_cfg = LoraConfig {
        r: 4,
        alpha: 8,
        target_modules: vec!["q_proj".into()],
        ..Default::default()
    };
    let layer = LoraLayer::new_with_zeros(32, 32, lora_cfg.clone(), &device)?;
    let hf_cfg = HfLoraConfig::from_lora_config(
        &lora_cfg,
        Some("fixture/base".into()),
        Some("FEATURE_EXTRACTION".into()),
    );

    let dir = TempDir::new()?;
    let prefix = "model.layers.0.self_attn.q_proj";
    save_pretrained_hf(
        &layer,
        &hf_cfg,
        dir.path(),
        &LoraKeyStyle::hf_module(prefix),
    )?;

    let raw = std::fs::read_to_string(dir.path().join(ADAPTER_CONFIG_FILENAME))?;
    assert!(raw.contains("\"peft_type\": \"LORA\""));
    assert!(raw.contains("\"lora_alpha\": 8"));

    let tensors =
        candle_core::safetensors::load(dir.path().join(ADAPTER_WEIGHTS_FILENAME), &device)?;
    assert!(tensors.contains_key(&format!("{prefix}.lora_A.{DEFAULT_ADAPTER_NAME}.weight")));
    assert!(tensors.contains_key(&format!("{prefix}.lora_B.{DEFAULT_ADAPTER_NAME}.weight")));

    let mut loaded = LoraLayer::new_with_zeros(32, 32, lora_cfg, &device)?;
    let loaded_cfg = load_pretrained_hf(&mut loaded, dir.path(), &device, Some(prefix))?;
    assert_eq!(loaded_cfg.base_model_name_or_path.as_deref(), Some("fixture/base"));
    assert_eq!(loaded_cfg.task_type.as_deref(), Some("FEATURE_EXTRACTION"));

    let x = Tensor::randn(0f32, 1f32, (2, 5, 32), &device)?;
    let y0 = layer.forward(&x, None)?;
    let y1 = loaded.forward(&x, None)?;
    let max = (y0 - y1)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    assert!(max < 1e-5, "max diff {max}");
    Ok(())
}

#[test]
fn extract_accepts_base_model_prefix_keys() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let a = Tensor::ones((2, 4), DType::F32, &device)?;
    let b = Tensor::full(2.0f32, (4, 2), &device)?;
    let mut map = HashMap::new();
    map.insert(
        "base_model.model.layers.0.q_proj.lora_A.default.weight".into(),
        a.clone(),
    );
    map.insert(
        "base_model.model.layers.0.q_proj.lora_B.default.weight".into(),
        b.clone(),
    );
    let (a2, b2) = extract_lora_ab(&map, Some("layers.0.q_proj"), Some("default"))?;
    assert_eq!(a2.dims(), a.dims());
    assert_eq!(b2.dims(), b.dims());
    Ok(())
}

#[test]
fn bare_hf_keys_load_into_layer() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let cfg = LoraConfig {
        r: 2,
        alpha: 4,
        ..Default::default()
    };
    let mut layer = LoraLayer::new_with_zeros(4, 4, cfg, &device)?;
    let a = Tensor::full(0.25f32, (2, 4), &device)?;
    let b = Tensor::full(0.5f32, (4, 2), &device)?;
    let sd = pack_lora_state_dict(&a, &b, &LoraKeyStyle::hf_default());
    layer.load_state_dict(sd)?;
    let native = layer.state_dict()?;
    assert!(native.contains_key("lora_a.weight"));
    Ok(())
}
