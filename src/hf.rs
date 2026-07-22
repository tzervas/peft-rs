//! HuggingFace PEFT interop for LoRA adapter config and weight keys.
//!
//! # Config schema (`adapter_config.json`)
//!
//! Python PEFT writes fields such as `peft_type`, `r`, `lora_alpha`,
//! `target_modules`, `base_model_name_or_path`, and `task_type`. This module
//! provides [`HfLoraConfig`] that round-trips those core fields.
//!
//! # Weight key mapping
//!
//! | Style | Example keys |
//! |-------|----------------|
//! | **Native** (legacy peft-rs) | `lora_a.weight`, `lora_b.weight` |
//! | **HF bare** (single module) | `lora_A.default.weight`, `lora_B.default.weight` |
//! | **HF module** | `{module}.lora_A.default.weight`, `{module}.lora_B.default.weight` |
//! | **HF full** | `base_model.model.{module}.lora_A.default.weight` |
//!
//! Loading accepts any of the above (case-sensitive `lora_A` / `lora_B` as in
//! PEFT, plus native lowercase). Saving can emit native or HF-style keys via
//! [`LoraKeyStyle`].
//!
//! # Non-goals
//!
//! - Full multi-adapter PEFT archive layouts beyond the `default` adapter name
//! - Auto-discovery of every PEFT tuner type (LoRA only for HF schema here)

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::adapters::lora::LoraConfig;
use crate::error::{PeftError, Result};
use crate::io::{
    load_adapter_config, save_adapter_config, SaveLoad, ADAPTER_CONFIG_FILENAME,
    ADAPTER_WEIGHTS_FILENAME,
};

/// Default HF PEFT adapter name embedded in weight keys.
pub const DEFAULT_ADAPTER_NAME: &str = "default";

/// PEFT type string written to `adapter_config.json` for LoRA.
pub const PEFT_TYPE_LORA: &str = "LORA";

/// HuggingFace-compatible LoRA `adapter_config.json` schema (core fields).
///
/// Serializes with the field names Python PEFT expects. Extra unknown fields
/// are ignored on load (`deny_unknown_fields` is **not** set) so minor PEFT
/// version differences still parse.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HfLoraConfig {
    /// PEFT tuner type — always `"LORA"` for this struct.
    #[serde(default = "default_peft_type")]
    pub peft_type: String,

    /// LoRA rank.
    pub r: usize,

    /// LoRA alpha (maps to [`LoraConfig::alpha`]).
    pub lora_alpha: usize,

    /// Target module name fragments (e.g. `q_proj`, `v_proj`).
    #[serde(default = "default_hf_target_modules")]
    pub target_modules: Vec<String>,

    /// Optional base model id/path (Hub id or local path).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_model_name_or_path: Option<String>,

    /// Optional task type (e.g. `CAUSAL_LM`, `SEQ_CLS`, `FEATURE_EXTRACTION`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_type: Option<String>,

    /// Dropout on the LoRA path (maps to [`LoraConfig::dropout`]).
    #[serde(default)]
    pub lora_dropout: f64,

    /// Bias handling (`"none"`, `"all"`, `"lora_only"`). peft-rs treats as
    /// documentation only for now (Linear LoRA is bias-free).
    #[serde(default = "default_bias")]
    pub bias: String,

    /// Fan-in/fan-out weight layout flag (HF field; Linear path ignores).
    #[serde(default)]
    pub fan_in_fan_out: bool,

    /// Inference-mode flag from PEFT (documentation / tooling).
    #[serde(default)]
    pub inference_mode: bool,

    /// Rank-stabilized LoRA scaling.
    #[serde(default)]
    pub use_rslora: bool,

    /// DoRA flag (config only here; use [`crate::DoraLayer`] for math).
    #[serde(default)]
    pub use_dora: bool,

    /// Modules to fully fine-tune (HF field).
    ///
    /// **Policy (PEFT-P0-11):** peft-rs does **not** auto-unfreeze or clone
    /// `modules_to_save` into the training graph. Callers that need full-module
    /// fine-tunes should train those candle modules themselves. The field is
    /// preserved on config round-trip for interop.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modules_to_save: Option<Vec<String>>,
}

fn default_peft_type() -> String {
    PEFT_TYPE_LORA.to_string()
}

fn default_hf_target_modules() -> Vec<String> {
    vec!["q_proj".into(), "v_proj".into()]
}

fn default_bias() -> String {
    "none".into()
}

impl Default for HfLoraConfig {
    fn default() -> Self {
        Self {
            peft_type: default_peft_type(),
            r: 8,
            lora_alpha: 16,
            target_modules: default_hf_target_modules(),
            base_model_name_or_path: None,
            task_type: None,
            lora_dropout: 0.0,
            bias: default_bias(),
            fan_in_fan_out: false,
            inference_mode: false,
            use_rslora: false,
            use_dora: false,
            modules_to_save: None,
        }
    }
}

impl HfLoraConfig {
    /// Build from a native [`LoraConfig`], optionally attaching Hub metadata.
    #[must_use]
    pub fn from_lora_config(
        config: &LoraConfig,
        base_model_name_or_path: Option<String>,
        task_type: Option<String>,
    ) -> Self {
        Self {
            peft_type: default_peft_type(),
            r: config.r,
            lora_alpha: config.alpha,
            target_modules: config.target_modules.clone(),
            base_model_name_or_path,
            task_type,
            lora_dropout: config.dropout,
            bias: default_bias(),
            fan_in_fan_out: false,
            inference_mode: false,
            use_rslora: config.use_rslora,
            use_dora: config.use_dora,
            modules_to_save: None,
        }
    }

    /// Convert to native [`LoraConfig`] (drops Hub-only metadata).
    #[must_use]
    pub fn to_lora_config(&self) -> LoraConfig {
        LoraConfig {
            r: self.r,
            alpha: self.lora_alpha,
            dropout: self.lora_dropout,
            target_modules: self.target_modules.clone(),
            use_rslora: self.use_rslora,
            use_dora: self.use_dora,
            ..Default::default()
        }
    }

    /// Validate core fields.
    ///
    /// # Errors
    /// Returns [`PeftError::InvalidConfig`] when rank/alpha are zero or
    /// `peft_type` is not LoRA.
    pub fn validate(&self) -> Result<()> {
        if self.peft_type.to_uppercase() != PEFT_TYPE_LORA {
            return Err(PeftError::InvalidConfig(format!(
                "unsupported peft_type '{}'; only LORA is supported for HfLoraConfig",
                self.peft_type
            )));
        }
        if self.r == 0 {
            return Err(PeftError::InvalidConfig("r must be > 0".into()));
        }
        if self.lora_alpha == 0 {
            return Err(PeftError::InvalidConfig("lora_alpha must be > 0".into()));
        }
        if !(0.0..1.0).contains(&self.lora_dropout) {
            return Err(PeftError::InvalidConfig(
                "lora_dropout must be in [0.0, 1.0)".into(),
            ));
        }
        Ok(())
    }
}

/// Weight key naming style for LoRA tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoraKeyStyle {
    /// peft-rs native: `lora_a.weight` / `lora_b.weight`
    Native,
    /// HF PEFT without module prefix: `lora_A.{adapter}.weight`
    HfBare {
        /// Adapter name (usually `"default"`).
        adapter_name: String,
    },
    /// HF PEFT with module path: `{module}.lora_A.{adapter}.weight`
    HfModule {
        /// Module path prefix (e.g. `model.layers.0.self_attn.q_proj`).
        module_prefix: String,
        /// Adapter name (usually `"default"`).
        adapter_name: String,
    },
}

impl LoraKeyStyle {
    /// HF bare keys with the `default` adapter name.
    #[must_use]
    pub fn hf_default() -> Self {
        Self::HfBare {
            adapter_name: DEFAULT_ADAPTER_NAME.into(),
        }
    }

    /// HF module-prefixed keys with the `default` adapter name.
    #[must_use]
    pub fn hf_module(module_prefix: impl Into<String>) -> Self {
        Self::HfModule {
            module_prefix: module_prefix.into(),
            adapter_name: DEFAULT_ADAPTER_NAME.into(),
        }
    }

    /// Key for the LoRA A weight under this style.
    #[must_use]
    pub fn key_a(&self) -> String {
        match self {
            Self::Native => "lora_a.weight".into(),
            Self::HfBare { adapter_name } => format!("lora_A.{adapter_name}.weight"),
            Self::HfModule {
                module_prefix,
                adapter_name,
            } => format!("{module_prefix}.lora_A.{adapter_name}.weight"),
        }
    }

    /// Key for the LoRA B weight under this style.
    #[must_use]
    pub fn key_b(&self) -> String {
        match self {
            Self::Native => "lora_b.weight".into(),
            Self::HfBare { adapter_name } => format!("lora_B.{adapter_name}.weight"),
            Self::HfModule {
                module_prefix,
                adapter_name,
            } => format!("{module_prefix}.lora_B.{adapter_name}.weight"),
        }
    }
}

/// Build a state dict from A/B weights using the given key style.
#[must_use]
pub fn pack_lora_state_dict(
    lora_a: &Tensor,
    lora_b: &Tensor,
    style: &LoraKeyStyle,
) -> HashMap<String, Tensor> {
    let mut map = HashMap::new();
    map.insert(style.key_a(), lora_a.clone());
    map.insert(style.key_b(), lora_b.clone());
    map
}

/// Extract LoRA A/B tensors from a state dict that may use native or HF keys.
///
/// Accepted patterns (first match wins per matrix):
/// - Native: `lora_a.weight` / `lora_b.weight`
/// - HF bare: `lora_A.{name}.weight` / `lora_B.{name}.weight`
/// - HF short: `lora_A.weight` / `lora_B.weight`
/// - Any key ending with `.lora_A.{name}.weight` / `.lora_B.{name}.weight`
/// - Any key ending with `lora_A.weight` / `lora_B.weight` (suffix)
///
/// When multiple module-prefixed keys are present, pass `module_prefix` to
/// disambiguate; otherwise the first matching A/B pair is used.
///
/// # Errors
/// Returns [`PeftError::WeightLoad`] if A or B cannot be found.
pub fn extract_lora_ab(
    state_dict: &HashMap<String, Tensor>,
    module_prefix: Option<&str>,
    adapter_name: Option<&str>,
) -> Result<(Tensor, Tensor)> {
    let adapter = adapter_name.unwrap_or(DEFAULT_ADAPTER_NAME);

    let a = find_lora_tensor(state_dict, 'A', module_prefix, adapter)?;
    let b = find_lora_tensor(state_dict, 'B', module_prefix, adapter)?;
    Ok((a, b))
}

fn find_lora_tensor(
    state_dict: &HashMap<String, Tensor>,
    which: char,
    module_prefix: Option<&str>,
    adapter: &str,
) -> Result<Tensor> {
    let native = if which == 'A' {
        "lora_a.weight"
    } else {
        "lora_b.weight"
    };
    let upper = if which == 'A' { "lora_A" } else { "lora_B" };

    // 1. Exact native
    if let Some(t) = state_dict.get(native) {
        return Ok(t.clone());
    }

    // 2. Prefer module-scoped HF keys when prefix given
    if let Some(prefix) = module_prefix {
        let candidates = [
            format!("{prefix}.{upper}.{adapter}.weight"),
            format!("{prefix}.{upper}.weight"),
            format!("base_model.model.{prefix}.{upper}.{adapter}.weight"),
            format!("base_model.model.{prefix}.{upper}.weight"),
        ];
        for key in candidates {
            if let Some(t) = state_dict.get(&key) {
                return Ok(t.clone());
            }
        }
        // Suffix search restricted to keys containing the module prefix
        for (k, t) in state_dict {
            if k.contains(prefix)
                && (k.ends_with(&format!("{upper}.{adapter}.weight"))
                    || k.ends_with(&format!("{upper}.weight")))
            {
                return Ok(t.clone());
            }
        }
    }

    // 3. HF bare exact
    let bare = [
        format!("{upper}.{adapter}.weight"),
        format!("{upper}.weight"),
    ];
    for key in bare {
        if let Some(t) = state_dict.get(&key) {
            return Ok(t.clone());
        }
    }

    // 4. Any suffix match (single-module archives)
    let suffix_with_adapter = format!("{upper}.{adapter}.weight");
    let suffix_short = format!("{upper}.weight");
    for (k, t) in state_dict {
        if k == &suffix_with_adapter
            || k.ends_with(&format!(".{suffix_with_adapter}"))
            || k == &suffix_short
            || k.ends_with(&format!(".{suffix_short}"))
        {
            return Ok(t.clone());
        }
    }

    Err(PeftError::WeightLoad(format!(
        "missing LoRA {upper} weight (tried native '{native}', HF '{upper}.{adapter}.weight', and module-prefixed forms)"
    )))
}

/// Rewrite a native single-layer state dict into HF key style.
///
/// # Errors
/// Returns an error if native A/B keys are missing.
pub fn native_state_dict_to_hf(
    native: &HashMap<String, Tensor>,
    style: &LoraKeyStyle,
) -> Result<HashMap<String, Tensor>> {
    let (a, b) = extract_lora_ab(native, None, Some(DEFAULT_ADAPTER_NAME))?;
    Ok(pack_lora_state_dict(&a, &b, style))
}

/// Rewrite an HF (or mixed) state dict into native keys.
///
/// # Errors
/// Returns an error if A/B cannot be extracted.
pub fn hf_state_dict_to_native(
    hf: &HashMap<String, Tensor>,
    module_prefix: Option<&str>,
    adapter_name: Option<&str>,
) -> Result<HashMap<String, Tensor>> {
    let (a, b) = extract_lora_ab(hf, module_prefix, adapter_name)?;
    Ok(pack_lora_state_dict(&a, &b, &LoraKeyStyle::Native))
}

/// Prefix every key in a multi-module HF state dict builder entry.
///
/// Inserts `{module_prefix}.lora_A.{adapter}.weight` style keys.
pub fn insert_module_lora_weights(
    out: &mut HashMap<String, Tensor>,
    module_prefix: &str,
    lora_a: &Tensor,
    lora_b: &Tensor,
    adapter_name: &str,
) {
    let style = LoraKeyStyle::HfModule {
        module_prefix: module_prefix.into(),
        adapter_name: adapter_name.into(),
    };
    out.insert(style.key_a(), lora_a.clone());
    out.insert(style.key_b(), lora_b.clone());
}

/// Save adapter weights using an explicit key style plus HF `adapter_config.json`.
///
/// # Errors
/// Returns I/O or serialization errors.
pub fn save_pretrained_hf<P: AsRef<Path>>(
    adapter: &dyn SaveLoad,
    hf_config: &HfLoraConfig,
    dir: P,
    key_style: &LoraKeyStyle,
) -> Result<()> {
    hf_config.validate()?;
    let dir = dir.as_ref();
    if !dir.exists() {
        fs::create_dir_all(dir)
            .map_err(|e| PeftError::Io(format!("Failed to create directory: {e}")))?;
    }

    let native = adapter.state_dict()?;
    let hf_dict = native_state_dict_to_hf(&native, key_style)?;
    let weights_path = dir.join(ADAPTER_WEIGHTS_FILENAME);
    candle_core::safetensors::save(&hf_dict, &weights_path)
        .map_err(|e| PeftError::Io(format!("Failed to save safetensors: {e}")))?;

    let config_path = dir.join(ADAPTER_CONFIG_FILENAME);
    save_adapter_config(hf_config, &config_path)?;
    Ok(())
}

/// Load HF-style adapter directory into an adapter that understands native keys.
///
/// Weights are converted to native keys before [`SaveLoad::load_state_dict`].
///
/// # Errors
/// Returns I/O, parse, or weight-load errors.
pub fn load_pretrained_hf<P: AsRef<Path>>(
    adapter: &mut dyn SaveLoad,
    dir: P,
    device: &Device,
    module_prefix: Option<&str>,
) -> Result<HfLoraConfig> {
    let dir = dir.as_ref();
    if !dir.exists() {
        return Err(PeftError::Io(format!(
            "Directory does not exist: {}",
            dir.display()
        )));
    }

    let weights_path = dir.join(ADAPTER_WEIGHTS_FILENAME);
    let raw = candle_core::safetensors::load(&weights_path, device)?;
    let native = hf_state_dict_to_native(&raw, module_prefix, Some(DEFAULT_ADAPTER_NAME))?;
    adapter.load_state_dict(native)?;

    let config_path = dir.join(ADAPTER_CONFIG_FILENAME);
    let config: HfLoraConfig = load_adapter_config(&config_path)?;
    config.validate()?;
    Ok(config)
}

/// Filter a multi-module HF state dict down to one module's A/B (native keys).
///
/// # Errors
/// Returns an error if the module's keys are missing.
pub fn slice_module_state_dict(
    full: &HashMap<String, Tensor>,
    module_prefix: &str,
    adapter_name: Option<&str>,
) -> Result<HashMap<String, Tensor>> {
    hf_state_dict_to_native(full, Some(module_prefix), adapter_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::lora::LoraLayer;
    use crate::io::{load_pretrained, save_pretrained};
    use crate::traits::Adapter;
    use candle_core::{DType, Device, Tensor};
    use tempfile::TempDir;

    #[test]
    fn test_hf_config_roundtrip_json() -> Result<()> {
        let cfg = HfLoraConfig {
            peft_type: "LORA".into(),
            r: 16,
            lora_alpha: 32,
            target_modules: vec!["q_proj".into(), "v_proj".into(), "k_proj".into()],
            base_model_name_or_path: Some("meta-llama/Llama-2-7b-hf".into()),
            task_type: Some("CAUSAL_LM".into()),
            lora_dropout: 0.05,
            modules_to_save: Some(vec!["lm_head".into()]),
            ..Default::default()
        };
        let json = serde_json::to_string_pretty(&cfg).map_err(|e| PeftError::Io(e.to_string()))?;
        assert!(json.contains("\"peft_type\": \"LORA\""));
        assert!(json.contains("\"lora_alpha\": 32"));
        assert!(json.contains("\"base_model_name_or_path\""));
        assert!(json.contains("\"task_type\": \"CAUSAL_LM\""));
        assert!(!json.contains("\"alpha\"")); // HF name, not native

        let parsed: HfLoraConfig =
            serde_json::from_str(&json).map_err(|e| PeftError::Io(e.to_string()))?;
        assert_eq!(parsed, cfg);

        let native = parsed.to_lora_config();
        assert_eq!(native.r, 16);
        assert_eq!(native.alpha, 32);
        assert!((native.dropout - 0.05).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn test_hf_config_from_python_like_json() -> Result<()> {
        // Minimal subset a Python peft dump might contain
        let json = r#"{
            "peft_type": "LORA",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "base_model_name_or_path": "facebook/opt-125m",
            "task_type": "CAUSAL_LM",
            "lora_dropout": 0.0,
            "bias": "none",
            "inference_mode": true,
            "fan_in_fan_out": false
        }"#;
        let cfg: HfLoraConfig =
            serde_json::from_str(json).map_err(|e| PeftError::Io(e.to_string()))?;
        cfg.validate()?;
        assert_eq!(cfg.r, 8);
        assert_eq!(cfg.lora_alpha, 16);
        assert_eq!(
            cfg.base_model_name_or_path.as_deref(),
            Some("facebook/opt-125m")
        );
        Ok(())
    }

    #[test]
    fn test_key_styles() {
        assert_eq!(LoraKeyStyle::Native.key_a(), "lora_a.weight");
        assert_eq!(
            LoraKeyStyle::hf_default().key_a(),
            "lora_A.default.weight"
        );
        assert_eq!(
            LoraKeyStyle::hf_module("layers.0.q_proj").key_b(),
            "layers.0.q_proj.lora_B.default.weight"
        );
    }

    #[test]
    fn test_extract_hf_and_native_keys() -> Result<()> {
        let device = Device::Cpu;
        let a = Tensor::zeros((4, 8), DType::F32, &device)?;
        let b = Tensor::zeros((8, 4), DType::F32, &device)?;

        let native = pack_lora_state_dict(&a, &b, &LoraKeyStyle::Native);
        let (a2, b2) = extract_lora_ab(&native, None, None)?;
        assert_eq!(a2.dims(), &[4, 8]);
        assert_eq!(b2.dims(), &[8, 4]);

        let hf = pack_lora_state_dict(&a, &b, &LoraKeyStyle::hf_default());
        let (a3, b3) = extract_lora_ab(&hf, None, Some("default"))?;
        assert_eq!(a3.dims(), a2.dims());
        assert_eq!(b3.dims(), b2.dims());

        let mod_style = LoraKeyStyle::hf_module("model.layers.0.self_attn.q_proj");
        let prefixed = pack_lora_state_dict(&a, &b, &mod_style);
        let (a4, _) = extract_lora_ab(
            &prefixed,
            Some("model.layers.0.self_attn.q_proj"),
            Some("default"),
        )?;
        assert_eq!(a4.dims(), &[4, 8]);
        Ok(())
    }

    #[test]
    fn test_save_load_pretrained_hf_roundtrip() -> Result<()> {
        let device = Device::Cpu;
        let lora_cfg = LoraConfig {
            r: 4,
            alpha: 8,
            target_modules: vec!["q_proj".into()],
            ..Default::default()
        };
        let layer = LoraLayer::new_with_zeros(16, 16, lora_cfg.clone(), &device)?;
        let hf_cfg = HfLoraConfig::from_lora_config(
            &lora_cfg,
            Some("test/base-model".into()),
            Some("FEATURE_EXTRACTION".into()),
        );

        let temp = TempDir::new().map_err(|e| PeftError::Io(e.to_string()))?;
        save_pretrained_hf(
            &layer,
            &hf_cfg,
            temp.path(),
            &LoraKeyStyle::hf_module("encoder.layer.0.attention.q_proj"),
        )?;

        // Config file uses HF field names
        let raw = fs::read_to_string(temp.path().join(ADAPTER_CONFIG_FILENAME))
            .map_err(|e| PeftError::Io(e.to_string()))?;
        assert!(raw.contains("lora_alpha"));
        assert!(raw.contains("peft_type"));

        // Weight file uses HF keys
        let tensors = candle_core::safetensors::load(
            temp.path().join(ADAPTER_WEIGHTS_FILENAME),
            &device,
        )?;
        assert!(tensors.contains_key(
            "encoder.layer.0.attention.q_proj.lora_A.default.weight"
        ));
        assert!(tensors.contains_key(
            "encoder.layer.0.attention.q_proj.lora_B.default.weight"
        ));
        assert!(!tensors.contains_key("lora_a.weight"));

        let mut loaded = LoraLayer::new_with_zeros(16, 16, lora_cfg, &device)?;
        let loaded_cfg = load_pretrained_hf(
            &mut loaded,
            temp.path(),
            &device,
            Some("encoder.layer.0.attention.q_proj"),
        )?;
        assert_eq!(loaded_cfg.r, 4);
        assert_eq!(loaded_cfg.lora_alpha, 8);
        assert_eq!(
            loaded_cfg.base_model_name_or_path.as_deref(),
            Some("test/base-model")
        );

        // Forward parity after round-trip
        let x = Tensor::randn(0f32, 1f32, (1, 2, 16), &device)?;
        let y0 = layer.forward(&x, None)?;
        let y1 = loaded.forward(&x, None)?;
        let max_diff = (y0 - y1)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(max_diff < 1e-5, "max_diff={max_diff}");
        Ok(())
    }

    #[test]
    fn test_native_save_still_works() -> Result<()> {
        // Regression: native save_pretrained path unchanged
        let device = Device::Cpu;
        let cfg = LoraConfig::default();
        let layer = LoraLayer::new_with_zeros(8, 8, cfg.clone(), &device)?;
        let temp = TempDir::new().map_err(|e| PeftError::Io(e.to_string()))?;
        save_pretrained(&layer, &cfg, temp.path())?;
        let mut loaded = LoraLayer::new_with_zeros(8, 8, LoraConfig::default(), &device)?;
        let _: LoraConfig = load_pretrained(&mut loaded, temp.path(), &device)?;
        Ok(())
    }

    #[test]
    fn test_load_state_dict_accepts_hf_keys_on_layer() -> Result<()> {
        let device = Device::Cpu;
        let cfg = LoraConfig {
            r: 4,
            alpha: 8,
            ..Default::default()
        };
        let mut layer = LoraLayer::new_with_zeros(8, 8, cfg, &device)?;
        let a = Tensor::ones((4, 8), DType::F32, &device)?;
        let b = Tensor::full(0.5f32, (8, 4), &device)?;
        let hf = pack_lora_state_dict(&a, &b, &LoraKeyStyle::hf_default());
        layer.load_state_dict(hf)?;
        let sd = layer.state_dict()?;
        assert!(sd.contains_key("lora_a.weight"));
        let sum_a = sd["lora_a.weight"].sum_all()?.to_scalar::<f32>()?;
        assert!((sum_a - 32.0).abs() < 1e-4); // 4*8 ones
        Ok(())
    }
}
