//! Model integration for PEFT adapters.
//!
//! # Product path (1.1+)
//!
//! [`LinearWithLora`] wraps a candle [`Linear`] base layer plus a [`LoraLayer`]
//! residual and implements a real forward:
//!
//! ```text
//! y = Linear(x) + LoRA(x) * scaling
//! ```
//!
//! [`PeftLinearModel`] / [`get_peft_model`] build named wrappers from a list of
//! base modules (caller supplies the Linear weights — no full transformers port).
//!
//! # Legacy registry
//!
//! [`PeftModel`] remains a name→adapter map used for multi-adapter switching
//! without owning base weights. Prefer [`PeftLinearModel`] when you need a
//! trainable inject path.
//!
//! # `modules_to_save` policy (PEFT-P0-11)
//!
//! HuggingFace PEFT can fully fine-tune selected modules via `modules_to_save`.
//! peft-rs **does not** implement automatic modules_to_save training:
//! - The field is preserved on [`crate::hf::HfLoraConfig`] for config interop.
//! - Callers that need full-module updates should train those candle modules
//!   themselves (or leave them in the optimizer). Adapter inject only wraps
//!   `target_modules` / named Linear layers you pass in.

use std::collections::HashMap;

use candle_core::{Module as _, Tensor};
use candle_nn::{Linear, VarBuilder};

use crate::adapters::lora::{LoraConfig, LoraLayer};
use crate::error::{PeftError, Result};
use crate::traits::{Adapter, AdapterConfig, Mergeable};

/// Pattern for matching module names.
#[derive(Debug, Clone)]
pub enum ModulePattern {
    /// Match exact module name
    Exact(String),
    /// Match modules ending with suffix (e.g., `*.attention`)
    Suffix(String),
    /// Match modules starting with prefix (e.g., `layer.*`)
    Prefix(String),
    /// Match all modules
    All,
}

impl ModulePattern {
    /// Parse a pattern string into a `ModulePattern`.
    ///
    /// # Examples
    /// - `"encoder.layer.0"` -> `Exact`
    /// - `"*.attention"` -> `Suffix`
    /// - `"layer.*"` -> `Prefix`
    /// - `"*"` -> `All`
    #[must_use]
    pub fn parse(pattern: &str) -> Self {
        match pattern {
            "*" => Self::All,
            s if s.starts_with("*.") => Self::Suffix(s[2..].to_string()),
            s if s.ends_with(".*") => Self::Prefix(s[..s.len() - 2].to_string()),
            s => Self::Exact(s.to_string()),
        }
    }

    /// Check if a module name matches this pattern.
    #[must_use]
    pub fn matches(&self, module_name: &str) -> bool {
        match self {
            Self::Exact(name) => module_name == name,
            Self::Suffix(suffix) => module_name.ends_with(suffix),
            Self::Prefix(prefix) => module_name.starts_with(prefix),
            Self::All => true,
        }
    }
}

/// Candle `Linear` base weights + LoRA residual with a real forward pass.
///
/// Base weights are held as a plain [`Linear`] (typically **not** registered in
/// the adapter `VarMap`, so AdamW on adapter vars leaves the base frozen).
pub struct LinearWithLora {
    /// Frozen (or caller-owned) base linear layer.
    base: Linear,
    /// LoRA residual adapter.
    lora: LoraLayer,
    /// Module name (for HF key prefixes / debugging).
    name: String,
}

impl LinearWithLora {
    /// Wrap an existing Linear with a newly constructed LoRA residual.
    ///
    /// # Errors
    /// Returns an error if LoRA construction fails or weight shapes mismatch.
    pub fn new(name: impl Into<String>, base: Linear, config: LoraConfig, vb: VarBuilder) -> Result<Self> {
        let w = base.weight();
        let dims = w.dims();
        if dims.len() != 2 {
            return Err(PeftError::InvalidConfig(
                "LinearWithLora expects rank-2 base weight [out, in]".into(),
            ));
        }
        let out_features = dims[0];
        let in_features = dims[1];
        let lora = LoraLayer::new(in_features, out_features, config, vb)?;
        Ok(Self {
            base,
            lora,
            name: name.into(),
        })
    }

    /// Wrap base Linear with an existing [`LoraLayer`].
    ///
    /// # Errors
    /// Returns shape mismatch if adapter dims do not match the base weight.
    pub fn from_parts(name: impl Into<String>, base: Linear, lora: LoraLayer) -> Result<Self> {
        let dims = base.weight().dims();
        if dims.len() != 2 {
            return Err(PeftError::InvalidConfig(
                "base weight must be rank-2".into(),
            ));
        }
        let (out_f, in_f) = (dims[0], dims[1]);
        let a = lora.lora_a_shape();
        let b = lora.lora_b_shape();
        if a.len() != 2 || b.len() != 2 || a[1] != in_f || b[0] != out_f {
            return Err(PeftError::ShapeMismatch {
                expected: vec![out_f, in_f],
                actual: vec![b.first().copied().unwrap_or(0), a.get(1).copied().unwrap_or(0)],
            });
        }
        Ok(Self {
            base,
            lora,
            name: name.into(),
        })
    }

    /// Module name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Base linear layer (weights are not updated by adapter-only optimizers).
    #[must_use]
    pub fn base(&self) -> &Linear {
        &self.base
    }

    /// Mutable base access (for merge writeback etc.).
    pub fn base_mut(&mut self) -> &mut Linear {
        &mut self.base
    }

    /// LoRA adapter.
    #[must_use]
    pub fn lora(&self) -> &LoraLayer {
        &self.lora
    }

    /// Mutable LoRA adapter.
    pub fn lora_mut(&mut self) -> &mut LoraLayer {
        &mut self.lora
    }

    /// Forward: `base(x) + lora(x)`.
    ///
    /// # Errors
    /// Propagates candle / adapter errors.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let base_out = self.base.forward(input)?;
        self.lora.forward(input, Some(&base_out))
    }

    /// Merge LoRA into a new weight tensor (does not mutate base in place).
    ///
    /// # Errors
    /// Returns merge errors from the adapter.
    pub fn merged_weight(&self) -> Result<Tensor> {
        self.lora.merge(self.base.weight())
    }

    /// Number of LoRA trainable parameters.
    #[must_use]
    pub fn num_adapter_parameters(&self) -> usize {
        self.lora.num_parameters()
    }
}

/// Ordered collection of [`LinearWithLora`] modules with a real multi-layer forward.
///
/// This is the **product inject path** for tiny stacks and custom candle models.
pub struct PeftLinearModel {
    /// Named modules in insertion order.
    modules: Vec<LinearWithLora>,
    /// Index by name for lookup.
    index: HashMap<String, usize>,
    /// Active adapter label (single adapter per module today).
    active_adapter: String,
    /// Shared LoRA config snapshot.
    config: LoraConfig,
}

impl PeftLinearModel {
    /// Build from ordered `(name, Linear)` pairs, injecting LoRA into each.
    ///
    /// Adapter parameters are created under `vb.pp(name)` so a single
    /// [`candle_nn::VarMap`] can drive AdamW on all adapter weights.
    /// Base Linear tensors are **not** put into that VarMap by this constructor
    /// (base frozen for adapter-only training).
    ///
    /// # Errors
    /// Returns an error if any module fails construction or names collide.
    pub fn from_linears(
        base_modules: Vec<(String, Linear)>,
        config: LoraConfig,
        adapter_name: impl Into<String>,
        vb: VarBuilder,
    ) -> Result<Self> {
        config.validate()?;
        let active_adapter = adapter_name.into();
        let mut modules = Vec::with_capacity(base_modules.len());
        let mut index = HashMap::new();

        for (name, linear) in base_modules {
            if index.contains_key(&name) {
                return Err(PeftError::AdapterExists { name });
            }
            let layer = LinearWithLora::new(name.clone(), linear, config.clone(), vb.pp(&name))?;
            index.insert(name, modules.len());
            modules.push(layer);
        }

        Ok(Self {
            modules,
            index,
            active_adapter,
            config,
        })
    }

    /// Active adapter name.
    #[must_use]
    pub fn active_adapter(&self) -> &str {
        &self.active_adapter
    }

    /// LoRA config used at construction.
    #[must_use]
    pub fn config(&self) -> &LoraConfig {
        &self.config
    }

    /// Module names in forward order.
    #[must_use]
    pub fn module_names(&self) -> Vec<&str> {
        self.modules.iter().map(LinearWithLora::name).collect()
    }

    /// Number of wrapped modules.
    #[must_use]
    pub fn num_modules(&self) -> usize {
        self.modules.len()
    }

    /// Total adapter parameters across modules.
    #[must_use]
    pub fn num_adapter_parameters(&self) -> usize {
        self.modules
            .iter()
            .map(LinearWithLora::num_adapter_parameters)
            .sum()
    }

    /// Get a module by name.
    ///
    /// # Errors
    /// Returns [`PeftError::AdapterNotFound`] if missing.
    pub fn get(&self, name: &str) -> Result<&LinearWithLora> {
        let idx = self.index.get(name).ok_or_else(|| PeftError::AdapterNotFound {
            name: format!("module '{name}' not found"),
        })?;
        Ok(&self.modules[*idx])
    }

    /// Mutable module by name.
    ///
    /// # Errors
    /// Returns [`PeftError::AdapterNotFound`] if missing.
    pub fn get_mut(&mut self, name: &str) -> Result<&mut LinearWithLora> {
        let idx = self.index.get(name).ok_or_else(|| PeftError::AdapterNotFound {
            name: format!("module '{name}' not found"),
        })?;
        Ok(&mut self.modules[*idx])
    }

    /// Sequential forward through all modules (tiny MLP / stack).
    ///
    /// # Errors
    /// Propagates layer errors.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        for module in &self.modules {
            x = module.forward(&x)?;
        }
        Ok(x)
    }

    /// Forward a single named module.
    ///
    /// # Errors
    /// Missing module or forward failure.
    pub fn forward_module(&self, name: &str, input: &Tensor) -> Result<Tensor> {
        self.get(name)?.forward(input)
    }

    /// Iterate modules in order.
    pub fn iter(&self) -> impl Iterator<Item = &LinearWithLora> {
        self.modules.iter()
    }

    /// Mutable iteration.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut LinearWithLora> {
        self.modules.iter_mut()
    }
}

/// Inject LoRA into named Linear modules (product `get_peft_model` path).
///
/// This is the successor to the string-only registry: each matching base Linear
/// becomes a [`LinearWithLora`] with a real forward.
///
/// # Arguments
/// * `base_modules` — full list of `(name, Linear)` from the host model
/// * `pattern` — which names receive LoRA (`*`, `*.q_proj`, exact, …)
/// * `config` — LoRA hyperparameters (`target_modules` is **not** auto-applied;
///   the `pattern` argument is the source of truth for injection)
/// * `adapter_name` — label stored on the model (default adapter name for docs/HF)
/// * `vb` — variable builder for trainable adapter weights
///
/// Modules that do **not** match `pattern` are dropped from the returned model
/// (callers should keep non-target layers outside this wrapper). For a full
/// network, compose matched [`LinearWithLora`] layers with plain Linears yourself.
///
/// # Errors
/// Returns construction errors from [`PeftLinearModel::from_linears`].
pub fn get_peft_model(
    base_modules: Vec<(String, Linear)>,
    pattern: &str,
    config: LoraConfig,
    adapter_name: impl Into<String>,
    vb: VarBuilder,
) -> Result<PeftLinearModel> {
    let pat = ModulePattern::parse(pattern);
    let selected: Vec<(String, Linear)> = base_modules
        .into_iter()
        .filter(|(name, _)| pat.matches(name))
        .collect();
    if selected.is_empty() {
        return Err(PeftError::InvalidConfig(format!(
            "no modules matched pattern '{pattern}' for LoRA injection"
        )));
    }
    PeftLinearModel::from_linears(selected, config, adapter_name, vb)
}

/// Legacy name-list adapter registry (no base weights).
///
/// Prefer [`PeftLinearModel`] / [`get_peft_model`] for real inject + forward.
struct ModuleAdapter<A: Adapter> {
    adapter: A,
    active: bool,
}

/// PEFT model wrapper for managing adapters across modules (registry only).
///
/// Provides module-level adapter management with pattern-based targeting.
/// Does **not** own base Linear weights — see [`PeftLinearModel`].
pub struct PeftModel<A: Adapter> {
    /// Map of module names to their adapters
    module_adapters: HashMap<String, HashMap<String, ModuleAdapter<A>>>,
    /// Currently active adapter name (global default)
    active_adapter: Option<String>,
    /// List of all registered adapter names
    adapter_names: Vec<String>,
}

impl<A: Adapter> PeftModel<A> {
    /// Create a new PEFT model wrapper.
    #[must_use]
    pub fn new() -> Self {
        Self {
            module_adapters: HashMap::new(),
            active_adapter: None,
            adapter_names: Vec::new(),
        }
    }

    /// Add an adapter to modules matching the given pattern.
    ///
    /// # Arguments
    /// * `adapter_name` - Unique name for the adapter
    /// * `pattern` - Pattern to match module names
    /// * `module_names` - List of all module names in the model
    /// * `adapter_factory` - Function to create adapter instances
    ///
    /// # Errors
    /// Returns an error if adapter creation fails
    pub fn add_adapter<F>(
        &mut self,
        adapter_name: impl Into<String>,
        pattern: &str,
        module_names: &[&str],
        adapter_factory: F,
    ) -> Result<usize>
    where
        F: Fn(&str) -> Result<A>,
    {
        let adapter_name = adapter_name.into();
        let pattern = ModulePattern::parse(pattern);
        let mut count = 0;

        for &module_name in module_names {
            if pattern.matches(module_name) {
                let adapter = adapter_factory(module_name)?;
                let module_name_owned = module_name.to_string();

                let module_entry = self.module_adapters.entry(module_name_owned).or_default();

                module_entry.insert(
                    adapter_name.clone(),
                    ModuleAdapter {
                        adapter,
                        active: self.active_adapter.is_none(),
                    },
                );
                count += 1;
            }
        }

        // Track adapter name
        if !self.adapter_names.contains(&adapter_name) {
            self.adapter_names.push(adapter_name.clone());
        }

        // Set as active if first adapter
        if self.active_adapter.is_none() && count > 0 {
            self.active_adapter = Some(adapter_name);
        }

        Ok(count)
    }

    /// Set the active adapter for a specific module.
    ///
    /// # Errors
    /// Returns an error if the module or adapter doesn't exist
    pub fn set_adapter(&mut self, module_name: &str, adapter_name: &str) -> Result<()> {
        let adapters = self.module_adapters.get_mut(module_name).ok_or_else(|| {
            PeftError::AdapterNotFound {
                name: format!("module '{module_name}' not found"),
            }
        })?;

        if !adapters.contains_key(adapter_name) {
            return Err(PeftError::AdapterNotFound {
                name: format!("adapter '{adapter_name}' not found in module '{module_name}'"),
            });
        }

        // Deactivate all adapters for this module
        for adapter_entry in adapters.values_mut() {
            adapter_entry.active = false;
        }

        // Activate the requested adapter
        if let Some(entry) = adapters.get_mut(adapter_name) {
            entry.active = true;
        }

        Ok(())
    }

    /// Set the active adapter for all modules.
    ///
    /// # Errors
    /// Returns an error if the adapter doesn't exist in any module
    pub fn set_adapter_all(&mut self, adapter_name: impl Into<String>) -> Result<()> {
        let adapter_name = adapter_name.into();

        if !self.adapter_names.contains(&adapter_name) {
            return Err(PeftError::AdapterNotFound { name: adapter_name });
        }

        for adapters in self.module_adapters.values_mut() {
            // Deactivate all
            for entry in adapters.values_mut() {
                entry.active = false;
            }
            // Activate the requested one if it exists
            if let Some(entry) = adapters.get_mut(&adapter_name) {
                entry.active = true;
            }
        }

        self.active_adapter = Some(adapter_name);
        Ok(())
    }

    /// Get the active adapter name.
    #[must_use]
    pub fn active_adapter_name(&self) -> Option<&str> {
        self.active_adapter.as_deref()
    }

    /// Get all registered adapter names.
    #[must_use]
    pub fn adapter_names(&self) -> &[String] {
        &self.adapter_names
    }

    /// Get module names that have adapters.
    #[must_use]
    pub fn module_names(&self) -> Vec<&str> {
        self.module_adapters.keys().map(String::as_str).collect()
    }

    /// Check if a module has any adapters.
    #[must_use]
    pub fn has_adapter(&self, module_name: &str) -> bool {
        self.module_adapters.contains_key(module_name)
    }

    /// Forward pass for a specific module.
    ///
    /// # Arguments
    /// * `module_name` - Name of the module
    /// * `input` - Input tensor
    /// * `base_output` - Optional base layer output
    ///
    /// # Errors
    /// Returns an error if module not found or no active adapter
    pub fn forward_module(
        &self,
        module_name: &str,
        input: &Tensor,
        base_output: Option<&Tensor>,
    ) -> Result<Tensor> {
        let adapters =
            self.module_adapters
                .get(module_name)
                .ok_or_else(|| PeftError::AdapterNotFound {
                    name: format!("module '{module_name}' not found"),
                })?;

        // Find active adapter
        for entry in adapters.values() {
            if entry.active {
                return entry.adapter.forward(input, base_output);
            }
        }

        Err(PeftError::AdapterNotFound {
            name: format!("no active adapter for module '{module_name}'"),
        })
    }

    /// Get a reference to an adapter for a module.
    ///
    /// # Errors
    /// Returns an error if module or adapter not found
    pub fn get_adapter(&self, module_name: &str, adapter_name: &str) -> Result<&A> {
        let adapters =
            self.module_adapters
                .get(module_name)
                .ok_or_else(|| PeftError::AdapterNotFound {
                    name: format!("module '{module_name}' not found"),
                })?;

        adapters
            .get(adapter_name)
            .map(|entry| &entry.adapter)
            .ok_or_else(|| PeftError::AdapterNotFound {
                name: format!("adapter '{adapter_name}' not found in module '{module_name}'"),
            })
    }

    /// Get the total number of trainable parameters across all active adapters.
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        self.module_adapters
            .values()
            .flat_map(|adapters| adapters.values())
            .filter(|entry| entry.active)
            .map(|entry| entry.adapter.num_parameters())
            .sum()
    }

    /// Get the number of modules with adapters.
    #[must_use]
    pub fn num_modules(&self) -> usize {
        self.module_adapters.len()
    }
}

impl<A: Adapter> Default for PeftModel<A> {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a **legacy** PEFT adapter registry (name-list only, no base Linear).
///
/// For real base+adapter forward and training, use [`get_peft_model`] with
/// `Vec<(String, Linear)>` instead.
///
/// # Errors
/// Returns an error if adapter creation fails
pub fn get_peft_model_registry<A: Adapter, F>(
    module_names: &[&str],
    pattern: &str,
    adapter_name: impl Into<String>,
    adapter_factory: F,
) -> Result<PeftModel<A>>
where
    F: Fn(&str) -> Result<A>,
{
    let mut model = PeftModel::new();
    model.add_adapter(adapter_name, pattern, module_names, adapter_factory)?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LoraConfig, LoraLayer};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{linear_no_bias, VarBuilder, VarMap};

    #[test]
    fn test_module_pattern_exact() {
        let pattern = ModulePattern::parse("encoder.layer.0");
        assert!(pattern.matches("encoder.layer.0"));
        assert!(!pattern.matches("encoder.layer.1"));
        assert!(!pattern.matches("decoder.layer.0"));
    }

    #[test]
    fn test_module_pattern_suffix() {
        let pattern = ModulePattern::parse("*.attention");
        assert!(pattern.matches("layer.0.attention"));
        assert!(pattern.matches("encoder.layer.0.attention"));
        assert!(!pattern.matches("attention.output"));
    }

    #[test]
    fn test_module_pattern_prefix() {
        let pattern = ModulePattern::parse("encoder.*");
        assert!(pattern.matches("encoder.layer.0"));
        assert!(pattern.matches("encoder.attention"));
        assert!(!pattern.matches("decoder.layer.0"));
    }

    #[test]
    fn test_module_pattern_all() {
        let pattern = ModulePattern::parse("*");
        assert!(pattern.matches("anything"));
        assert!(pattern.matches("encoder.layer.0"));
        assert!(pattern.matches(""));
    }

    #[test]
    fn test_peft_model_creation() {
        let model: PeftModel<LoraLayer> = PeftModel::new();
        assert!(model.module_names().is_empty());
        assert!(model.active_adapter_name().is_none());
    }

    #[test]
    fn test_add_adapter_with_pattern() -> Result<()> {
        let mut model: PeftModel<LoraLayer> = PeftModel::new();
        let device = Device::Cpu;
        let config = LoraConfig::default();

        let module_names = vec![
            "encoder.layer.0.attention",
            "encoder.layer.0.mlp",
            "encoder.layer.1.attention",
            "encoder.layer.1.mlp",
            "decoder.layer.0.attention",
        ];

        let count = model.add_adapter("lora", "*.attention", &module_names, |_| {
            LoraLayer::new_with_zeros(768, 768, config.clone(), &device)
        })?;

        assert_eq!(count, 3); // 3 attention modules
        assert_eq!(model.active_adapter_name(), Some("lora"));
        assert!(model.has_adapter("encoder.layer.0.attention"));
        assert!(model.has_adapter("encoder.layer.1.attention"));
        assert!(model.has_adapter("decoder.layer.0.attention"));
        assert!(!model.has_adapter("encoder.layer.0.mlp"));

        Ok(())
    }

    #[test]
    fn test_set_adapter() -> Result<()> {
        let mut model: PeftModel<LoraLayer> = PeftModel::new();
        let device = Device::Cpu;
        let config = LoraConfig::default();

        let module_names = vec!["layer.0"];

        model.add_adapter("adapter1", "*", &module_names, |_| {
            LoraLayer::new_with_zeros(768, 768, config.clone(), &device)
        })?;

        model.add_adapter("adapter2", "*", &module_names, |_| {
            LoraLayer::new_with_zeros(768, 768, config.clone(), &device)
        })?;

        // Switch adapter for specific module
        model.set_adapter("layer.0", "adapter2")?;

        Ok(())
    }

    #[test]
    fn test_set_adapter_all() -> Result<()> {
        let mut model: PeftModel<LoraLayer> = PeftModel::new();
        let device = Device::Cpu;
        let config = LoraConfig::default();

        let module_names = vec!["layer.0", "layer.1"];

        model.add_adapter("adapter1", "*", &module_names, |_| {
            LoraLayer::new_with_zeros(768, 768, config.clone(), &device)
        })?;

        model.add_adapter("adapter2", "*", &module_names, |_| {
            LoraLayer::new_with_zeros(768, 768, config.clone(), &device)
        })?;

        assert_eq!(model.active_adapter_name(), Some("adapter1"));

        model.set_adapter_all("adapter2")?;
        assert_eq!(model.active_adapter_name(), Some("adapter2"));

        Ok(())
    }

    #[test]
    fn test_forward_module() -> Result<()> {
        let mut model: PeftModel<LoraLayer> = PeftModel::new();
        let device = Device::Cpu;
        let config = LoraConfig::default();

        let module_names = vec!["layer.0"];

        model.add_adapter("lora", "*", &module_names, |_| {
            LoraLayer::new_with_zeros(768, 768, config.clone(), &device)
        })?;

        let input = Tensor::zeros(&[1, 10, 768], DType::F32, &device)?;
        let output = model.forward_module("layer.0", &input, None)?;

        assert_eq!(output.dims(), &[1, 10, 768]);

        Ok(())
    }

    #[test]
    fn test_num_parameters() -> Result<()> {
        let mut model: PeftModel<LoraLayer> = PeftModel::new();
        let device = Device::Cpu;
        let config = LoraConfig::default();

        let module_names = vec!["layer.0", "layer.1"];

        model.add_adapter("lora", "*", &module_names, |_| {
            LoraLayer::new_with_zeros(768, 768, config.clone(), &device)
        })?;

        // 2 modules, each with 768*8 + 8*768 = 12,288 parameters
        assert_eq!(model.num_parameters(), 2 * (768 * 8 + 8 * 768));

        Ok(())
    }

    #[test]
    fn test_get_peft_model_registry() -> Result<()> {
        let device = Device::Cpu;
        let config = LoraConfig::default();

        let module_names = vec!["layer.0.attention", "layer.0.mlp", "layer.1.attention"];

        let model = get_peft_model_registry(&module_names, "*.attention", "lora", |_| {
            LoraLayer::new_with_zeros(768, 768, config.clone(), &device)
        })?;

        assert_eq!(model.num_modules(), 2);
        assert!(model.has_adapter("layer.0.attention"));
        assert!(model.has_adapter("layer.1.attention"));
        assert!(!model.has_adapter("layer.0.mlp"));

        Ok(())
    }

    #[test]
    fn test_linear_with_lora_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        // Base linear not in adapter varmap
        let base_w = Tensor::randn(0f32, 0.02, (16, 16), &device)?;
        let base = Linear::new(base_w, None);
        let config = LoraConfig {
            r: 4,
            alpha: 8,
            ..Default::default()
        };
        let layer = LinearWithLora::new("fc1", base, config, vb.pp("fc1"))?;
        let x = Tensor::zeros(&[2, 3, 16], DType::F32, &device)?;
        let y = layer.forward(&x)?;
        assert_eq!(y.dims(), &[2, 3, 16]);
        assert_eq!(layer.num_adapter_parameters(), 4 * (16 + 16));
        Ok(())
    }

    #[test]
    fn test_get_peft_model_injects_linears() -> Result<()> {
        let device = Device::Cpu;
        let mut base_vm = VarMap::new();
        let base_vb = VarBuilder::from_varmap(&base_vm, DType::F32, &device);
        // Build two base linears (simulating host model params — we clone tensors out)
        let l0 = linear_no_bias(8, 8, base_vb.pp("fc1"))?;
        let l1 = linear_no_bias(8, 8, base_vb.pp("fc2"))?;
        // Detach into plain Linears so adapter varmap is separate
        let base_modules = vec![
            ("mlp.fc1".into(), Linear::new(l0.weight().copy()?, l0.bias().cloned())),
            ("mlp.fc2".into(), Linear::new(l1.weight().copy()?, l1.bias().cloned())),
            (
                "other".into(),
                Linear::new(Tensor::zeros((8, 8), DType::F32, &device)?, None),
            ),
        ];

        let adapter_vm = VarMap::new();
        let adapter_vb = VarBuilder::from_varmap(&adapter_vm, DType::F32, &device);
        let config = LoraConfig {
            r: 2,
            alpha: 4,
            ..Default::default()
        };
        let model = get_peft_model(base_modules, "mlp.*", config, "default", adapter_vb)?;
        assert_eq!(model.num_modules(), 2);
        assert_eq!(model.module_names(), vec!["mlp.fc1", "mlp.fc2"]);

        let x = Tensor::randn(0f32, 1f32, (1, 4, 8), &device)?;
        let y = model.forward(&x)?;
        assert_eq!(y.dims(), &[1, 4, 8]);

        // Adapter varmap should hold LoRA params only
        assert!(!adapter_vm.all_vars().is_empty());
        // base varmap still independent
        assert!(!base_vm.all_vars().is_empty());
        let _ = &mut base_vm;
        Ok(())
    }
}
