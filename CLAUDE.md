# peft-rs (agent notes)

**SoT:** [`Cargo.toml`](Cargo.toml) + [README.md](README.md) + [docs/VERSIONING.md](docs/VERSIONING.md).

Candle **0.11** PEFT **layer math** (LoRA/DoRA/…) plus Linear inject (`LinearWithLora` / `get_peft_model`). **Not** a HuggingFace PEFT port.

- Optional `unsloth`: consume RMSNorm/SwiGLU. `should_dispatch_unsloth_lora()` is always **false**.
- Optional `cuda`: `candle-core/cuda` only. Fused kernels are quarantined under `src/kernels/archive/`.
- Hub-safe save: `save_pretrained_hf` / `save_multi_module_pretrained_hf` (`lora_A.default.weight`). Native `save_pretrained` keys are not Hub-safe.
- MSRV **1.96**. Consumers pin `peft-rs = "1"`.
- Hosted CI: default `cargo test` plus `--features unsloth`. Do **not** `--all-features` (pulls `cuda`).

Do not invent `dora.rs` / `prefix.rs` / compiled `inference.rs`. DoRA lives in `adapters/lora.rs`; prefix is `prefix_tuning.rs`.
