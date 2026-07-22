# PEFT-RS Roadmap

## Product class

Candle PEFT **adapter layer library** — modular low-rank / soft-prompt layers for
Rust + [candle](https://github.com/huggingface/candle). **Not** a drop-in
HuggingFace PEFT framework (no automatic base-model injection, no QLoRA, no
trainer loop in this crate).

## Success criteria (honest)

| Horizon | Criteria | Status |
|---------|----------|--------|
| **1.0.4 honesty train** | Docs match code; flags honesty; METRICS scaffold | **Met** |
| **1.1.0 HF + inject + parity + train/multi/quant** | HF config/keys, Linear inject, LoRA goldens, train step, weighted multi-adapter, quant bridge | **Met** (this release) |
| **Mid** | freeze/grad truth on Var path; wall-time METRICS | **Not met** |
| **Later** | Optional fused CUDA kernels; broader tuner surface | **Not met** / kernels quarantined |

> **Do not claim “success criteria already met” or full Python parity.** Layer math
> for several adapters exists; framework depth does not.

## Requirements (active)

1. Keep `cargo test --lib` green on CPU without CUDA toolkit.
2. Prefer honesty: implement behavior or document/remove config flags that no-op.
3. METRICS.md tracks comparison plan; numbers may be “not yet measured”.
4. Showcase bar: target parity+ later — correctness before vanity speed claims.

## Deliverables (tracked)

| Item | Notes |
|------|--------|
| Honest README status matrix | PR-010 |
| METRICS.md scaffold | PR-010 |
| Dropout / LoftQ / DoRA SaveLoad / freeze honesty | PR-020 |
| Kernels quarantine or rewire | PR-021 (quarantine chosen) |
| HF config + weight keys | PR-040 |
| Real model inject | PR-041 |

## Remaining tasks

- [x] HF adapter_config schema + LoRA key interop (PEFT-P0-07/08) — PR-040
- [x] Real Linear inject / get_peft_model path (PEFT-P0-09) — PR-041
- [x] Golden numerical parity for LoRA forward/merge (PEFT-P0-10) — PR-042
- [x] modules_to_save policy documented config-only (PEFT-P0-11) — PR-041
- [x] Training: real Var updates via `train_step_mse` (PEFT-P0-12) — PR-072; full PeftTrainer remains non-goal
- [x] Multi-adapter weighted composition (PEFT-P1-01) — PR-080
- [x] AdaLoRA top-k budget (PEFT-P1-02)
- [x] Prefix reparam + prompt text init (PEFT-P1-03)
- [x] Quant bridge trait for qlora-rs (PEFT-P1-04) — PR-081
- [ ] Optional CubeCL kernel restore under feature (PEFT-P1-05) — currently quarantined
- [ ] Conv2d / Embedding LoRA (PEFT-P1-06)
- [ ] Additional tuners (p-tuning, X-LoRA, etc.) only after P0/P1 (PEFT-P2-01)
- [x] Non-empty criterion benches (PEFT-P2-02 partial) — forward/merge/compose; property tests still open
- [ ] Fill METRICS.md with real numbers (PEFT-P2-03)

## Non-goals (until later PRs)

- Full HuggingFace PEFT drop-in parity
- Built-in QLoRA / bitsandbytes / GPTQ / AWQ
- PeftTrainer-equivalent training loop
- Automatic transformers module introspection
- Active fused CUDA kernels in default or `cuda` feature (see `src/kernels/`)
