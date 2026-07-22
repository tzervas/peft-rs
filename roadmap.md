# PEFT-RS Roadmap

## Product class

Candle PEFT **adapter layer library** — modular low-rank / soft-prompt layers for
Rust + [candle](https://github.com/huggingface/candle). **Not** a drop-in
HuggingFace PEFT framework (no automatic base-model injection, no QLoRA, no
trainer loop in this crate).

## Success criteria (honest)

| Horizon | Criteria | Status |
|---------|----------|--------|
| **Now (1.0.4 honesty train)** | Docs match code; overclaimed flags fixed or documented; default tests green; METRICS scaffold | **In progress / this release** |
| **Near (HF interop)** | `adapter_config.json` + LoRA key names loadable from Python peft for core LoRA | **Not met** (PR-040) |
| **Near (model path)** | Real Linear+adapter wrap / `get_peft_model` that owns a base module | **Not met** (PR-041) |
| **Mid** | Numerical parity fixtures vs Python peft; freeze/grad truth on Var path | **Not met** |
| **Later** | Optional fused CUDA kernels (CubeCL) if CI-proven; broader tuner surface | **Not met** / kernels quarantined |

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

- [ ] HF adapter_config schema + LoRA key interop (PEFT-P0-07/08)
- [ ] Real PeftModel / get_peft_model path (PEFT-P0-09)
- [ ] Golden numerical parity vs Python peft (PEFT-P0-10)
- [ ] modules_to_save policy (PEFT-P0-11)
- [ ] Training: real Var updates or explicit non-goal rename (PEFT-P0-12)
- [ ] Multi-adapter weighted composition (PEFT-P1-01)
- [ ] AdaLoRA top-k budget (PEFT-P1-02)
- [ ] Prefix reparam + prompt text init (PEFT-P1-03)
- [ ] Optional CubeCL kernel restore under feature (PEFT-P1-05) — currently quarantined
- [ ] Conv2d / Embedding LoRA (PEFT-P1-06)
- [ ] Additional tuners (p-tuning, X-LoRA, etc.) only after P0/P1 (PEFT-P2-01)
- [ ] Non-empty criterion benches + property tests (PEFT-P2-02)
- [ ] Fill METRICS.md with real numbers (PEFT-P2-03)

## Non-goals (until later PRs)

- Full HuggingFace PEFT drop-in parity
- Built-in QLoRA / bitsandbytes / GPTQ / AWQ
- PeftTrainer-equivalent training loop
- Automatic transformers module introspection
- Active fused CUDA kernels in default or `cuda` feature (see `src/kernels/`)
