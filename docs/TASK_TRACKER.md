# PEFT-RS Task Tracker

Honest implementation status for peft-rs. Prefer [README.md](../README.md) status
matrix and [METRICS.md](../METRICS.md) over older “all complete” tables.

> **Product class:** Candle PEFT adapter **layer library** — not full HF PEFT.

## Status overview (2026-07-22 honesty reset)

| Feature | Status | Notes |
|---------|--------|-------|
| **LoRA** | partial | Forward/merge/save; dropout applied when unfrozen; no HF keys |
| **DoRA** | partial | Layer math + SaveLoad; fallback without base weight |
| **AdaLoRA** | partial | SVD params; simplified mask vs full adaptive budget |
| **IA³** | partial | Layer OK; no auto target inject |
| **LoHa / LoKr** | partial | Linear implementations |
| **OFT / BOFT** | partial | Math present; some options incomplete |
| **VeRA** | partial | Core path |
| **Prefix / Prompt** | thin | Embeddings helpers; limited integration |
| **Adapter / Mergeable / Config traits** | done | Core interfaces |
| **Trainable freeze** | partial | Flag + dropout gate; no Var detach |
| **Weight I/O** | partial | safetensors; schema ≠ HF peft |
| **PeftModel / get_peft_model** | stub | Name-list registry only |
| **Multi-adapter registry** | partial | Single active switch |
| **Training utilities** | stub | LR schedules / counters only |
| **Fused CUDA kernels** | quarantined | `src/kernels/archive/` — not built |
| **QLoRA / quant** | missing | Non-goal in this crate |
| **HF full interop** | missing | PR-040+ |
| **METRICS vs Python peft** | scaffold | Numbers not yet measured |

## Open work (from UNIFIED_GAP_LEDGER)

| Gap | Topic | Wave |
|-----|--------|------|
| PEFT-P0-07/08 | HF config + weight keys | 3b |
| PEFT-P0-09 | Real model inject | 3b |
| PEFT-P0-10 | Numerical parity | 3b |
| PEFT-P0-11 | modules_to_save | 3b |
| PEFT-P0-12 | Real training step or non-goal | 3b |
| PEFT-P1-* | Compose, AdaLoRA top-k, prefix reparam, kernels restore | 3b |
| PEFT-P2-* | Extra tuners, benches, showcase metrics | 3c |

## Historical notes

Earlier versions of this file marked nearly every row ✅ Complete. That overstated
framework depth. Layer modules exist and unit tests pass; model integration,
interop, and parity remain open.
