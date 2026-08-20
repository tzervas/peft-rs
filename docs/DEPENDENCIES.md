# peft-rs dependency graph

## DAG (no cycles)

```text
candle-core 0.11, candle-nn 0.11, safetensors, serde, …
              │
              ▼
         ┌─────────┐
         │ peft-rs │  ← foundation; NO sister crate deps
         └────┬────┘
              │ depended on by
              ▼
    qlora-rs, axolotl-rs (optional)
    third-party: vox-foundation/vox (crates.io 1.0.3 + local candle patch)
```

**peft-rs must never depend on qlora-rs, unsloth-rs, or axolotl-rs.**

Quant codecs live in qlora-rs; peft only exposes `quant` **traits** (`QuantizedBaseLinear`)
so qlora can implement them without a reverse edge into peft’s dependency list beyond
the existing qlora → peft edge.

## Candle

| Tree | candle-core / candle-nn |
|------|-------------------------|
| crates.io **1.0.3** | 0.9 |
| GitHub tag **v1.1.0** (unpublished) | 0.9 |
| **this tree 1.2.0** | **0.11** (latest stable 2026-06-26) |

Candle types (`Tensor`, `Linear`, `VarBuilder`, `Device`) are in the public API.
Mixing peft-rs built on 0.11 with a workspace on 0.9/0.10 will fail to unify types
(the failure Vox patched around on 1.0.3).

## Cargo features

| Feature | Effect |
|---------|--------|
| *(default)* | CPU candle 0.11 |
| `cuda` | `candle-core/cuda` only — no CubeCL fused kernels |

## Consumers

| Crate | How it depends on peft |
|-------|-------------------------|
| qlora-rs | Required dep (must move to peft-rs 1.2.0 + candle 0.11 together) |
| axolotl-rs | Optional feature `peft` (still pinned `1.0` on crates.io; GitHub must bump) |
| rust-ai-core | Re-export / facade (must not force reverse deps) |
| vox-foundation/vox | crates.io 1.0.3 + `[patch]` candle 0.10 — external |
