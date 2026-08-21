# peft-rs dependency graph

## DAG (no cycles)

```text
candle-core 0.11, candle-nn 0.11, safetensors, serde, …
              │
              ▼
         ┌─────────┐     optional feature `unsloth`
         │ peft-rs │──────────────────────────► unsloth-rs (consume RMSNorm/SwiGLU)
         └────┬────┘
              │ depended on by
              ▼
    qlora-rs, axolotl-rs (optional)
    third-party: vox-foundation/vox (crates.io 1.0.3 + local candle patch)
```

**peft-rs must never depend on qlora-rs or axolotl-rs.** Optional `unsloth-rs` is a
**consume-path only** (`RmsNorm` / `SwiGLU` re-export). It does not un-quarantine
`src/kernels/archive` and does not dispatch a fused LoRA-add kernel.

Quant codecs live in qlora-rs; peft only exposes `quant` **traits** (`QuantizedBaseLinear`)
so qlora can implement them without a reverse edge into peft’s dependency list beyond
the existing qlora → peft edge.

## Candle

| Tree | candle-core / candle-nn |
|------|-------------------------|
| crates.io **1.0.3** | 0.9 |
| GitHub tag **v1.1.0** (unpublished) | 0.9 |
| **this tree** | **0.11** (see `Cargo.toml`) |

Candle types (`Tensor`, `Linear`, `VarBuilder`, `Device`) are in the public API.
Mixing peft-rs built on 0.11 with a workspace on 0.9/0.10 will fail to unify types
(the failure Vox patched around on 1.0.3).

## Cargo features

| Feature | Effect |
|---------|--------|
| *(default)* | CPU candle 0.11 |
| `cuda` | `candle-core/cuda` only — no CubeCL fused kernels |
| `unsloth` | Optional `unsloth-rs` consume (RMSNorm/SwiGLU). `should_dispatch_unsloth_lora() == false` |

Live sister versions: each crate's `Cargo.toml`. Consumers pin major (`peft-rs = "1"`).

## Consumers

| Crate | How it depends on peft |
|-------|-------------------------|
| qlora-rs | Required dep (`peft-rs = "1.2.1"` on crates.io; local path overlay for SoT) |
| axolotl-rs | Optional feature `peft` (`peft-rs = "1.2"` registry; local path overlay for SoT) |
| rust-ai-core | Re-export / facade (must not force reverse deps) |
| vox-foundation/vox | crates.io 1.0.3 + `[patch]` candle 0.10 — external |
