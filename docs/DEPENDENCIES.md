# peft-rs dependency graph

## DAG (no cycles)

```text
candle-core, candle-nn, safetensors, serde, …
              │
              ▼
         ┌─────────┐
         │ peft-rs │  ← foundation; NO sister crate deps
         └────┬────┘
              │ depended on by
              ▼
    qlora-rs, axolotl-rs (optional)
```

**peft-rs must never depend on qlora-rs, unsloth-rs, or axolotl-rs.**

Quant codecs live in qlora-rs; peft only exposes `quant` **traits** (`QuantizedBaseLinear`)
so qlora can implement them without a reverse edge into peft’s dependency list beyond
the existing qlora → peft edge.

## Cargo features

| Feature | Effect |
|---------|--------|
| *(default)* | CPU candle |
| `cuda` | `candle-core/cuda` only — no CubeCL fused kernels |

## Consumers

| Crate | How it depends on peft |
|-------|-------------------------|
| qlora-rs | Required dep (`peft-rs ≥ 1.1.0` for 1.1 train/HF surface) |
| axolotl-rs | Optional feature `peft` |
| rust-ai-core | Re-export / facade (must not force reverse deps) |
