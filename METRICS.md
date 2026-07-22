# peft-rs Metrics Scaffold

**Status:** Scaffold only — **not yet measured** against HuggingFace PEFT.  
**Purpose:** Showcase-bar tracking for honesty releases (PR-010 / PEFT-P0-03).  
**Product class:** Candle PEFT **adapter layer library** (not a drop-in HF PEFT framework).

---

## Comparison target

| Side | Stack |
|------|--------|
| **This crate** | `peft-rs` 1.0.4+ on Candle 0.9 (CPU default; optional candle CUDA) |
| **Baseline** | HuggingFace [`peft`](https://github.com/huggingface/peft) on PyTorch (version pin TBD at measure time) |

Fair comparisons must use the **same adapter math surface** (e.g. LoRA rank/alpha, same shapes). peft-rs does **not** yet inject into full transformer models; layer-level benches only until PR-041+.

---

## Methods to measure

| Method ID | What | Unit | Status |
|-----------|------|------|--------|
| `wall_lora_fwd` | LoRA layer forward (batch×seq×hidden) | ms / step | **not yet measured** |
| `wall_lora_merge` | Merge ΔW into base weight | ms | **not yet measured** |
| `wall_dora_fwd` | DoRA forward with base weight | ms / step | **not yet measured** |
| `rss_lora_layer` | Peak RSS for layer construct + N forwards | MiB | **not yet measured** |
| `correctness_lora_fwd` | Allclose vs HF peft fixed-seed fixture | max abs err | **not yet measured** |
| `correctness_lora_merge` | Merge residual vs Python | max abs err | **not yet measured** |
| `throughput_tokens` | Tokens/s on tiny stack (when model inject exists) | tok/s | **blocked** (no real PeftModel wrap) |

### Planned shapes (when first numbers land)

- Hidden sizes: 768, 2048  
- Rank `r`: 8, 64  
- Batch×seq: `(1, 128)`, `(4, 512)`  
- DType: F32 CPU first; F16/BF16 CUDA optional  

---

## How to reproduce (once instrumented)

```bash
# Unit tests (smoke only today)
cargo test --lib

# Criterion harness exists but is empty stub (benches/adapters.rs)
cargo bench --bench adapters

# Future: parity fixtures (not yet)
# cargo test --test parity_lora -- --nocapture
```

Record environment in every result row:

- CPU model / GPU model  
- `rustc` / candle versions  
- Python `peft` / `torch` versions  
- Date and git SHA  

---

## Result table (fill when measured)

| Method | peft-rs | HF peft | Notes | Date / SHA |
|--------|---------|---------|-------|------------|
| `wall_lora_fwd` | — | — | not yet measured | — |
| `wall_lora_merge` | — | — | not yet measured | — |
| `wall_dora_fwd` | — | — | not yet measured | — |
| `rss_lora_layer` | — | — | not yet measured | — |
| `correctness_lora_fwd` | — | — | not yet measured | — |
| `correctness_lora_merge` | — | — | not yet measured | — |

---

## Showcase bar (target, not current claim)

Later releases aim for **parity+** on layer math (correctness first) and competitive wall time / RSS on fair fixtures. Current release prioritizes **docs and API honesty** over vanity benchmarks.

---

## Related gaps

| Gap | Topic |
|-----|--------|
| PEFT-P0-03 | This file |
| PEFT-P0-10 | Numerical parity suite |
| PEFT-P2-02 | Real criterion benches |
| PEFT-P2-03 | Showcase outperform metrics |
