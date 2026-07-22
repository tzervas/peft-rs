# peft-rs Metrics

**Status:** Correctness fixtures landed (PR-042); wall-time / RSS vs HF peft **not yet measured**.  
**Purpose:** Showcase-bar tracking for honesty releases.  
**Product class:** Candle PEFT **adapter layer library** + Linear inject path (not a drop-in HF PEFT framework).

---

## Comparison target

| Side | Stack |
|------|--------|
| **This crate** | `peft-rs` **1.1.0** on Candle 0.9 (CPU default; optional candle CUDA) |
| **Baseline** | HuggingFace [`peft`](https://github.com/huggingface/peft) LoRA linear math on PyTorch (optional offline verify) |

Fair comparisons must use the **same adapter math surface** (e.g. LoRA rank/alpha, same shapes).

---

## Methods to measure

| Method ID | What | Unit | Status |
|-----------|------|------|--------|
| `wall_lora_fwd` | LoRA layer forward (batch×seq×hidden) | ms / step | **not yet measured** |
| `wall_lora_merge` | Merge ΔW into base weight | ms | **not yet measured** |
| `wall_dora_fwd` | DoRA forward with base weight | ms / step | **not yet measured** |
| `rss_lora_layer` | Peak RSS for layer construct + N forwards | MiB | **not yet measured** |
| `correctness_lora_fwd` | Allclose vs fixed-seed / fixed-matrix golden | max abs err | **green in CI** (see below) |
| `correctness_lora_merge` | Merge residual vs golden | max abs err | **green in CI** |
| `throughput_tokens` | Tokens/s on tiny stack | tok/s | **not yet measured** (inject path exists; no bench numbers) |

### Planned shapes (when wall-time numbers land)

- Hidden sizes: 768, 2048  
- Rank `r`: 8, 64  
- Batch×seq: `(1, 128)`, `(4, 512)`  
- DType: F32 CPU first; F16/BF16 CUDA optional  

---

## Correctness (PR-042)

### Formula (HF peft Linear LoRA)

```
y = x @ W.T + (x @ A.T @ B.T) * (alpha / r)
W_merged = W + (B @ A) * (alpha / r)
```

Layout: `A` `[r, in]`, `B` `[out, r]`, `W` `[out, in]` (candle `Linear` / PEFT).

### Tolerances

| Check | atol | rtol |
|-------|------|------|
| `correctness_lora_fwd` | **1e-5** | **1e-5** |
| `correctness_lora_merge` | **1e-5** | **1e-5** |

### Fixture

- Path: [`tests/parity/fixtures/lora_fwd_merge.json`](tests/parity/fixtures/lora_fwd_merge.json)
- Runner: `cargo test --test parity_lora`
- Docs: [`tests/parity/README.md`](tests/parity/README.md)
- Regenerate (offline): `python3 scripts/gen_lora_parity_fixture.py`
- Optional peft import check: `python3 scripts/gen_lora_parity_fixture.py --verify-peft` (not CI)

### Result

| Method | peft-rs | Notes | Date |
|--------|---------|-------|------|
| `correctness_lora_fwd` | **pass** (allclose) | Fixed matrices; CI Rust-only | 2026-07-22 |
| `correctness_lora_merge` | **pass** (allclose) | Residual forward ≡ merged Linear | 2026-07-22 |

---

## How to reproduce

```bash
# Unit + HF I/O + inject + parity
cargo test --lib
cargo test --tests

# Parity only
cargo test --test parity_lora -- --nocapture

# Inject train demo
cargo run --example lora_inject_train

# Criterion harness still a stub for wall time
cargo bench --bench adapters
```

Record environment in every **wall-time** result row:

- CPU model / GPU model  
- `rustc` / candle versions  
- Python `peft` / `torch` versions (if comparing)  
- Date and git SHA  

---

## Result table (performance — fill when measured)

| Method | peft-rs | HF peft | Notes | Date / SHA |
|--------|---------|---------|-------|------------|
| `wall_lora_fwd` | — | — | not yet measured | — |
| `wall_lora_merge` | — | — | not yet measured | — |
| `wall_dora_fwd` | — | — | not yet measured | — |
| `rss_lora_layer` | — | — | not yet measured | — |
| `correctness_lora_fwd` | pass @ 1e-5 | math-equivalent fixture | PR-042 | 2026-07-22 |
| `correctness_lora_merge` | pass @ 1e-5 | math-equivalent fixture | PR-042 | 2026-07-22 |

---

## Showcase bar (target, not current claim)

Later releases aim for **parity+** on layer math (correctness first) and competitive wall time / RSS on fair fixtures. **1.1.0** closes correctness goldens for LoRA forward/merge; speed numbers remain open (PEFT-P2-03).

---

## Related gaps

| Gap | Topic | Status after 1.1.0 |
|-----|--------|--------------------|
| PEFT-P0-03 | This file | closed (scaffold + correctness) |
| PEFT-P0-10 | Numerical parity suite | **closed** |
| PEFT-P2-02 | Real criterion benches | open |
| PEFT-P2-03 | Showcase outperform metrics | open |
