# peft-rs Metrics

**Status:** Correctness fixtures green (PR-042); **CPU LoRA wall-time baselines** from criterion (`--quick`) recorded for 1.1.0.
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
| `wall_lora_fwd` | LoRA layer forward (batch×seq×hidden) | ms / step | **measured** (CPU criterion, below) |
| `wall_lora_merge` | Merge ΔW into base weight | ms | **measured** (CPU criterion, below) |
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

## CPU wall-time (PEFT-P2-02 / 1.1.0)

**Environment**

| Field | Value |
|-------|--------|
| CPU | Intel Core i7-14700K |
| Arch | x86_64 |
| OS | Linux |
| rustc | 1.98.0-nightly (2026-07-01) |
| candle | 0.9 |
| DType / device | F32 / CPU |
| Command | `cargo bench --bench adapters -- --quick` |
| Date / SHA | 2026-07-22 / `5f01624` (local tree) |
| Notes | Criterion `--quick` (short sample); not a full statistical run. **Not** compared to HF peft yet. Non-zero A/B weights. |

### LoRA forward

| Method | Shape | peft-rs (median-ish) | HF peft | Notes |
|--------|-------|----------------------|---------|-------|
| `wall_lora_fwd` | b1 s128 h768 r8 | **~0.87 ms** | — | residual only |
| `wall_lora_fwd` | b4 s128 h768 r8 | **~3.04 ms** | — | residual only |
| `wall_lora_fwd` | b1 s128 h768 r64 | **~2.69 ms** | — | residual only |
| `wall_lora_fwd` + base | b4 s128 h768 r8 | **~3.04 ms** | — | `forward(x, Some(base))` |

### LoRA merge

| Method | Shape | peft-rs | HF peft | Notes |
|--------|-------|---------|---------|-------|
| `wall_lora_merge` | h768 r8 | **~2.56 ms** | — | `W + B@A * scale` |
| `wall_lora_merge` | h768 r64 | **~2.51 ms** | — | |
| `wall_lora_merge` | h2048 r8 | **~14.6 ms** | — | |

Times are criterion central estimates from `--quick` (see band in bench log). Re-run without `--quick` for tighter CIs before publishing vs Python.

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

# Criterion LoRA benches
cargo bench --bench adapters -- --quick
# full sample:
cargo bench --bench adapters
```

Record environment in every **wall-time** result row:

- CPU model / GPU model
- `rustc` / candle versions
- Python `peft` / `torch` versions (if comparing)
- Date and git SHA

---

## Result table (performance summary)

| Method | peft-rs | HF peft | Notes | Date / SHA |
|--------|---------|---------|-------|------------|
| `wall_lora_fwd` b1/s128/h768/r8 | ~0.87 ms | — | CPU F32 criterion --quick | 2026-07-22 / 5f01624 |
| `wall_lora_fwd` b4/s128/h768/r8 | ~3.04 ms | — | CPU F32 | 2026-07-22 |
| `wall_lora_merge` h768/r8 | ~2.56 ms | — | CPU F32 | 2026-07-22 |
| `wall_dora_fwd` | — | — | not yet measured | — |
| `rss_lora_layer` | — | — | not yet measured | — |
| `correctness_lora_fwd` | pass @ 1e-5 | math-equivalent fixture | PR-042 | 2026-07-22 |
| `correctness_lora_merge` | pass @ 1e-5 | math-equivalent fixture | PR-042 | 2026-07-22 |

---

## Showcase bar (target, not current claim)

Later releases aim for **parity+** on layer math (correctness first) and competitive wall time / RSS on fair fixtures. **1.1.0** adds CPU LoRA criterion baselines; cross-runtime outperform metrics remain open (PEFT-P2-03).

---

## Related gaps

| Gap | Topic | Status after 1.1.0 |
|-----|--------|--------------------|
| PEFT-P0-03 | This file | closed (scaffold + correctness + CPU benches) |
| PEFT-P0-10 | Numerical parity suite | **closed** |
| PEFT-P0-12 | Minimal train step | **closed** |
| PEFT-P1-01 | Weighted multi-adapter | **closed** |
| PEFT-P1-02 | AdaLoRA top-k budget | **closed** |
| PEFT-P1-03 | Prefix/prompt experimental path | **closed** |
| PEFT-P1-04 | Quantization bridge trait | **closed** |
| PEFT-P2-02 | Real criterion benches | **closed** |
| PEFT-P2-03 | Showcase outperform metrics vs HF | open |
