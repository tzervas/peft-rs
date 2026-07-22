# DECISION: peft-rs SoT vs crates.io 1.0.3 skew heal (PR-002)

**Date:** 2026-07-22  
**Agent:** Wave-3 G0-C  
**Gaps:** ECO-P0-03, PEFT-P0-02  
**Status:** **Accepted — intentional supersede (option B)**

---

## Problem

| Source | Version / surface |
|--------|-------------------|
| crates.io `peft-rs` | **1.0.3** (published ~2026-01-28) |
| Git tag `v1.0.3` | **1.0.3** at `1bc18f1` |
| Local tree (pre-heal) | **1.0.1** in `Cargo.toml` despite ancestry **containing** `v1.0.3` |
| Local kernels | Source under `src/kernels/` present, **not** exported from `lib.rs` |
| v1.0.3 / crates.io kernels | Optional `cubecl` / `cubecl-cuda` + `#[cfg(feature = "cuda")] pub mod kernels` |

Local TLC was **behind** crates.io on the version field while **ahead** on product work (examples, LoRA weight-loading fixes, fleet CI, MIT LICENSE). That is reverse version skew, not dual-tree divergence of two unrelated codebases.

### How the skew happened

1. `v1.0.3` was cut and published from a release line with CubeCL kernel deps and `pub mod kernels` under `cuda`.
2. Subsequent development (examples, adapter docs, workspace reconcile, fleet packs) continued on branches that still carried **`version = "1.0.1"`**.
3. Promote / merge into `main` kept the **newer code** but **regressed the package version** back to 1.0.1 and dropped CubeCL from the default `cuda` feature wiring (`cuda = ["candle-core/cuda"]` only).
4. `CHANGELOG.md` also lost the **1.0.2** and **1.0.3** historical sections (restored as part of this heal).

Evidence:

- `git merge-base v1.0.3 HEAD` == `v1.0.3` tip (`1bc18f1`); **no** commits on tag that are missing from HEAD.
- `git diff --stat v1.0.3..HEAD` is additive (examples, fleet workflows, adapter improvements) plus intentional kernel unexport / cubecl dep removal.
- Kernel files on disk at HEAD match the v1.0.3 tree shape (`src/kernels/{mod,lora,dora}.rs`); only **compilation gate / deps / lib export** differ.

---

## Options considered

### A — Restore-from-tag (re-materialize published 1.0.3 surface)

Re-add optional `cubecl` / `cubecl-cuda`, restore `cuda = […, dep:cubecl, dep:cubecl-cuda]`, and re-export `pub mod kernels` under `cuda` as at `v1.0.3`.

**Rejected for this PR:**

- CubeCL 0.9 + candle CUDA feature path is **environment-fragile** (host arch / nvcc) and not proven green on fleet CI here.
- Kernel modules are **optional GPU surface**, not the default-feature public API used by reverse deps for layer math.
- Full kernel honesty / rewire is already scheduled as **PR-021** (PEFT-P0-05 / PEFT-P1-05).
- Restoring a half-working `cuda` feature would re-lie about acceleration until PR-021 lands.

### B — Intentional supersede toward 1.0.4 honesty *(chosen)*

Treat **this tree as Source of Truth**. Bump package version **past** crates.io so local and future publish cannot be confused with the historical 1.0.3 tarball. Document kernels as **present but unwired** until PR-021. Leave default/`cargo test --lib` green.

**Chosen because:**

- RELEASE_TRAIN already targets **peft-rs 1.0.4** for honesty + flags/kernels truth.
- SoT path is `/root/work/peft-rs` (R5 / RELEASE_TRAIN); publish must not come from stale tags without this heal.
- Pragmatic: avoids a messy CubeCL restore while fixing version truth immediately.
- Downstream honesty work (PR-010 docs, PR-021 kernels) can land on a correctly numbered tree.

---

## Decision

1. **Strategy:** intentional supersede — do **not** hard-reset or cherry-pick a pure restore of tag `v1.0.3`.
2. **Version:** set `Cargo.toml` / lock to **`1.0.4`** (pre-publish SoT version for the next honesty release train).
3. **Published 1.0.3:** remains valid on crates.io as a historical artifact. No yank required for this skew alone (installs; docs overclaim is a separate honesty track).
4. **Kernels / CubeCL:** **deferred to PR-021**. Files stay in-tree as optional future surface; not exported from `lib.rs`; `cuda` feature = candle CUDA only until that PR decides rewire vs quarantine.
5. **Docs honesty matrix / METRICS.md:** **deferred to PR-010** (depends on this heal).
6. **Changelog:** restore missing **1.0.2** / **1.0.3** notes; record this skew heal under **1.0.4**.

---

## Acceptance for this PR

| Check | Result |
|-------|--------|
| Written decision | This file |
| `Cargo.toml` version strategy | **1.0.4** |
| `cargo check` | required green |
| `cargo test --lib` | required green |
| ECO-P0-03 | addressed (SoT + version ≥ crates.io intent) |
| PEFT-P0-02 | addressed (strategy recorded; not left as silent 1.0.1) |
| PEFT-P0-05 kernels | **deferred** → PR-021 |
| PEFT-P0-01 docs honesty | **deferred** → PR-010 |

---

## Non-goals (explicit)

- Do not `cargo publish` from this PR.
- Do not force-push tags or rewrite `v1.0.3`.
- Do not re-enable CubeCL in this PR.
- Do not rewrite README parity tables here (PR-010).
