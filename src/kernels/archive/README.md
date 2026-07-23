# Archived CubeCL kernels (not compiled)

These files (`lora.rs`, `dora.rs`) are **historical reference** from the
crates.io / tag `v1.0.3` CubeCL fused-kernel experiment.

- **Not** compiled by default or with `features = ["cuda"]`
- **Not** part of the public API
- Require CubeCL deps that are intentionally absent from this SoT tree
- Do not edit expecting runtime effect

See parent `src/kernels/mod.rs` and crate README for the quarantine policy
(PR-021 / PEFT-P0-05).
