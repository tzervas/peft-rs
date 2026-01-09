# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-01-09

### Added
- Development plan and comprehensive roadmap (DEVELOPMENT_PLAN.md)
- CHANGELOG.md for tracking version history
- Cargo-audit integration for CVE checking
- CI/CD pipeline (.github/workflows/ci.yml)
- Quality check scripts (quality-check.sh, pre-commit.sh, setup-dev.sh)
- Testing branch for proper QA workflow
- PLAN_SUMMARY.md for quick reference

### Changed
- Version strategy documented: strict semantic versioning
- Branch strategy formalized: working → dev → testing → main

### Fixed
- All 207 clippy warnings resolved (documentation, code quality, formatting)
- Added proper error documentation to all Result-returning functions
- Fixed comparison warnings (op_ref)
- Updated format strings to inline syntax

## [0.3.0] - 2026-01-08

### Added
- Model integration system (PeftModel wrapper)
- Training utilities (LR schedules, training state)
- Multi-adapter registry with runtime switching
- Enhanced I/O with HuggingFace PEFT format compatibility
- BOFT (Butterfly Orthogonal Fine-Tuning) adapter
- VeRA (Vector-based Random Matrix Adaptation) adapter
- Comprehensive test suite (124 tests)

### Changed
- Improved model.rs with pattern matching
- Enhanced training.rs with multiple LR schedules

## [0.2.0] - 2026-01-07

### Added
- LoHa (Low-Rank Hadamard Product) adapter
- LoKr (Low-Rank Kronecker Product) adapter
- OFT (Orthogonal Fine-Tuning) adapter
- Prefix Tuning adapter
- Prompt Tuning adapter

## [0.1.0] - 2026-01-06

### Added
- Initial implementation of LoRA adapter
- DoRA (Weight-Decomposed Low-Rank Adaptation) support
- AdaLoRA (Adaptive Low-Rank Adaptation) with SVD
- IA³ (Infused Adapter by Inhibiting and Amplifying) adapter
- Core trait system (Adapter, Mergeable, Trainable, AdapterConfig)
- Error handling (PeftError enum)
- Basic I/O with safetensors
- Configuration system
- Initial documentation (README, AGENT_GUIDE, GAP_ANALYSIS, TASK_TRACKER)

[Unreleased]: https://github.com/tzervas/peft-rs/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/tzervas/peft-rs/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/tzervas/peft-rs/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/tzervas/peft-rs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tzervas/peft-rs/releases/tag/v0.1.0
