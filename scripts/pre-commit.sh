#!/usr/bin/env bash
# Pre-commit quality checks for peft-rs
# Install: cp scripts/pre-commit.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit

set -e

echo "Running pre-commit quality checks..."

# Add ~/.cargo/bin to PATH if not present
if [[ ":$PATH:" != *":$HOME/.cargo/bin:"* ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Dynamically auto-detect whether nvcc is present in the environment
if command -v nvcc &> /dev/null; then
    echo "CUDA compiler (nvcc) detected. Enabling all features (including CUDA)."
    FEATURES_ARG="--all-features"
else
    echo "CUDA compiler (nvcc) not detected. Running in CPU-only mode (omitting --all-features)."
    FEATURES_ARG=""
fi

# 1. Format check
echo "1. Checking code formatting..."
if ! cargo fmt -- --check; then
    echo "❌ Code formatting check failed. Run 'cargo fmt' to fix."
    exit 1
fi
echo "✅ Code formatting passed"

# 2. Clippy check
echo "2. Running clippy..."
if ! cargo clippy --all-targets $FEATURES_ARG -- -D warnings; then
    echo "❌ Clippy check failed. Fix warnings before committing."
    exit 1
fi
echo "✅ Clippy passed"

# 3. Test suite
echo "3. Running test suite..."
if ! cargo test $FEATURES_ARG; then
    echo "❌ Tests failed. Fix failing tests before committing."
    exit 1
fi
echo "✅ Tests passed"

# 4. Security audit (warning only, doesn't block)
echo "4. Running security audit..."
if command -v cargo-audit &> /dev/null; then
    if ! cargo audit; then
        echo "⚠️  Security audit found issues. Review before pushing."
        # Don't exit, just warn
    else
        echo "✅ Security audit passed"
    fi
else
    echo "⚠️  cargo-audit not installed. Run 'bash scripts/setup-dev.sh' to install precompiled binary"
fi

echo ""
echo "✅ All pre-commit checks passed!"
echo ""
