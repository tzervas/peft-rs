#!/usr/bin/env bash
# Pre-commit quality checks for peft-rs
# Install: cp scripts/pre-commit.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit

set -e

echo "Running pre-commit quality checks..."

# Check if nvcc is available to decide if CUDA feature should be tested
if command -v nvcc &> /dev/null; then
    echo "CUDA compiler (nvcc) found. Enabling all features (including CUDA)."
    FEATURES_FLAG="--all-features"
else
    echo "CUDA compiler (nvcc) not found. Running in CPU-only mode (excluding CUDA)."
    FEATURES_FLAG=""
fi
echo ""

# 1. Format check
echo "1. Checking code formatting..."
if ! cargo fmt -- --check; then
    echo "❌ Code formatting check failed. Run 'cargo fmt' to fix."
    exit 1
fi
echo "✅ Code formatting passed"

# 2. Clippy check
echo "2. Running clippy..."
if ! cargo clippy --all-targets $FEATURES_FLAG -- -D warnings; then
    echo "❌ Clippy check failed. Fix warnings before committing."
    exit 1
fi
echo "✅ Clippy passed"

# 3. Test suite
echo "3. Running test suite..."
if ! cargo test $FEATURES_FLAG; then
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
    echo "⚠️  cargo-audit not installed. Run: cargo install cargo-audit"
fi

echo ""
echo "✅ All pre-commit checks passed!"
echo ""
