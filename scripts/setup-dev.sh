#!/usr/bin/env bash
# Quick setup script for peft-rs development environment

set -e

echo "Setting up peft-rs development environment..."
echo ""

# Ensure ~/.cargo/bin is in PATH for the setup session
if [[ ":$PATH:" != *":$HOME/.cargo/bin:"* ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Install required cargo tools
echo "Installing cargo tools..."
if ! command -v cargo-audit &> /dev/null; then
    echo "  Fetching precompiled statically linked musl binary of cargo-audit from GitHub..."
    ver="0.22.2"
    asset="cargo-audit-x86_64-unknown-linux-gnu-v${ver}.tgz"
    url="https://github.com/rustsec/rustsec/releases/download/cargo-audit/v${ver}/${asset}"
    mkdir -p "${HOME}/.cargo/bin"
    tmp="$(mktemp -d)"
    if curl -fsSL "$url" | tar -xz -C "$tmp"; then
        bin="$(find "$tmp" -type f -name cargo-audit | head -n1)"
        install -m 0755 "$bin" "${HOME}/.cargo/bin/cargo-audit"
        echo "  ✅ Installed cargo-audit to ${HOME}/.cargo/bin"
    else
        echo "  ⚠️  Failed to fetch precompiled cargo-audit. Falling back to cargo install..."
        cargo install cargo-audit --locked
    fi
    rm -rf "$tmp"
else
    echo "  ✅ cargo-audit is already installed"
fi

if ! command -v cargo-tarpaulin &> /dev/null; then
    echo "  Installing cargo-tarpaulin (for coverage)..."
    cargo install cargo-tarpaulin --locked
fi

if ! command -v cargo-outdated &> /dev/null; then
    echo "  Installing cargo-outdated..."
    cargo install cargo-outdated --locked
fi

echo ""
echo "Setting up git hooks..."
if [ -f scripts/pre-commit.sh ]; then
    cp scripts/pre-commit.sh .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit
    echo "  ✅ Pre-commit hook installed"
else
    echo "  ⚠️  Pre-commit script not found"
fi

echo ""
echo "Making scripts executable..."
chmod +x scripts/*.sh

echo ""
echo "Running initial quality check..."
if bash scripts/quality-check.sh; then
    echo ""
    echo "✅ Development environment setup complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Review docs/DEVELOPMENT_PLAN.md for project roadmap"
    echo "  2. Check docs/TASK_TRACKER.md for current tasks"
    echo "  3. Create a working branch: git checkout -b working/your-feature"
    echo "  4. Run 'bash scripts/quality-check.sh' before creating PRs"
else
    echo ""
    echo "⚠️  Initial quality check found issues"
    echo "    This is expected if you're starting fresh"
    echo "    Review the output above and fix as needed"
fi
