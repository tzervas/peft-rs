# Fleet standards (tzervas)

Applied from the workstation pack under `plans/fleet-standards/pack/`.

## Workflows

| Workflow | When | Runner |
|----------|------|--------|
| `fleet-ci.yml` | push/PR to main|dev | thin caller of `tzervas/ap-workflows/.github/workflows/reusable-ci-autodetect.yml@v0.1` on `[self-hosted, linux, x64, podman, rust]` |
| `fleet-security.yml` | push/PR + weekly | same labels (`rust` selects the fleet work image) |
| `close-issues-on-main.yml` | PR closed→main | GitHub-hosted (API-only) |
| `reopen-issues-closed-off-main.yml` | PR merged off-main with Closes | same |

Action pins follow `tzervas/ap-workflows` `pins/actions.yml` (`actions/checkout@v7`, `astral-sh/setup-uv@v9`). Do **not** `--all-features` (pulls `cuda`).

## License

MIT (`LICENSE`). Third-party crates and Apache-2.0 inspirations: `NOTICE`.
`fleet-security.yml` job `cargo deny licenses` (GitHub-hosted) is fail-closed.
Allow-list is permissive only (no GPL / AGPL / MPL).

The `rust` `runs-on` label is the composition point for the fleet work image (`ghcr.io/tzervas/ap-workflows/runner-rust` or `ap-fleet-work-images/scribe-cpu-build` via host `GHA_IMAGE_MAP`). See [ap-workflows RUNNER-IMAGES](https://github.com/tzervas/ap-workflows/blob/main/docs/RUNNER-IMAGES.md).

## Issue close policy

- **`dev` / feature merges:** `Refs #n` only — issues stay open
- **`main` merges:** `Closes #n` / `Fixes #n`
- **Epics:** close only when delivery PR to main includes `Closes #<epic>`

## Badges

README badges use GitHub Actions SVG for **trunk** branch — live status, not static green.

## Copilot

Automatic Copilot code reviews are **disabled** for fleet-managed repos. Do not request Copilot on PRs.

## Gitleaks / gitignore

- **Local pre-commit is the real gate.** `bash scripts/install-hooks.sh` sets
  `core.hooksPath=.githooks`. That hook runs `gitleaks protect --staged`
  (`scripts/gitleaks-staged.sh`). Missing gitleaks **fails the commit** (not a
  skip). A finding in staged files: unstage it. A finding that already hit a
  remote: **rotate the credential** — rewriting history does not un-leak it.
  `git commit --no-verify` is how secrets land in git.
- `fleet-security.yml` is defense-in-depth after push. It **must** pass
  `--config .gitleaks.toml`.
- `.gitignore` must cover `/target/`, `.env*`, keys/PEMs, `.cargo/config.toml` (local path overrides), and `*.crate`.
- Libraries gitignore `Cargo.lock`; the axolotl binary crate **tracks** `Cargo.lock`.

## Permissions

Workflows use minimum `permissions:` blocks (contents read; issues write only for close/reopen jobs).
