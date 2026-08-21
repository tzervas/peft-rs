# Versioning and releases

## Source of truth

The live crate version is **`Cargo.toml` `package.version`**.
`.cz.toml` `tool.commitizen.version` must match it.

```bash
cz version --project
```

This file is **not** a pin. Do not copy a `x.y.z` from here into `Cargo.toml`.
When you claim a version shipped, say *where* (git tag vs crates.io). A GitHub
Release is not a registry publication.

## This repo does not use `major_version_zero`

Fleet 0.x repos set `major_version_zero = true` so breaking changes stay MINOR.
This crate is published on crates.io at **1.x**. That key does not stop applying
after 1.0 — it pins major forever and demotes BREAKING to MINOR.

A dependent on `peft-rs = "1"` accepts `>=1.0.0, <2.0.0`. A breaking change
shipped as MINOR would land silently in every consumer rebuild. Do not add the
key. `.cz.toml` carries the same warning.

```
BREAKING, major_version_zero = true   ->  1.2.10 -> 1.3.0   (MINOR)
BREAKING, major_version_zero absent   ->  1.2.10 -> 2.0.0   (MAJOR)
```

## At 1.x, MAJOR is the breaking position

| Change                        | Bump      | Example           |
| ----------------------------- | --------- | ----------------- |
| `fix:`                        | PATCH     | 1.2.0 → 1.2.1     |
| `feat:`                       | MINOR     | 1.2.0 → 1.3.0     |
| `feat!:` / `BREAKING CHANGE:` | **MAJOR** | 1.2.0 → **2.0.0** |

Examples are illustrative. Consumers pin the **major**: `peft-rs = "1"`.
Do not pin an exact patch in install examples.

`cz bump` will compute `2.0.0` from `feat!:`. **No agent may cut 2.0.0.**
A major bump against a published package is a maintainer decision.

## Version files

[`.cz.toml`](../.cz.toml) `version_files` move together via `cz bump`:

- `Cargo.toml`
- `.cz.toml` `version`

Do not hand-edit those.

```bash
cz bump --yes --dry-run              # show the next bump, change nothing
cz bump --yes --increment patch --files-only
```

`--files-only` updates files without a local tag. Do **not** pass
`--changelog` (commitizen rewrites Keep a Changelog). Move
`## [Unreleased]` by hand. Tags come from
[`.github/workflows/release.yml`](../.github/workflows/release.yml) after the
bump PR merges. `main` takes PRs — do not `cz bump` a tag straight onto `main`.

## Release steps

1. Land work on `main` via PR (conventional commits). `dev` is not the gate.
2. On a `release/*` branch: `cz bump --files-only` (or the release workflow
   `bump=patch|minor`). Open that PR to `main`.
3. After merge, tag with the release workflow (`bump=none`) so the tag matches
   `Cargo.toml`.
4. crates.io publish is a **separate** step (`CARGO_REGISTRY_TOKEN`). Until
   that runs, consumers still resolve whatever the registry last accepted.
