# LoRA parity fixtures (PR-042)

Checked-in tensors for **CPU** LoRA forward + merge correctness.

## Fixture

| File | Contents |
|------|----------|
| `fixtures/lora_fwd_merge.json` | Fixed A, B, W, x + golden `forward_output`, `merged_weight`, `merged_forward_output` |

### Math (matches HuggingFace PEFT Linear LoRA)

```
y = x @ W.T + (x @ A.T @ B.T) * (alpha / r)
W_merged = W + (B @ A) * (alpha / r)
```

Weight layout (candle `Linear` / PEFT):

- `A`: `[r, in_features]`
- `B`: `[out_features, r]`
- `W`: `[out_features, in_features]`

### Tolerances

- `atol = 1e-5`
- `rtol = 1e-5`

Documented again in crate root `METRICS.md` (`correctness_lora_*`).

## Run (CI / local)

```bash
cargo test --test parity_lora
cargo test --lib
cargo test --tests
```

Python is **not** required.

## Regenerate offline

```bash
python3 scripts/gen_lora_parity_fixture.py \
  --out tests/parity/fixtures/lora_fwd_merge.json
```

Optional verification against installed `peft` + `torch` (not used in CI):

```bash
python3 scripts/gen_lora_parity_fixture.py --verify-peft --out /tmp/check.json
```

If `--verify-peft` is set and imports fail, the script exits non-zero after still writing pure-math goldens.
