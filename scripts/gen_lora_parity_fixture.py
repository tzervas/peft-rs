#!/usr/bin/env python3
"""Generate LoRA forward/merge parity fixtures for peft-rs (PR-042).

Default path uses pure Python math matching HF peft Linear LoRA:
  y = x @ W.T + (x @ A.T @ B.T) * (alpha / r)
  W_merged = W + (B @ A) * (alpha / r)

Optional --verify-peft compares against torch+peft if installed (offline only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def matmul(a, b):
    m, k = len(a), len(a[0])
    n = len(b[0])
    out = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for t in range(k):
                s += a[i][t] * b[t][j]
            out[i][j] = s
    return out


def transpose(a):
    return [list(row) for row in zip(*a)]


def add(a, b):
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def scale(a, s):
    return [[x * s for x in row] for row in a]


def flat(a):
    return [float(v) for row in a for v in row]


def build_fixed_matrices(in_f: int, out_f: int, r: int):
    # Deterministic, hand-specified pattern (no RNG).
    a = [[0.01 * (i * in_f + j + 1) for j in range(in_f)] for i in range(r)]
    b = [[0.02 * ((i * r + j) % 7 - 3) for j in range(r)] for i in range(out_f)]
    w = [[0.05 * ((i + j) % 5 - 2) for j in range(in_f)] for i in range(out_f)]
    x_flat = [[0.1 * ((i * in_f + j) % 9 - 4) for j in range(in_f)] for i in range(6)]
    return a, b, w, x_flat


def pure_math_fixture(in_f=8, out_f=8, r=4, alpha=8, batch=2, seq=3):
    assert batch * seq == 6
    scaling = alpha / float(r)
    a, b, w, x_flat = build_fixed_matrices(in_f, out_f, r)
    base = matmul(x_flat, transpose(w))
    tmp = matmul(x_flat, transpose(a))
    lora = scale(matmul(tmp, transpose(b)), scaling)
    out = add(base, lora)
    delta = scale(matmul(b, a), scaling)
    w_merged = add(w, delta)
    merged_out = matmul(x_flat, transpose(w_merged))
    return {
        "meta": {
            "name": "lora_fwd_merge_v1",
            "in_features": in_f,
            "out_features": out_f,
            "r": r,
            "alpha": alpha,
            "scaling": scaling,
            "batch": batch,
            "seq": seq,
            "dtype": "f32",
            "formula": "y = x@W.T + (x@A.T@B.T)*(alpha/r); W_merged = W + B@A*(alpha/r)",
            "weight_layout": "A:(r,in) B:(out,r) W:(out,in) matches candle Linear / HF peft LoRA",
            "atol": 1e-5,
            "rtol": 1e-5,
            "generator": "scripts/gen_lora_parity_fixture.py",
        },
        "lora_A": {"shape": [r, in_f], "data": flat(a)},
        "lora_B": {"shape": [out_f, r], "data": flat(b)},
        "base_weight": {"shape": [out_f, in_f], "data": flat(w)},
        "input": {"shape": [batch, seq, in_f], "data": flat(x_flat)},
        "forward_output": {"shape": [batch, seq, out_f], "data": flat(out)},
        "merged_weight": {"shape": [out_f, in_f], "data": flat(w_merged)},
        "merged_forward_output": {"shape": [batch, seq, out_f], "data": flat(merged_out)},
    }


def verify_peft(fixture: dict) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import LoraLayer as PeftLoraLayer  # type: ignore
    import torch.nn as nn

    meta = fixture["meta"]
    in_f = meta["in_features"]
    out_f = meta["out_features"]
    r = meta["r"]
    alpha = meta["alpha"]

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(in_f, out_f, bias=False)

        def forward(self, x):
            return self.proj(x)

    model = Tiny()
    w = torch.tensor(fixture["base_weight"]["data"], dtype=torch.float32).view(out_f, in_f)
    with torch.no_grad():
        model.proj.weight.copy_(w)

    # Manual LoRA path equivalent (avoid full PEFT inject complexity for tiny Linear)
    a = torch.tensor(fixture["lora_A"]["data"], dtype=torch.float32).view(r, in_f)
    b = torch.tensor(fixture["lora_B"]["data"], dtype=torch.float32).view(out_f, r)
    x = torch.tensor(fixture["input"]["data"], dtype=torch.float32).view(
        meta["batch"], meta["seq"], in_f
    )
    scaling = alpha / float(r)
    base = torch.nn.functional.linear(x, w, None)
    # PEFT Linear: lora_A weight (r, in), lora_B (out, r); F.linear(x, A) = x @ A.T
    lora = torch.nn.functional.linear(
        torch.nn.functional.linear(x, a, None), b, None
    ) * scaling
    y = base + lora
    y_ref = torch.tensor(fixture["forward_output"]["data"], dtype=torch.float32).view_as(y)
    if not torch.allclose(y, y_ref, atol=1e-5, rtol=1e-5):
        raise SystemExit(f"torch path mismatch max={(y - y_ref).abs().max().item()}")

    # Optional: peft installed check (config construct only if get_peft_model path desired)
    _ = LoraConfig(r=r, lora_alpha=alpha, target_modules=["proj"], bias="none")
    print("verify-peft: torch LoRA math allclose OK; peft import OK", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("tests/parity/fixtures/lora_fwd_merge.json"),
    )
    ap.add_argument(
        "--verify-peft",
        action="store_true",
        help="Import torch/peft and verify math (offline; not CI)",
    )
    args = ap.parse_args()
    fixture = pure_math_fixture()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {args.out}")
    if args.verify_peft:
        try:
            verify_peft(fixture)
        except ImportError as e:
            print(f"verify-peft failed import: {e}", file=sys.stderr)
            sys.exit(2)


if __name__ == "__main__":
    main()
