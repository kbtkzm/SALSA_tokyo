#!/usr/bin/env python3
"""
Generate original (A, b) samples from a stored secret.

- secret.npy must exist in --out_dir and contain one secret vector of shape (n,).
- By default, generates circulant A from a vector a (uniform over Z_q^n),
  then computes b = A @ s + e (mod q). This matches VERDE-style sampling.
- Optional legacy mode (--circulant false) keeps the 4n-by-n sampling path.
- Saves A_original.npy (num_samples, n, n) and b_original.npy (num_samples, n).
"""

import argparse
import os
import numpy as np
from scipy.linalg import circulant


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True, help="Directory containing secret.npy and outputs")
    p.add_argument("--n", type=int, required=True, help="dimension n")
    p.add_argument("--q", type=int, default=251, help="modulus q")
    p.add_argument("--sigma", type=float, default=3.0, help="stddev for gaussian error")
    p.add_argument("--num_samples", type=int, default=10000, help="number of samples to generate")
    p.add_argument("--circulant", type=lambda x: x.lower() in {"1","true","yes"}, default=True,
                   help="generate circulant A from a (default: true).")
    p.add_argument("--oversample", type=int, default=4, help="A_big rows multiplier (legacy path)")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    secret_path = os.path.join(out_dir, "secret.npy")
    if not os.path.isfile(secret_path):
        raise FileNotFoundError(f"secret.npy not found at {secret_path}")

    s = np.load(secret_path)
    s = np.array(s, dtype=np.int64).reshape(-1)
    if s.shape[0] != args.n:
        raise ValueError(f"secret.npy has shape {s.shape}, expected ({args.n},)")

    rng = np.random.default_rng(args.seed)
    A_all = np.zeros((args.num_samples, args.n, args.n), dtype=np.int64)
    b_all = np.zeros((args.num_samples, args.n), dtype=np.int64)

    for i in range(args.num_samples):
        if args.circulant:
            a = rng.integers(0, args.q, size=args.n, dtype=np.int64)
            A = circulant(a)
            tri = np.triu_indices(args.n, 1)
            A[tri] *= -1
            A = A % args.q
            e = rng.normal(0, args.sigma, size=(args.n,)).round().astype(np.int64)
            b = (A @ s + e) % args.q
        else:
            m_big = args.oversample * args.n
            A_big = rng.integers(0, args.q, size=(m_big, args.n), dtype=np.int64)
            e_big = rng.normal(0, args.sigma, size=(m_big,)).round().astype(np.int64)
            b_big = (A_big @ s + e_big) % args.q
            idx = rng.choice(m_big, size=args.n, replace=False)
            A = A_big[idx]
            b = b_big[idx]

        A_all[i] = A
        b_all[i] = b

        if (i + 1) % 500 == 0:
            print(f"Generated {i+1}/{args.num_samples} samples")

    np.save(os.path.join(out_dir, "A_original.npy"), A_all)
    np.save(os.path.join(out_dir, "b_original.npy"), b_all)
    print("Saved:", os.path.join(out_dir, "A_original.npy"))
    print("Saved:", os.path.join(out_dir, "b_original.npy"))


if __name__ == "__main__":
    main()
