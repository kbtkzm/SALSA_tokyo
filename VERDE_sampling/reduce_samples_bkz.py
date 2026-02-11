#!/usr/bin/env python3
"""
BKZ-reduce original samples and export reduced vectors for circulant training (Option B).

Input:
- A_original.npy (num_samples, n, n)
- b_original.npy (num_samples, n)  [not used for B, kept for reference]
- secret.npy (n,)

Output:
- A_reduced.npy (num_samples * n, n)   # vector pool (rows of A')
- b_reduced.npy (num_samples * n,)     # b' recomputed from a' (circulant)

Option B:
1) Run BKZ to get A' = R @ A (mod q)
2) Extract each row of A' as a' (length n)
3) Recompute b' = circulant(a') @ s + e (mod q)
"""

import argparse
import os
import time
from datetime import datetime
import numpy as np
from scipy.linalg import circulant

from fpylll import IntegerMatrix, GSO, LLL, BKZ, FPLLL
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True, help="Directory containing A_original.npy / b_original.npy")
    p.add_argument("--n", type=int, required=True, help="dimension n")
    p.add_argument("--q", type=int, default=251, help="modulus q")
    p.add_argument("--secret_path", type=str, default="", help="Path to secret.npy (default: out_dir/secret.npy)")
    p.add_argument("--sigma", type=float, default=3.0, help="stddev for gaussian error in b'")
    p.add_argument("--lll_penalty", type=int, default=15, help="embedding penalty (w) in VERDE")
    p.add_argument("--beta", type=int, default=5, help="BKZ block size")
    p.add_argument("--delta", type=float, default=0.99, help="LLL delta")
    p.add_argument("--max_time", type=int, default=60, help="BKZ max_time per loop")
    p.add_argument("--precision", type=int, default=200, help="FPLLL precision bits")
    p.add_argument("--filter_zero_rows", action="store_true", help="drop all-zero rows in R before applying")
    p.add_argument("--rnorm", type=float, default=1.0, help="if <1, filter rows of R by norm/Q < rnorm")
    p.add_argument("--longtype", action="store_true", help="use longdouble splitting to avoid overflow")
    return p.parse_args()


def bkz_reduce_one(A, q, lll_penalty, beta, delta, max_time, precision):
    n = A.shape[0]
    # Build VERDE embedding Ap
    Ap = np.zeros((n + n, n + n), dtype=int)
    Ap[:n, n:] = q * np.eye(n, dtype=int)
    Ap[n:, :n] = lll_penalty * np.eye(n, dtype=int)
    Ap[n:, n:] = A

    FPLLL.set_precision(precision)
    M = IntegerMatrix.from_matrix(Ap.tolist())
    G = GSO.Mat(M, float_type="mpfr", update=True)
    L = LLL.Reduction(G, delta=delta)
    L()

    bkz = BKZ2(G)
    flags = BKZ.AUTO_ABORT | BKZ.MAX_TIME
    params = BKZ.Param(block_size=beta, delta=delta, max_time=max_time, flags=flags)
    bkz(params)

    Ap_after = np.zeros((n + n, n + n), dtype=int)
    for i in range(n + n):
        for j in range(n + n):
            Ap_after[i, j] = int(M[i, j])

    R = (Ap_after[:, :n] // lll_penalty).astype(int)
    return R


def apply_R(R, A, b, q, longtype=False):
    if longtype:
        R = R.astype(np.longdouble)
        A = A.astype(np.int64)
        b = b.astype(np.int64)
        RA = ((R // 10000) @ (A * 10000 % q) + (R % 10000) @ A) % q
        Rb = ((R // 10000) @ (b * 10000 % q) + (R % 10000) @ b) % q
    else:
        RA = (R @ A) % q
        Rb = (R @ b) % q
    return RA, Rb


def recompute_b_from_a(a_vec, s, q, sigma, rng):
    A = circulant(a_vec)
    tri = np.triu_indices(A.shape[0], 1)
    A[tri] *= -1
    A = A % q
    e = rng.normal(0, sigma, size=(A.shape[0],)).round().astype(np.int64)
    b = (A @ s + e) % q
    return b


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    timing_path = os.path.join(out_dir, "bkz_timing.log")

    A_path = os.path.join(out_dir, "A_original.npy")
    b_path = os.path.join(out_dir, "b_original.npy")
    if not os.path.isfile(A_path) or not os.path.isfile(b_path):
        raise FileNotFoundError("A_original.npy or b_original.npy not found in out_dir")

    A_all = np.load(A_path)
    b_all = np.load(b_path)

    if A_all.ndim != 3 or A_all.shape[1:] != (args.n, args.n):
        raise ValueError(f"A_original.npy shape {A_all.shape} is invalid for n={args.n}")
    if b_all.ndim != 2 or b_all.shape[1] != args.n:
        raise ValueError(f"b_original.npy shape {b_all.shape} is invalid for n={args.n}")

    num_samples = A_all.shape[0]
    # vector pool size = num_samples * n
    A_red = np.zeros((num_samples * args.n, args.n), dtype=np.int64)
    b_red = np.zeros((num_samples * args.n, args.n), dtype=np.int64)

    secret_path = args.secret_path or os.path.join(out_dir, "secret.npy")
    if not os.path.isfile(secret_path):
        raise FileNotFoundError(f"secret.npy not found at {secret_path}")
    s = np.load(secret_path).reshape(-1).astype(np.int64)
    if s.shape[0] != args.n:
        raise ValueError(f"secret.npy shape {s.shape} is invalid for n={args.n}")
    rng = np.random.default_rng(0)

    def log_timing(elapsed_seconds: float):
        ts = datetime.now().strftime("%m/%d/%y %H:%M:%S")
        elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))
        with open(timing_path, "a", encoding="utf-8") as f:
            f.write(f"INFO - {ts} - {elapsed}\n")

    log_timing(0.0)
    batch_start = time.perf_counter()

    for i in range(num_samples):
        A = A_all[i]
        b = b_all[i]

        R = bkz_reduce_one(A, args.q, args.lll_penalty, args.beta, args.delta, args.max_time, args.precision)

        if args.filter_zero_rows:
            R = R[[i for i in range(len(R)) if set(R[i]) != {0}]]
        if args.rnorm != 1:
            R = R[[i for i in range(len(R)) if np.linalg.norm(R[i]) / args.q < args.rnorm]]

        RA, Rb = apply_R(R, A, b, args.q, longtype=args.longtype)

        # If R filtering changes row count, keep only first n rows to keep shape consistent.
        if RA.shape[0] != args.n:
            RA = RA[: args.n]
            Rb = Rb[: args.n]

        # Option B: extract rows of RA as a' and recompute b' from circulant(a')
        for j in range(args.n):
            idx = i * args.n + j
            a_vec = RA[j].astype(np.int64) % args.q
            A_red[idx] = a_vec
            b_vec = recompute_b_from_a(a_vec, s, args.q, args.sigma, rng)
            b_red[idx] = b_vec

        if (i + 1) % 100 == 0:
            print(f"Reduced {i+1}/{num_samples} samples")
        if (i + 1) % 1000 == 0:
            elapsed = time.perf_counter() - batch_start
            log_timing(elapsed)
            batch_start = time.perf_counter()

    total_elapsed = time.perf_counter() - batch_start
    log_timing(total_elapsed)

    np.save(os.path.join(out_dir, "A_reduced.npy"), A_red)
    np.save(os.path.join(out_dir, "b_reduced.npy"), b_red)
    print("Saved:", os.path.join(out_dir, "A_reduced.npy"))
    print("Saved:", os.path.join(out_dir, "b_reduced.npy"))


if __name__ == "__main__":
    main()
