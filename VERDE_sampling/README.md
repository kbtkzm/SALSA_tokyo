# VERDE sampling (BKZ pre-processing)

This folder contains helper scripts to generate original (A, b) samples from a stored secret and then BKZ-reduce each sample.

Design assumptions based on the current discussion:
- Each dimension folder (e.g. `n=80`) contains a `secret.npy` with **one** secret vector `s` of shape `(n,)`.
- By default, each sample uses a random vector `a` to build a **circulant** matrix `A = circulant(a)`, then `b = A·s + e (mod q)` (uniform-style).
- BKZ reduction is applied to each `A`, producing a transformation matrix `R`.
- Two output modes:
  - `vector_pool` (Option B): extract each row of `A'` as `a'`, recompute `b' = circulant(a')·s + e`.
    Saves `A_reduced.npy` as `(num_samples * n, n)` and `b_reduced.npy` as `(num_samples * n, n)`.
  - `matrix_pair` (VERDE-style): save `A' = R·A` and `b' = R·b`.
    Saves `A_reduced.npy` as `(num_samples, n, n)` and `b_reduced.npy` as `(num_samples, n)`.

Scripts
- `generate_original_samples.py`: generate original `A_original.npy` and `b_original.npy` from `secret.npy`.
- `reduce_samples_bkz.py`: BKZ-reduce `A_original.npy` and export reduced data using `--output_mode`.

Typical usage (per dimension folder):
```
python3 generate_original_samples.py --out_dir /home/kubota/SALSA_tokyo/VERDE_sampling/n=80 --n 80 --q 251 --sigma 3 --num_samples 10000
python3 reduce_samples_bkz.py --out_dir /home/kubota/SALSA_tokyo/VERDE_sampling/n=80 --n 80 --q 251 --lll_penalty 15 --beta 5 --delta 0.99 --max_time 60
```

```

Note: these scripts require `numpy` and `fpylll` in the Python environment used to run them.
