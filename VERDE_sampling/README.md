# VERDE sampling (BKZ pre-processing)

This folder contains helper scripts to generate original (A, b) samples from a stored secret and then BKZ-reduce each sample to produce reduced vectors (Option B).

Design assumptions based on the current discussion:
- Each dimension folder (e.g. `n=80`) contains a `secret.npy` with **one** secret vector `s` of shape `(n,)`.
- By default, each sample uses a random vector `a` to build a **circulant** matrix `A = circulant(a)`, then `b = A·s + e (mod q)` (uniform-style).
- BKZ reduction is applied to each `A`, producing a transformation matrix `R`.
- We compute `A' = R·A (mod q)`, then extract each row of `A'` as a reduced vector `a'`.
- For each `a'`, we recompute `b' = circulant(a')·s + e (mod q)` (Option B).
- We save `A_reduced.npy` as a vector pool (shape: `num_samples * n, n`) and `b_reduced.npy` as paired vectors.

Scripts
- `generate_original_samples.py`: generate original `A_original.npy` and `b_original.npy` from `secret.npy`.
- `reduce_samples_bkz.py`: BKZ-reduce `A_original.npy` and export `A_reduced.npy` / `b_reduced.npy` as vector pools (Option B).

Typical usage (per dimension folder):
```
python3 generate_original_samples.py --out_dir /home/kubota/SALSA_tokyo/VERDE_sampling/n=80 --n 80 --q 251 --sigma 3 --num_samples 10000
python3 reduce_samples_bkz.py --out_dir /home/kubota/SALSA_tokyo/VERDE_sampling/n=80 --n 80 --q 251 --lll_penalty 15 --beta 5 --delta 0.99 --max_time 60
```

```

Note: these scripts require `numpy` and `fpylll` in the Python environment used to run them.
