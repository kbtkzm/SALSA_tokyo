import numpy as np
import json
from fpylll import IntegerMatrix, BKZ
import os
import matplotlib.pyplot as plt

SAVE_DIR = "rf_distributions"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------
# 1. 一様 A を生成
# -----------------------
def generate_uniform_A(N, Q):
    return np.random.randint(0, Q, size=(N, N))

# -----------------------
# 2. BKZ で簡約
# -----------------------
def bkz_reduce(A, beta, loops=3):
    B = IntegerMatrix.from_matrix(A.tolist())
    bkz = BKZ(B)
    par = BKZ.Param(
        block_size=beta,
        max_loops=loops,
        strategies=BKZ.DEFAULT_STRATEGY
    )
    bkz(par)
    return np.array([[B[i, j] for j in range(A.shape[1])] for i in range(A.shape[0])])

# -----------------------
# 3. エントリ分布を作成
# -----------------------
def compute_histogram(A, Q, bins=Q):
    flat = A.flatten() % Q
    hist, edges = np.histogram(flat, bins=bins, range=(0, Q))
    return hist.tolist(), edges.tolist()

# -----------------------
# 4. JSON 形式で保存
# -----------------------
def save_distribution(hist, edges, rf, beta):
    path = f"{SAVE_DIR}/rf_{rf:.2f}_beta{beta}.json"
    with open(path, "w") as f:
        json.dump({
            "rf": rf,
            "beta": beta,
            "hist": hist,
            "edges": edges
        }, f, indent=2)
    print(f"Saved distribution for RF={rf} → {path}")

# -----------------------
# 5. メイン関数
# -----------------------
def generate_rf_distribution(N, Q, rf, beta, loops=3):
    print(f"Generating BKZ distribution: RF={rf}, beta={beta}")

    A = generate_uniform_A(N, Q)
    A_reduced = bkz_reduce(A, beta, loops)

    hist, edges = compute_histogram(A_reduced, Q)

    save_distribution(hist, edges, rf, beta)

    return hist, edges


# Example usage
if __name__ == "__main__":

    N = 128
    Q = 251

    RF_to_beta = {
        1.00: 0,      # no reduction
        0.96: 16,
        0.95: 18,
        0.94: 20,
        0.90: 25,
        0.85: 30,
        0.80: 35
    }

    for rf, beta in RF_to_beta.items():
        if beta == 0:
            print("RF=1.00 → uniform distribution (no BKZ)")
            # 一様分布は自明なので JSON だけ保存
            hist = [1] * Q
            edges = list(range(Q + 1))
            save_distribution(hist, edges, rf, beta)
        else:
            generate_rf_distribution(N, Q, rf, beta)