import numpy as np
from fpylll import IntegerMatrix, GSO, LLL, BKZ
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
import matplotlib.pyplot as plt
import os


# ================================
# パラメータ（VERDE と同じ構造）
# ================================
N = 30
Q = 251
m = N  # VERDE の RA ステップと同じ

lll_penalty = 1   # VERDE では params.lll_penalty
beta = 20         # block size (ここを変えるとRFが変わる)
delta = 0.99      # LLL reduction parameter
MAX_TIME = 60     # BKZ2 の1ループ最大時間
# ================================


# 保存ファイル
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
out_A = os.path.join(BASE_DIR, "A_reduced.npy")
out_hist = os.path.join(BASE_DIR, "histogram.png")


# -------------------------------
# Step 1: ランダム行列 A
# -------------------------------
A = np.random.randint(0, Q, size=(m, N))


# -------------------------------
# Step 2: VERDE の embedding Ap を構築
#
# Ap =
#   [ Q·I      0 ]
#   [ penalty·I  A ]
# -------------------------------
Ap = np.zeros((m+N, m+N), dtype=int)

# 上ブロック: Q·I
Ap[:N, N:] = Q * np.eye(N, dtype=int)

# 下ブロック: penalty·I と A
Ap[N:, :m] = lll_penalty * np.eye(m, dtype=int)
Ap[N:, m:] = A


# -------------------------------
# Step 3: fpylll で BKZ2 実行
# -------------------------------
M = IntegerMatrix.from_matrix(Ap.tolist())
G = GSO.Mat(M, float_type="double", update=True)
L = LLL.Reduction(G)

bkz = BKZ2(G)
params = BKZ.Param(block_size=beta, delta=delta, max_time=MAX_TIME)

print("Running BKZ2 ...")
bkz(params)
print("BKZ2 done.")


# -------------------------------
# Step 4: BKZ後の Ap を numpy に戻す
# -------------------------------
Ap_after = np.zeros((m+N, m+N), dtype=int)
for i in range(m+N):
    for j in range(m+N):
        Ap_after[i, j] = int(M[i, j])


# -------------------------------
# Step 5: A'（下右ブロック）を取得
# -------------------------------
A_reduced = Ap_after[N:, m:] % Q

np.save(out_A, A_reduced)
print("Saved:", out_A)


# -------------------------------
# Step 6: ヒストグラム（これが U 字になる）
# -------------------------------
flat = A_reduced.flatten()

plt.hist(flat, bins=20)
plt.title(f"BKZ2.0-reduced entries (beta={beta}),(delta={delta})")
plt.xlabel("Entry value")
plt.ylabel("Number of entries")
plt.savefig(out_hist)
print("Saved:", out_hist)