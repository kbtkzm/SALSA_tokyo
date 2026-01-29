import numpy as np
from fpylll import IntegerMatrix, LLL, BKZ, GSO
import matplotlib.pyplot as plt
import os

# スクリプトの場所を基準に保存する
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
out_A = os.path.join(BASE_DIR, "A_reduced.npy")
out_hist = os.path.join(BASE_DIR, "histogram.png")

N = 256
Q = 251

# 一様分布からAを生成
A = np.random.randint(0, Q, size=(N, N))
B = IntegerMatrix.from_matrix(A.tolist())

# GSO + LLL のセットアップ
M = GSO.Mat(B, float_type="double", update=True)
L = LLL.Reduction(M)

# BKZ パラメータ
par = BKZ.Param(
    block_size=30,   # β
    max_loops=100000000,     # 何周させるか
    delta=0.99       # LLL の δ
)

# BKZ 実行
BKZ.Reduction(M, L, par)()

# BKZ後の行列を numpy に変換
A_red = np.zeros((N, N), dtype=int)
for i in range(N):
    for j in range(N):
        A_red[i, j] = int(B[i, j]) % Q

# BKZ後の行列を保存
np.save(out_A, A_red)
print("Saved BKZ-reduced matrix to", out_A)

# ヒストグラム生成
flat = A_red.flatten()
plt.hist(flat, bins=10)
plt.title("Histogram of BKZ-reduced matrix entries")
plt.xlabel("Value")
plt.ylabel("Frequency")

# ヒストグラム保存
plt.savefig(out_hist)
print("Saved histogram to", out_hist)
