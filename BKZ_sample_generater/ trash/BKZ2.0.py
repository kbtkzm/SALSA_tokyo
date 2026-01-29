import numpy as np
from fpylll import IntegerMatrix, LLL, GSO, BKZ
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
import matplotlib.pyplot as plt
import os


# ================================
# パラメータ
# ================================
N = 150        # dimension
Q = 251        # modulus
beta = 20      # BKZ block size (PICANTEの図に対応)
delta = 0.99   # LLL reduction parameter
max_time = 60  # BKZ2.0 の1ループの最大秒数
# ================================

# 保存ファイル
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
out_A = os.path.join(BASE_DIR, "A_reduced.npy")
out_hist = os.path.join(BASE_DIR, "histogram.png")


# ================================
# Step 1 : 一様分布で A を生成
# ================================
A = np.random.randint(0, Q, size=(N, N)).astype(int)


# ================================
# Step 2 : PICANTE と同じ embedding 行列 Ap を生成
#   Ap = [[ Q*I_N , 0     ],
#         [ I_N   ,  A    ]]
# サイズは (2N × 2N)
# ================================
Ap = np.zeros((2*N, 2*N), dtype=int)

# 上左 = Q * I
Ap[:N, :N] = Q * np.eye(N, dtype=int)

# 下左 = I
Ap[N:, :N] = np.eye(N, dtype=int)

# 下右 = A
Ap[N:, N:] = A


# ================================
# Step 3 : fpylll の IntegerMatrix に変換
# ================================
B = IntegerMatrix.from_matrix(Ap.tolist())
M = GSO.Mat(B, float_type="double", update=True)
L = LLL.Reduction(M)


# ================================
# Step 4 : BKZ2.0 をセットアップ
# ================================
params = BKZ.Param(
    block_size=beta,
    max_time=max_time,
    delta=delta
)

bkz = BKZ2(M)  # BKZ 2.0 インスタンス


# ================================
# Step 5 : BKZ 実行
# ================================
print("Running BKZ2.0... this may take a while.")
bkz(params)  # これが PICANTE と同じ動きをする
print("BKZ completed.")


# ================================
# Step 6 : BKZ 後の行列を numpy に戻す
# ================================
A_after = np.zeros((2*N, 2*N), dtype=int)
for i in range(2*N):
    for j in range(2*N):
        A_after[i, j] = int(B[i, j])


# ================================
# Step 7 : 下右の N×N ブロックだけ取り出す（= 簡約された A'）
# ================================
A_reduced = A_after[N:, N:] % Q

# 保存
np.save(out_A, A_reduced)
print("Saved BKZ-reduced matrix to", out_A)


# ================================
# Step 8 : ヒストグラムを生成
# ================================
flat = A_reduced.flatten()
plt.hist(flat, bins=10)
plt.title("Histogram of BKZ2-reduced matrix entries")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.savefig(out_hist)
print("Saved histogram to", out_hist)
