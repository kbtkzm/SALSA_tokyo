import numpy as np

# ファイルを読み込み
A = np.load("/home/kubota/SALSA/BKZ_sample_generater/A_reduced.npy")

Q = 251

# --- NoMod (−Q/2, Q/2] に変換 ---
A_nomod = A.copy()
A_nomod[A_nomod > Q//2] -= Q   # Q/2=125 の場合 126〜250 を負側に移動

# --- reduction factor を計算 ---
std = np.std(A_nomod)        # 標準偏差（全要素）
RF = np.sqrt(12) * std / Q   # VERDE / PICANTE と同じ定義

print("std =", std)
print("Reduction Factor (RF) =", RF)