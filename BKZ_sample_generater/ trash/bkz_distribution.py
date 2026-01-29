import numpy as np
from fpylll import IntegerMatrix, BKZ
import matplotlib.pyplot as plt

def generate_uniform_A(N, Q):
    """Uniform random A in [0, Q)."""
    return np.random.randint(0, Q, size=(N, N))

def bkz_reduce_A(A, beta, max_loops=3):
    """Apply BKZ reduction (block size = beta)."""
    B = IntegerMatrix.from_matrix(A.tolist())

    bkz = BKZ(B)
    par = BKZ.Param(
        block_size=beta,
        strategies=BKZ.DEFAULT_STRATEGY,
        max_loops=max_loops,
    )
    bkz(par)

    return np.array([[B[i, j] for j in range(A.shape[1])] for i in range(A.shape[0])])

def get_entry_distribution(A_reduced, Q, bins=20):
    """Return histogram counts of entries mod Q."""
    flat = A_reduced.flatten() % Q
    hist, edges = np.histogram(flat, bins=bins, range=(0, Q))
    return hist, edges

def plot_distribution(hist, edges, title):
    plt.bar((edges[:-1] + edges[1:]) / 2, hist, width=edges[1] - edges[0])
    plt.title(title)
    plt.xlabel("Entry values")
    plt.ylabel("Number of entries")
    plt.show()

def generate_and_plot(N=128, Q=251, beta=20, max_loops=3):
    """Main function to generate A, reduce it, and plot histogram."""
    print(f"Generating A with N={N}, Q={Q} ...")
    A = generate_uniform_A(N, Q)

    print(f"Running BKZ with beta={beta}, max_loops={max_loops} ...")
    A_reduced = bkz_reduce_A(A, beta, max_loops)

    print("Computing histogram...")
    hist, edges = get_entry_distribution(A_reduced, Q)

    print("Plotting result ...")
    plot_distribution(hist, edges, f"BKZ(beta={beta})")

    return hist, edges, A_reduced
