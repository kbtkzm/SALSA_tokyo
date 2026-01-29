import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def build_title(args, npy_name, arr):
    # --title が来たらそれを最優先
    if args.title is not None:
        return args.title

    # beta/delta/n が来た分だけタイトルに入れる
    parts = [f"{npy_name}"]
    if args.n is not None:
        parts.append(f"n={args.n}")
    if args.beta is not None:
        parts.append(f"beta={args.beta}")
    if args.delta is not None:
        parts.append(f"delta={args.delta}")

    # ついでに shape/dtype も入れたいならここ（不要なら消してOK）
    parts.append(f"shape={arr.shape}, dtype={arr.dtype}")

    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Create a histogram from a .npy file.")
    parser.add_argument("--npypath", type=str, required=True, help="Path to input .npy file")
    parser.add_argument("--bins", type=int, default=20, help="Number of histogram bins")
    parser.add_argument("--out", type=str, default=None, help="Output image path (png/pdf/etc.)")

    # タイトル系
    parser.add_argument("--title", type=str, default=None, help="Plot title (overrides auto title)")
    parser.add_argument("--n", type=int, default=None, help="n to include in title")
    parser.add_argument("--beta", type=int, default=None, help="beta to include in title")
    parser.add_argument("--delta", type=float, default=None, help="delta to include in title")

    parser.add_argument("--xlabel", type=str, default="Entry value", help="X-axis label")
    parser.add_argument("--ylabel", type=str, default="Number of entries", help="Y-axis label")
    parser.add_argument("--density", action="store_true", help="Plot density instead of counts")
    parser.add_argument("--range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                        help="Value range for histogram, e.g., --range -125 125")
    parser.add_argument("--logy", action="store_true", help="Use log scale on y-axis")

    args = parser.parse_args()

    npy_path = Path(args.npypath)
    if not npy_path.exists():
        raise FileNotFoundError(f"Not found: {npy_path}")

    arr = np.load(npy_path, allow_pickle=False)
    if arr.dtype == object:
        raise TypeError(f"dtype=object is not supported. path={npy_path}")

    flat = np.ravel(arr)

    out_path = Path(args.out) if args.out else (npy_path.with_suffix("").with_name(npy_path.stem + "_hist.png"))
    title = build_title(args, npy_path.name, arr)

    plt.figure()
    plt.hist(
        flat,
        bins=args.bins,
        density=args.density,
        range=tuple(args.range) if args.range else None,
    )
    plt.title(title)
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    if args.logy:
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()