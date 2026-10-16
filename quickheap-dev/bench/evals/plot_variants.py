#!/usr/bin/env python3
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

DATA_DIR = "./data/variants/"
OUT_DIR = "./plots"

os.makedirs(OUT_DIR, exist_ok=True)

# Load all variants; variant name = file name without extension
frames = []
for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
    df = pd.read_csv(path)
    df["variant"] = os.path.splitext(os.path.basename(path))[0]
    frames.append(df)

data = pd.concat(frames, ignore_index=True)
variants = sorted(data["variant"].unique())
cmap = plt.get_cmap("Set2")
colors = {v: cmap(i % cmap.N) for i, v in enumerate(variants)}

for workload, wdf in data.groupby("workload"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in variants:
        vdf = wdf[wdf["variant"] == variant].sort_values("n")
        if vdf.empty:
            continue
        y = vdf["nanoseconds"] / np.log2(vdf["n"])
        ax.plot(vdf["n"], y, linewidth=2, color=colors[variant], label=variant)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("n")
    ax.set_ylabel("Time per element / log₂(n) [ns]")
    ax.set_title(workload)
    ax.grid(True, alpha=0.3)

    # 2^i ticks covering the data range
    lo, hi = ax.get_ylim()
    exps = np.arange(np.floor(np.log2(lo)), np.ceil(np.log2(hi)) + 1)
    ax.yaxis.set_major_locator(FixedLocator(2.0 ** exps))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"$2^{{{int(round(np.log2(v)))}}}$"))

    ax.legend(title="Heap variant", loc="upper center",
              bbox_to_anchor=(0.5, -0.15), ncol=min(len(variants), 4),
              frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{workload}.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, f"{workload}.pdf"), bbox_inches="tight")
    plt.close(fig)
