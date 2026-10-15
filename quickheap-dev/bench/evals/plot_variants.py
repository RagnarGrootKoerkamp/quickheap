#!/usr/bin/env python3
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

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
colors = dict(zip(variants, plt.get_cmap("tab10").colors))

for workload, wdf in data.groupby("workload"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in variants:
        vdf = wdf[wdf["variant"] == variant].sort_values("n")
        if vdf.empty:
            continue
        ax.plot(vdf["n"], vdf["nanoseconds"], marker="o", linewidth=2,
                color=colors[variant], label=variant)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("n")
    ax.set_ylabel("Time per element [ns]")
    ax.set_title(workload)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Heap variant", loc="upper center",
                  bbox_to_anchor=(0.5, -0.15), ncol=min(len(variants), 4),
                  frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{workload}.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, f"{workload}.pdf"), bbox_inches="tight")
    plt.close(fig)
