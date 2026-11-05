#!/usr/bin/env python3
"""Usage: python gen.py experiment.json   (a single object or a list of objects)"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data" / "experiments"
OUT_DIR = DATA_DIR / "plots"
KEY_COLS = ["elem_type", "simd_backend", "n", "workload", "heap"]
LINESTYLES = {"Avx512": "-", "Avx2": "--", "Neon": ":"}

TEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}"}


def tex(s):
    return "".join(TEX_ESC.get(c, c) for c in str(s))


def load(name):
    df = pd.read_csv(DATA_DIR / f"{name}.csv", quotechar='"')
    dup = df.duplicated(subset=KEY_COLS)
    if dup.any():
        raise ValueError(f"Duplicate rows:\n{df[dup][KEY_COLS]}")
    return df


def make_plots(df, cfg):
    hm = cfg.get("heap_mappings", {})
    hc = cfg.get("heap_colors", {})
    mm = cfg.get("metric_mappings", {})
    types = cfg["plot_types"]
    heaps = list(dict.fromkeys(df["heap"]))
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {h: hc.get(h, default_colors[i % len(default_colors)]) for i, h in enumerate(heaps)}
    backends = [b for b in LINESTYLES if b in set(df["simd_backend"])] + \
               sorted(set(df["simd_backend"]) - set(LINESTYLES))

    for metric in cfg["plot_by"]:
        for workload in sorted(df["workload"].unique()):
            fig, axes = plt.subplots(1, len(types), figsize=(5.5 * len(types), 4),
                                     sharey=True, squeeze=False)
            for ax, t in zip(axes[0], types):
                sub = df[(df["workload"] == workload) & (df["elem_type"] == t)]
                for (heap, be), g in sub.groupby(["heap", "simd_backend"]):
                    g = g.sort_values("n")
                    ax.plot(g["n"], g[metric], color=colors[heap],
                            linestyle=LINESTYLES.get(be, "-"))
                ax.set_xscale("log", base=2)
                ax.set_title(t)
                ax.set_xlabel("n")
                ax.grid(alpha=0.3)
            axes[0][0].set_ylabel(mm.get(metric, metric))
            handles = [Line2D([], [], color=colors[h], label=hm.get(h, h)) for h in heaps]
            handles += [Line2D([], [], color="k", linestyle=LINESTYLES.get(b, "-"), label=b)
                        for b in backends]
            fig.suptitle(workload)
            fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 3),
                       fontsize=7, frameon=False)
            fig.tight_layout(rect=(0, 0.12 + 0.04 * (len(handles) // 3), 1, 0.95))
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            stem = f"{cfg['name']}_{metric}_{workload}"
            for ext in ("pdf", "png"):
                fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=200)
            plt.close(fig)


def make_table(df, cfg):
    hm = cfg.get("heap_mappings", {})
    mm = cfg.get("metric_mappings", {})
    cols = cfg["table_cols"]
    sub = df[(df["n"] == cfg["table_n"]) & (df["elem_type"] == cfg["table_type"]) &
             (df["workload"] == cfg["table_workload"]) &
             (df["simd_backend"] == cfg["table_backend"])]
    caption = (f"$n = {cfg['table_n']}$, backend: {tex(cfg['table_backend'])}, "
               f"workload: {tex(cfg['table_workload'])}")
    lines = [r"\begin{table}[ht]", r"\centering",
             r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\hline",
             "Heap & " + " & ".join(tex(mm.get(c, c)) for c in cols) + r" \\", r"\hline"]
    for heap in dict.fromkeys(df["heap"]):
        row = sub[sub["heap"] == heap]
        vals = [f"{row.iloc[0][c]:.2f}" if len(row) else "--" for c in cols]
        lines.append(tex(hm.get(heap, heap)) + " & " + " & ".join(vals) + r" \\")
    lines += [r"\hline", r"\end{tabular}",
              rf"\caption{{{caption}}}",
              rf"\label{{tab:{cfg['name']}}}",
              r"\end{table}"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{cfg['name']}_table.tex").write_text("\n".join(lines) + "\n")


def main():
    cfgs = json.loads(Path(sys.argv[1]).read_text())
    if isinstance(cfgs, dict):
        cfgs = [cfgs]
    for cfg in cfgs:
        df = load(cfg["name"])
        make_plots(df, cfg)
        make_table(df, cfg)


if __name__ == "__main__":
    main()
