#!/usr/bin/env python3
"""Usage: python gen.py experiment.json   (a single object or a list of objects)

Mode is selected per experiment:
  - "mode": "scaling" | "instance"  (explicit), otherwise
  - "baseline" in config -> instance mode (per-instance speedup, grouped bars)
  - else -> scaling mode (n on x-axis, line plots)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data" / "experiments"
OUT_DIR = DATA_DIR / "plots"
KEY_COLS = ["elem_type", "simd_backend", "n", "workload", "heap"]
INST_KEY_COLS = ["simd_backend", "instance", "workload", "heap"]
LINESTYLES = {"Avx512": "-", "Avx2": "--", "Neon": ":"}

TEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}"}


def tex(s):
    return "".join(TEX_ESC.get(c, c) for c in str(s))


def get_mode(cfg):
    if "mode" in cfg:
        return cfg["mode"]
    return "instance" if "baseline" in cfg else "scaling"


def load(cfg):
    name = cfg["name"]
    mode = get_mode(cfg)
    frames = []
    for suffix in ("neon", "avx"):
        path = DATA_DIR / f"{name}_{suffix}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, quotechar='"'))
    if not frames:
        raise FileNotFoundError(
            f"No data found: {DATA_DIR / (name + '_neon.csv')} or {DATA_DIR / (name + '_avx.csv')}")
    df = pd.concat(frames, ignore_index=True)
    if mode == "instance":
        if "workload" not in df.columns:  # e.g. decrease_key
            df["workload"] = cfg.get("default_workload", name)
        keys = INST_KEY_COLS
    else:
        keys = KEY_COLS
    dup = df.duplicated(subset=keys)
    if dup.any():
        raise ValueError(f"Duplicate rows:\n{df[dup][keys]}")
    return df


# ------------------------------------------------------------------ scaling mode
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
    speedup = cfg.get("show_speedup", False)
    sp_metric = cfg.get("speedup_metric", "nano_seconds")
    sp_j = cols.index(sp_metric) if sp_metric in cols else None

    sub = df[(df["n"] == cfg["table_n"]) & (df["elem_type"] == cfg["table_type"]) &
             (df["workload"] == cfg["table_workload"]) &
             (df["simd_backend"] == cfg["table_backend"])]
    heaps = list(dict.fromkeys(df["heap"]))

    values = {}
    for h in heaps:
        row = sub[sub["heap"] == h]
        values[h] = [float(row.iloc[0][c]) if len(row) else None for c in cols]

    def fmt(v, bold=False):
        if v is None:
            return "--"
        s = f"{v:.2f}"
        return rf"\textbf{{{s}}}" if bold else s

    best = []
    for j in range(len(cols)):
        present = [values[h][j] for h in heaps if values[h][j] is not None]
        best.append(min(present) if present else None)

    ref = None
    if sp_j is not None:
        ref = next((values[h][sp_j] for h in heaps if values[h][sp_j] is not None), None)

    header = [tex(mm.get(c, c)) for c in cols]
    if speedup and sp_j is not None:
        header.append("Speedup")

    caption = (f"type: {cfg['table_type']}, $n = {cfg['table_n']}$, "
               f"backend: {tex(cfg['table_backend'])}, "
               f"workload: {tex(cfg['table_workload'])}")
    lines = [r"\begin{table}[ht]", r"\centering",
             r"\begin{tabular}{l" + "r" * len(header) + "}", r"\hline",
             "Heap & " + " & ".join(header) + r" \\", r"\hline"]
    for h in heaps:
        cells = [fmt(v, j == sp_j and v is not None and v == best[j])
                 for j, v in enumerate(values[h])]
        if speedup and sp_j is not None:
            v = values[h][sp_j]
            cells.append("--" if v is None or ref is None
                         else rf"{ref / v:.2f}$\times$")
        lines.append(tex(hm.get(h, h)) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}",
              rf"\caption{{{caption}}}", rf"\label{{tab:{cfg['name']}}}",
              r"\end{table}"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{cfg['name']}_table.tex").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------- instance mode
def _geomean(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v) and v > 0])
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


def _pivot(df, workload, backend, metric, instances):
    sub = df[(df["workload"] == workload) & (df["simd_backend"] == backend)]
    return sub.pivot(index="instance", columns="heap", values=metric).reindex(instances)


def _inst_setup(df, cfg):
    metric = cfg.get("metric", "nano_seconds")
    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not in data columns {list(df.columns)}")
    base = cfg["baseline"]
    instances = cfg.get("instance_order") or list(dict.fromkeys(df["instance"]))
    heaps = cfg.get("heap_order") or list(dict.fromkeys(df["heap"]))
    backends = cfg.get("backends") or [b for b in LINESTYLES if b in set(df["simd_backend"])] + \
        sorted(set(df["simd_backend"]) - set(LINESTYLES))
    workloads = cfg.get("workloads") or sorted(df["workload"].unique())
    return metric, base, instances, heaps, backends, workloads


def make_instance_plots(df, cfg):
    hm = cfg.get("heap_mappings", {})
    hc = cfg.get("heap_colors", {})
    im = cfg.get("instance_mappings", {})
    metric, base, instances, heaps, backends, workloads = _inst_setup(df, cfg)
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {h: hc.get(h, default_colors[i % len(default_colors)]) for i, h in enumerate(heaps)}
    bars = [h for h in heaps if h != base]
    log_y = cfg.get("log_y", True)
    with_gm = cfg.get("geomean", True)
    ylabel = cfg.get("ylabel", f"Speedup vs. {hm.get(base, base)} (higher = better)")

    for workload in workloads:
        fig, axes = plt.subplots(1, len(backends),
                                 figsize=(max(5, 0.9 * (len(instances) + 1)) * len(backends), 4.2),
                                 sharey=True, squeeze=False)
        for ax, be in zip(axes[0], backends):
            piv = _pivot(df, workload, be, metric, instances)
            if base not in piv.columns:
                print(f"[warn] baseline '{base}' missing for {workload}/{be}", file=sys.stderr)
                continue
            sp = piv[base].values[:, None] / piv.reindex(columns=bars).values  # inst x heaps
            labels = [im.get(i, i) for i in instances]
            if with_gm:
                gm = np.array([_geomean(sp[:, j]) for j in range(len(bars))])
                sp = np.vstack([sp, gm])
                labels.append("geomean")
            x = np.arange(len(labels))
            w = 0.8 / max(len(bars), 1)
            for j, h in enumerate(bars):
                off = (j - (len(bars) - 1) / 2) * w
                v = sp[:, j]
                ok = np.isfinite(v)
                ax.bar(x[ok] + off, v[ok] - 1, width=w, bottom=1, color=colors[h])
            ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
            if with_gm:
                ax.axvline(len(labels) - 1.5, color="gray", linewidth=0.6)
            if log_y:
                ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_title(be)
            ax.grid(axis="y", alpha=0.3)
        axes[0][0].set_ylabel(ylabel)
        handles = [Line2D([], [], color=colors[h], linewidth=6, label=hm.get(h, h)) for h in bars]
        handles.append(Line2D([], [], color="k", linestyle="--",
                              label=f"{hm.get(base, base)} (baseline)"))
        fig.suptitle(workload)
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
                   fontsize=7, frameon=False)
        fig.tight_layout(rect=(0, 0.10 + 0.04 * (len(handles) // 4), 1, 0.95))
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = f"{cfg['name']}_{metric}_{workload}"
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=200)
        plt.close(fig)


def _fmt_val(v):
    """Print the value as it is in the CSV (no scaling/rounding)."""
    v = float(v)
    return str(int(v)) if v.is_integer() else repr(v)


def make_instance_tables(df, cfg):
    hm = cfg.get("heap_mappings", {})
    mm = cfg.get("metric_mappings", {})
    im = cfg.get("instance_mappings", {})
    metric, base, instances, heaps, backends, workloads = _inst_setup(df, cfg)
    be = cfg.get("table_backend", backends[0])
    unit = tex("ns" if metric == "nano_seconds" else mm.get(metric, metric))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for workload in workloads:
        piv = _pivot(df, workload, be, metric, instances)
        cols = [h for h in heaps if h in piv.columns]
        if base not in cols:
            print(f"[warn] baseline '{base}' missing for {workload}/{be}", file=sys.stderr)
            continue
        sp = {h: piv[base] / piv[h] for h in cols}
        lines = [r"\begin{table}[ht]", r"\centering",
                 r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\hline",
                 "Instance & " + " & ".join(tex(hm.get(h, h)) for h in cols) + r" \\", r"\hline"]
        for inst in instances:
            row = piv.loc[inst, cols]
            present = row.dropna()
            best = present.min() if len(present) else None
            cells = []
            for h in cols:
                v = row[h]
                if pd.isna(v):
                    cells.append("--")
                    continue
                s = _fmt_val(v)
                cells.append(rf"\textbf{{{s}}}" if v == best else s)
            lines.append(tex(im.get(inst, inst)) + " & " + " & ".join(cells) + r" \\")
        gm = {h: _geomean(sp[h].values) for h in cols}
        gm_best = max((v for v in gm.values() if np.isfinite(v)), default=None)
        cells = []
        for h in cols:
            v = gm[h]
            if not np.isfinite(v):
                cells.append("--")
                continue
            s = f"{v:.2f}$\\times$"
            cells.append(rf"\textbf{{{s}}}" if v == gm_best else s)
        lines += [r"\hline", "Geomean speedup & " + " & ".join(cells) + r" \\", r"\hline",
                  r"\end{tabular}",
                  rf"\caption{{workload: {tex(workload)}, backend: {tex(be)}, "
                  rf"runtime in {unit}; speedup relative to {tex(hm.get(base, base))}}}",
                  rf"\label{{tab:{cfg['name']}_{workload}}}",
                  r"\end{table}"]
        (OUT_DIR / f"{cfg['name']}_{workload}_table.tex").write_text("\n".join(lines) + "\n")


def main():
    cfgs = json.loads(Path(sys.argv[1]).read_text())
    if isinstance(cfgs, dict):
        cfgs = [cfgs]
    for cfg in cfgs:
        df = load(cfg)
        if get_mode(cfg) == "instance":
            make_instance_plots(df, cfg)
            make_instance_tables(df, cfg)
        else:
            make_plots(df, cfg)
            make_table(df, cfg)


if __name__ == "__main__":
    main()
