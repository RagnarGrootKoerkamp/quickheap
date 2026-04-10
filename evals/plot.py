#!/usr/bin/env python3
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import sys

benchname = sys.argv[1]

df = pd.read_csv(f"test-{benchname}.csv")

# Shorten heap names
df["name"] = df["heap"]
df["name"] = df["name"].str.replace(r"quickheap::\w+::", "", regex=True)
df["name"] = df["name"].str.replace(
    "alloc::collections::binary_heap::", "", regex=False
)
df["name"] = df["name"].str.replace(r"core::cmp::Reverse<[iu](32|64)>", "T", regex=True)
df["name"] = df["name"].str.replace(
    "orx_priority_queue::dary::daryheap::DaryHeap", "DaryHeapOrx", regex=False
)
df["name"] = df["name"].str.replace("dary_heap::", "", regex=False)
df["name"] = df["name"].str.replace("pheap::ph::", "", regex=False)
df["name"] = df["name"].str.replace("fibonacci_heap::", "", regex=False)
df["name"] = df["name"].str.replace("weakheap::", "", regex=False)
df["name"] = df["name"].str.replace("radix_heap::", "", regex=False)
df["name"] = df["name"].str.replace(r"\b[iu](32|64)\b", "T", regex=True)
df["name"] = df["name"].str.replace(r"[IU](32|64)", "<T>", regex=True)
df["name"] = df["name"].str.replace(r"<T><", "<T, ", regex=True)
df["name"] = df["name"].str.replace(r"<\(\), T", "<T", regex=True)
df["name"] = df["name"].str.replace(r", \(\)", "", regex=True)
df["name"] = df["name"].str.replace(r", 16, 1", "", regex=True)
df["name"] = df["name"].str.replace(r"Generic", "", regex=True)
df["type"] = df["name"].str.split("<").str[0]

# clean up ScalarQuickHeap for comparisons
df["name"] = df["name"].str.replace("Search::", "", regex=False)
df["name"] = df["name"].str.replace(", false", "", regex=False)
df["name"] = df["name"].str.replace("1, true", "∞", regex=False)
df["name"] = df["name"].str.replace(
    "(.), (BinarySearch|LinearScan)", "\\2, \\1", regex=True
)

# Shorten workload names
df["workload"] = (
    df["workload"].str.split("::").str[-1].str.replace("Workload", "", regex=False)
)

# Filter out some overly slow heaps
# df = df[
#     ~df["name"].str.contains(
#         "FibonacciHeap|PairingHeap|WeakHeap|DaryHeap<T, (2|4|8|16)>|DaryHeapOrx<T(|, 16)>"
#     )
# ]

df = df[~df["name"].str.contains("SimdQuickHeap<T, Avx512>")]
df["name"] = df["name"].str.replace(r"<true>", "", regex=True)

# Sort by type
type_order = [
    "BinaryHeap",
    "RadixHeapMap",
    "DaryHeapOrx",
    # "CustomBinaryHeap",
    # "CustomDaryHeap",
    "PairingHeap",
    "FibonacciHeap",
    "WeakHeap",
    "SequenceHeap",
    "S3qHeap",
    "ScalarQuickHeap",
    "SimdQuickHeap",
]

colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
styles = ["-", "--", "-.", ":"]
widths = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5]

# Assign colours from the fixed type_order so they are consistent across all plots
type_colour = {tp: colours[k % len(colours)] for k, tp in enumerate(type_order)}
type_colour["SimdQuickHeap"] = "black"

is_categorical = "graph" in df.columns

# TODO: Drop this
df["normalization"] = df["workload"].apply(lambda w: 3 if "Wiggle" in w else 1)
df["nanos"] /= df["normalization"]

if is_categorical:
    # Shorten graph input names: "input/GER_graph.gr" -> "GER"
    df["graph_name"] = df["graph"].str.replace(
        r".*/(.*?)(?:_graph)?\.gr$", r"\1", regex=True
    )

    df["millis"] = df["nanos"] / 1e6
    df = df[~df["graph"].str.contains("small")]
    df = df[~df["name"].str.contains("Custom")]

    # Take median over repeats
    df = (
        df.groupby(["name", "type", "graph_name", "workload"])["millis"]
        .median()
        .reset_index()
    )

    # Sort methods by type order then name
    df["order"] = pd.Categorical(df["type"], categories=type_order, ordered=True)
    df = df.sort_values(["order", "name"])

    # Normalize: for each (graph_name, workload), divide by the minimum nanos
    min_millis = df.groupby(["graph_name", "workload"])["millis"].transform("min")
    df["rel"] = df["millis"] / min_millis

    all_types = df["type"].unique()

    workloads = df["workload"].unique()
    graph_names = sorted(df["graph_name"].unique())
    methods = list(df["name"].unique())  # already sorted by type/name above

    # hatches = ["", "//", "--", "xx", "++", "\\\\", "oo", ".."]
    hatches = [""]
    graph_hatch = {gn: hatches[k % len(hatches)] for k, gn in enumerate(graph_names)}
    graph_alpha = {
        gn: (1.0 if k == 0 or k == 5 else 0.5) for k, gn in enumerate(graph_names)
    }

    n_methods = len(methods)
    bar_width = 0.8 / len(graph_names)
    method_type = {m: df[df["name"] == m]["type"].iloc[0] for m in methods}
    bar_colors = [type_colour.get(method_type[m], "gray") for m in methods]
    x = np.arange(n_methods)

    plt.close("all")
    fig, axs = plt.subplots(
        len(workloads),
        1,
        figsize=(max(8, n_methods * 0.7), 4 * len(workloads)),
        sharex=True,
    )
    if len(workloads) == 1:
        axs = [axs]

    for ax, workload in zip(axs, workloads):
        wdf = df[df["workload"] == workload]

        for gi, gn in enumerate(graph_names):
            gdf = wdf[wdf["graph_name"] == gn].set_index("name")
            heights = [
                gdf.loc[m, "rel"] if m in gdf.index else float("nan") for m in methods
            ]
            offset = (gi - (len(graph_names) - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                heights,
                bar_width,
                label=gn,
                color=bar_colors,
                hatch=graph_hatch[gn],
                alpha=graph_alpha[gn],
                edgecolor="white",
            )

        ax.axhline(1.0, color="gray", ls="--", lw=0.8)
        ax.set_yscale("log")
        ymax = df[df["workload"] == workload]["rel"].max()
        # tick_vals = [round(1.0 + i * 0.1, 1) for i in range(int((ymax - 1.0) / 0.1) + 2)]
        # ax.yaxis.set_major_locator(ticker.FixedLocator(tick_vals))
        ax.yaxis.set_major_locator(
            ticker.LogLocator(base=2, subs=(1.0, 1.2, 1.5), numticks=10)
        )
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.3g}×"))
        ax.set_ylabel("time (ms)")
        ax.set_title(workload)
        ax.grid(axis="y", which="major", linestyle="-", alpha=0.4)
        if workload == workloads[0]:
            ax.legend(title="Graph", loc="upper right")

    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(methods, rotation=35, ha="right", fontsize=8)

    # Color-coded type labels along x-axis
    for tick, method in zip(axs[-1].get_xticklabels(), methods):
        tick.set_color(type_colour.get(method_type[method], "black"))

    fig.tight_layout()
    fig.savefig(f"plot-{benchname}.svg", bbox_inches="tight")
    fig.savefig(f"plot-{benchname}.png", bbox_inches="tight", dpi=300)

elif benchname == "comparisons":
    n_max = df["n"].max()
    df = df[df["n"] == n_max].copy()

    df["push_comparisons"] /= df["normalization"]
    df["pop_comparisons"] /= df["normalization"]
    # df["pop_comparisons"] = (df["comparisons"] / df["normalization"]) - df[
    #     "push_comparisons"
    # ]
    ops = n_max * np.log2(n_max)

    workloads = list(df["workload"].unique())

    df = (
        df.groupby(["elem", "name", "type", "workload"])[
            ["push_comparisons", "pop_comparisons"]
        ]
        .median()
        .reset_index()
    )
    df["push_comparisons"] /= ops
    df["pop_comparisons"] /= ops

    df = df[df["elem"] == "i64"]

    df["order"] = pd.Categorical(df["type"], categories=type_order, ordered=True)
    df = df.sort_values(["order", "name"])
    methods = list(df["name"].unique())

    n_methods = len(methods)
    n_workloads = len(workloads)
    bar_width = 0.7 / n_methods  # fill most of each group slot
    method_type = {m: df[df["name"] == m]["type"].iloc[0] for m in methods}
    bar_colors = [type_colour.get(method_type[m], "gray") for m in methods]

    # Build per-method offsets with extra gaps:
    #   - a small gap before the first ScalarQuickHeap bar
    #   - another small gap between the 3rd and 4th ScalarQuickHeap bar
    sqh_gap = bar_width * 0.3  # gap size
    sqh_indices = [
        mi for mi, m in enumerate(methods) if method_type[m] == "ScalarQuickHeap"
    ]
    sqh_split = (
        sqh_indices[3] if len(sqh_indices) >= 4 else None
    )  # index of 4th SQH bar

    offsets = []
    extra = 0.0
    for mi in range(n_methods):
        if sqh_indices and mi == sqh_indices[0]:
            extra += sqh_gap
        if sqh_split is not None and mi == sqh_split:
            extra += sqh_gap
        offsets.append(mi * bar_width + extra)
    total = offsets[-1] + bar_width
    offsets = [o - total / 2 + bar_width / 2 for o in offsets]

    pop_alpha = 0.45

    plt.close("all")
    fig, ax = plt.subplots(figsize=(max(8, n_workloads * (total + 0.3) * 1.5), 5))
    x = np.arange(n_workloads)  # one tick per workload group

    legend_handles = [
        mpatches.Patch(
            facecolor="gray", alpha=pop_alpha, edgecolor="white", label="pop"
        ),
        mpatches.Patch(facecolor="gray", edgecolor="white", label="push"),
    ]

    for mi, method in enumerate(methods):
        mdf = df[df["name"] == method].set_index("workload")
        push = [
            mdf.loc[w, "push_comparisons"] if w in mdf.index else 0.0 for w in workloads
        ]
        pop = [
            mdf.loc[w, "pop_comparisons"] if w in mdf.index else 0.0 for w in workloads
        ]
        offset = offsets[mi]
        color = bar_colors[mi]
        ax.bar(x + offset, push, bar_width, color=color, edgecolor="white")
        ax.bar(
            x + offset,
            pop,
            bar_width,
            bottom=push,
            color=color,
            edgecolor="white",
            alpha=pop_alpha,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=15, ha="right", fontsize=9)

    ax.set_ylabel(r"comparisons / (push∘pop) / lg n")
    ax.grid(axis="y", linestyle="-", alpha=0.4)

    # Coloured method legend (top-left)
    method_handles = [
        mpatches.Patch(facecolor=bar_colors[mi], label=method)
        for mi, method in enumerate(methods)
    ]
    legend_methods = ax.legend(
        handles=method_handles, loc="upper left", fontsize=8, ncol=2
    )
    ax.add_artist(legend_methods)

    # Push / pop legend (top-right)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, ncol=1)

    fig.tight_layout()
    fig.savefig(f"plots/{benchname}.svg", bbox_inches="tight")
    fig.savefig(f"plots/{benchname}.png", bbox_inches="tight", dpi=300)

else:
    # Take median over repeats, then normalize each metric by n*log2(n)
    metrics = [
        ("nanos", r"ns / ($\mathsf{push}\circ\mathsf{pop}) / \lg n$"),
        # ("branch_misses", r"branch misses / ($\mathsf{push}\circ\mathsf{pop}) / \lg n$"),
        # (
        #     "l1_cache_misses",
        #     r"L1 cache misses / ($\mathsf{push}\circ\mathsf{pop}) / \lg n$",
        # ),
        # (
        #     "l3_cache_misses",
        #     r"L3 cache misses / ($\mathsf{push}\circ\mathsf{pop}) / \lg n$",
        # ),
    ]

    if "push_comparisons" not in df.columns:
        df["push_comparisons"] = 0
    df["comparisons"] /= df["normalization"]
    df["push_comparisons"] /= df["normalization"]
    df["pop_comparisons"] = df["comparisons"] - df["push_comparisons"]
    metric_cols = [m for m, _ in metrics] + [
        "comparisons",
        "push_comparisons",
        "pop_comparisons",
    ]

    workloads = df["workload"].unique()

    if df["nanos"].max() == 0:
        metrics = [
            ("comparisons", r"comparisons / ($\mathsf{push}\circ\mathsf{pop}) / \lg n$")
        ]

    df = (
        df.groupby(["elem", "name", "type", "n", "workload"])[metric_cols]
        .median()
        .reset_index()
    )

    df["order"] = pd.Categorical(df["type"], categories=type_order, ordered=True)
    df = df.sort_values(["order", "name"])
    ops = df["n"] * np.log2(df["n"])
    for m in metric_cols:
        df[m] = df[m] / ops

    elems = df["elem"].unique()

    all_types = df["type"].unique()
    all_names_by_type = {
        tp: list(df[df["type"] == tp]["name"].unique()) for tp in all_types
    }

    def width_for_type(tp):
        if tp == "SimdQuickHeap":
            return 2.0
        if tp == "SimdQuickHeap512":
            return 2.0
        if tp == "RadixHeapMap":
            return 1.4
        return 2.0

    def style_for_type(tp):
        if tp == "RadixHeapMap":
            return "--"
        return "-"

    for metric, label in metrics:
        plt.close("all")
        fig, axs = plt.subplots(
            len(elems),
            len(workloads),
            figsize=(4 * len(workloads), 6),
            sharex=True,
            sharey=True,
        )
        fig.suptitle(label)

        for j, elem in enumerate(elems):
            edf = df[df["elem"] == elem]
            for i, workload in enumerate(workloads):
                wdf = edf[edf["workload"] == workload]
                for tp, tgroup in wdf.groupby("type", sort=False):
                    c = type_colour[tp]
                    for name, ngroup in tgroup.groupby("name", sort=False):
                        lw = widths[all_names_by_type[tp].index(name)]
                        ngroup.plot(
                            kind="line",
                            x="n",
                            y=metric,
                            logx=True,
                            ax=axs[j][i],
                            title=workload if j == 0 else None,
                            label=name if i == 0 and j == 0 else None,
                            # color=c,
                            ls=style_for_type(tp),
                            lw=lw,
                        )
                        c = axs[j][i].get_lines()[-1].get_color()
                        if metric == "comparisons":
                            ngroup.plot(
                                kind="line",
                                x="n",
                                y="push_comparisons",
                                logx=True,
                                ax=axs[j][i],
                                label="_nolegend_",
                                color=c,
                                ls=":",
                                lw=lw,
                            )
                            ngroup.plot(
                                kind="line",
                                x="n",
                                y="pop_comparisons",
                                logx=True,
                                label="_nolegend_",
                                ax=axs[j][i],
                                color=c,
                                ls="--",
                                lw=lw,
                            )

                # cache_bytes_laptop = [32 * 1024, 256 * 1024, 12 * 1024 * 1024]
                cache_bytes = [64 * 1024, 1024 * 1024, 96 * 1024 * 1024]
                cache_n = [c // (4 if elem == "i32" else 8) for c in cache_bytes]
                for cn in cache_n:
                    axs[j][i].axvline(cn, color="gray", ls="-", lw=0.5, alpha=0.5)

                if metric != "comparisons":
                    axs[j][i].set_yscale("log", base=2)
                axs[j][i].set_xscale("log", base=2)
                axs[j][i].set_xticks([2**i for i in [10, 15, 20, 25]])
                axs[j][i].legend().remove()
                axs[j][i].set_xlabel(None)
                axs[j][i].grid(axis="y", which="both", linestyle="-")
                if j > 0:
                    axs[j][i].set_title(None)

            axs[j][0].set_ylabel(elem)

        handles, labels_leg = axs[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels_leg,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, -0.10),
        )
        fig.supxlabel("n = max #elements in heap", y=0.02)

        fig.savefig(f"plots/{benchname}-{metric}.svg", bbox_inches="tight")
        fig.savefig(f"plots/{benchname}-{metric}.png", bbox_inches="tight", dpi=300)
