#!/usr/bin/env python3
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import to_hex
import re
import numpy as np
import pandas as pd
import sys

benchname = sys.argv[1]
use_hatching = "--no-hatch" not in sys.argv[2:]

df = pd.read_csv(f"data/{benchname}.csv")

# Shorten heap names
df["name"] = df["heap"]
df["name"] = df["name"].str.replace(
    "alloc::collections::binary_heap::", "", regex=False
)
df["name"] = df["name"].str.replace(
    "ConfigurableSimdQuickHeap", "SimdQuickHeap", regex=False
)
df["name"] = df["name"].str.replace(r"core::cmp::Reverse<[iu](32|64)>", "T", regex=True)
df["name"] = df["name"].str.replace(
    "orx_priority_queue::dary::daryheap::DaryHeap", "DaryHeapOrx", regex=False
)
df["name"] = df["name"].str.replace(r"(\w+::)+", "", regex=True)
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

df["workload"] = df["workload"].str.replace(r"Prim", "Jarnik-Prim", regex=False)

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


df["name"] = df["name"].str.replace(r"<true>", "", regex=True)

df["type"] = df["name"].str.split("<").str[0]

# Sort by type
type_order = [
    # Dary
    "BinaryHeap",
    "DaryHeapOrx",
    # Amortized
    "WeakHeap",
    # Actual Competitors
    "RadixHeap",
    "SequenceHeap",
    "S3qHeap",
    "OriginalQuickHeap",
    "ReimplementedQuickHeap",
    "ScalarQuickHeap",
    "SimdQuickHeap",
    # When plotting all:
    "RadixHeapMap",
    "BoostDary4Heap",
    "BoostBinomialHeap",
    "FibonacciHeap",
    "BoostFibHeap",
    "PairingHeap",
    "BoostPairingHeap",
    "BoostSkewHeap",
]

# Filtered out for the graph plot
graph_instance_filter = ["NY"]  # , "rhg_22"

nanos_filter = [
    "BoostDary4Heap",
    "FibonacciHeap",
    "BoostPairingHeap",
    "BoostBinomialHeap",
    "BoostFibHeap",
    "PairingHeap",
    "BoostSkewHeap",
    "DaryHeapOrx<T, 4>",
    "FibonacciHeap<T>",
    "BoostBinomialHeap<T>",
    "BoostFibHeap<T>",
    "PairingHeap<T>",
    "BoostPairingHeap<T>",
    "BoostSkewHeap<T>",
]

unnormalized_colours = [
    (152, 78, 163),
    (55, 126, 184),
    (77, 175, 74),
    (228, 26, 28),
    (255, 127, 0),
    (166, 86, 40),
    (247, 129, 191),
    (153, 153, 153),
]
# [(228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163), (255, 127, 0), (255, 255, 51), (166, 86, 40), (247, 129, 191), (153, 153, 153)]

colours = [(x / 255, y / 255, z / 255) for (x, y, z) in unnormalized_colours]

# Assign colours from the fixed type_order so they are consistent across all plots
type_colour = {tp: colours[k % len(colours)] for k, tp in enumerate(type_order)}
type_colour["ScalarQuickHeap"] = "black"
type_colour["SimdQuickHeap"] = "black"


def rewrite_legend(s):
    s = re.sub("<T>", "", s)
    s = re.sub("DaryHeapOrx<T, 8>", "8-aryHeap", s)
    s = re.sub("LinearScan", "L", s)
    s = re.sub("BinarySearch", "B", s)
    s = re.sub("MedianOfM<3>", "", s)
    s = re.sub("16, ", "", s)
    s = re.sub("true", "", s)
    s = re.sub("<T, ", "", s)
    s = re.sub(", ", "", s)
    s = re.sub(">", "", s)
    return s


if "graph" in benchname:
    # Shorten graph input names: "input/GER_graph.gr" -> "GER"
    df["graph_name"] = df["graph"].str.replace(
        r".*/(.*?)(?:_graph)?\.gr$", r"\1", regex=True
    )

    df = df[~df["graph"].str.contains("small")]
    df = df[~df["name"].str.contains("Custom")]

    df["is_rhg"] = df["graph_name"].str.startswith("rhg").astype(int)
    graph_names = df.sort_values(["is_rhg", "vertices"])["graph_name"].unique()

    # Take median over repeats
    df = (
        df.groupby(
            [
                "name",
                "type",
                "graph_name",
                "workload",
                "vertices",
                "edges",
                "push_count",
                "pop_count",
            ]
        )["nanos"]
        .median()
        .reset_index()
    )

    # Sort methods by type order then name
    df["order"] = pd.Categorical(df["type"], categories=type_order, ordered=True)
    df = df.sort_values(["order", "name"])

    # Normalize: for each (graph_name, workload), divide by the number of edges in the graph to get ns/edge.
    df["rel"] = df["nanos"] / df["push_count"]

    all_types = df["type"].unique()
    workloads = df["workload"].unique()

    def filter_graph(name):
        cleaned = re.sub(r"<[^>]*>", "", name)
        return not (cleaned in nanos_filter or name in nanos_filter)

    methods = list(
        filter(filter_graph, df["name"].unique())
    )  # already sorted by type/name above

    graph_names = [g for g in graph_names if g not in graph_instance_filter]

    n_methods = len(methods)
    bar_width = 0.8 / n_methods
    method_type = {m: df[df["name"] == m]["type"].iloc[0] for m in methods}
    bar_colors = [type_colour.get(method_type[m], "gray") for m in methods]

    # The QuickHeap variants share the black type colour; tell them apart by pattern
    def hatch_for_name(tp, name):
        if not use_hatching:
            return ""
        if tp == "ScalarQuickHeap":
            return "xx"
        if tp == "SimdQuickHeap" and "Avx512" not in name:
            return "//"
        return ""

    bar_hatches = [hatch_for_name(method_type[m], m) for m in methods]
    x = np.arange(len(graph_names))

    plt.close("all")
    fig, axs = plt.subplots(
        len(workloads),
        1,
        # figsize=(max(8, n_methods * 0.7), 4 * len(workloads)),
        # figsize=(21/2.54, 29.7/2.54), # Size for A4 PDF
        figsize=(21 / 2.54, 4 * len(workloads)),  # Width for A$ PDF
        sharex=True,
    )
    if len(workloads) == 1:
        axs = [axs]

    for ax, workload in zip(axs, workloads):
        wdf = df[df["workload"] == workload]

        for mi, method in enumerate(methods):
            mdf = wdf[wdf["name"] == method].set_index("graph_name")
            heights = [
                mdf.loc[g, "rel"] if g in mdf.index else float("nan")
                for g in graph_names
            ]
            offset = (mi - (n_methods - 1) / 2) * bar_width

            ax.bar(
                x + offset,
                heights,
                bar_width,
                color=bar_colors[mi],
                hatch=bar_hatches[mi],
                edgecolor="white",
            )

        ax.yaxis.set_major_locator(ticker.MultipleLocator(base=50))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.3g}"))
        ax.set_title(workload)
        ax.grid(axis="y", which="major", linestyle="-", alpha=0.4)

    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(graph_names, fontsize=8)

    method_handles = [
        mpatches.Patch(
            facecolor=bar_colors[mi],
            hatch=bar_hatches[mi],
            edgecolor="white",
            label=rewrite_legend(method),
        )
        for mi, method in enumerate(methods)
    ]
    fig.legend(
        handles=method_handles,
        loc="center",
        ncol=4,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.020),
    )

    fig.supylabel(r"Run time (ns / N)")
    fig.tight_layout()
    fig.savefig(f"plots/{benchname}.pdf", bbox_inches="tight")
    fig.savefig(f"plots/{benchname}.png", bbox_inches="tight")
    fig.savefig(f"plots/{benchname}.svg", bbox_inches="tight")

elif "comparisons" in benchname:
    n_max = df["n"].max()
    df = df[df["n"] == n_max].copy()

    def norm(x):
        if "HeapSort" in x:
            return 1
        if "Wiggle" in x:
            return 3
        if "ConstantSize" in x:
            return 10
        assert False

    df["ops"] = df.apply(lambda row: norm(row["workload"]) * row["n"], axis=1)

    df["push_comparisons"] /= df["ops"]
    df["pop_comparisons"] /= df["ops"]
    df["comparisons"] = df["push_comparisons"] + df["pop_comparisons"]

    # workloads = list(df["workload"].unique())
    # workloads = ['HeapSort', 'MonotoneConstantSize', 'MonotoneWiggle', 'ConstantSize', 'Wiggle']
    workloads = ["HeapSort", "MonotoneConstantSize", "MonotoneWiggle", "RandomWiggle"]

    df = (
        df.groupby(["elem", "name", "type", "workload"])[
            ["push_comparisons", "pop_comparisons"]
        ]
        .median()
        .reset_index()
    )
    lgn = np.log2(n_max)
    df["push_comparisons"] /= lgn
    df["pop_comparisons"] /= lgn

    df = df[df["elem"] == "i64"]

    df["order"] = pd.Categorical(df["type"], categories=type_order, ordered=True)
    df = df.sort_values(["order", "name"])
    methods = list(df["name"].unique())

    filtered_methods = []

    for method in methods:
        if method not in nanos_filter:
            filtered_methods.append(method)

    methods = filtered_methods

    n_methods = len(methods)
    n_workloads = len(workloads)
    bar_width = 0.7 / n_methods  # fill most of each group slot
    method_type = {m: df[df["name"] == m]["type"].iloc[0] for m in methods}
    bar_colors = [type_colour.get(method_type[m], "gray") for m in methods]

    # Build per-method offsets with extra gaps:
    #   - a small gap before the first ScalarQuickHeap bar
    #   - another small gap between the 3rd and 4th ScalarQuickHeap bar
    sqh_gap = bar_width * 0.5  # gap size
    # sqh_gap = 0  # gap size
    offsets = []
    extra = 0.0
    for mi in range(n_methods):
        if mi in [6, 9]:
            extra += sqh_gap
        offsets.append(mi * bar_width + extra)
    total = offsets[-1] + bar_width
    offsets = [o - total / 2 + bar_width / 2 for o in offsets]

    pop_alpha = 0.60

    plt.close("all")
    # fig, ax = plt.subplots(figsize=(max(8, n_workloads * (total + 0.3) * 1.5), 5))
    fig, ax = plt.subplots(figsize=(21 / 2.8, 4))
    x = np.arange(n_workloads)  # one tick per workload group

    legend_handles = [
        mpatches.Patch(facecolor="#333333FF", edgecolor="white", label="pop"),
        mpatches.Patch(facecolor="#33333399", edgecolor="white", label="push"),
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
    # ax.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    ax.set_ylim(0, 3.5)
    ax.set_xticklabels(workloads, rotation=0, ha="center", fontsize=8)

    # Label the two ScalarQuickHeap clusters (BinarySearch / LinearScan) under
    # each workload group; the workload names move down a line to make room.
    b_center = np.mean([offsets[mi] for mi, m in enumerate(methods) if "BinarySearch" in m])
    l_center = np.mean([offsets[mi] for mi, m in enumerate(methods) if "LinearScan" in m])
    minor_pos = [xi + c for xi in x for c in (b_center, l_center)]
    minor_labels = ["B", "L"] * n_workloads
    ax.set_xticks(minor_pos, minor=True)
    ax.set_xticklabels(minor_labels, minor=True, fontsize=8)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.tick_params(axis="x", which="major", pad=14)

    ax.set_ylabel(r"Comparisons ($1 \:/\: (N \cdot \lg n)$)")
    ax.grid(axis="y", linestyle="-", alpha=0.4)

    # Coloured method legend (top-left); the ScalarQuickHeap variants collapse
    # into a single entry, their B/L clusters are labelled on the axis
    method_handles = [
        mpatches.Patch(facecolor=bar_colors[mi], label=rewrite_legend(method))
        for mi, method in enumerate(methods)
        if method_type[method] != "ScalarQuickHeap"
    ]
    method_handles.append(
        mpatches.Patch(facecolor="black", edgecolor="white", label="ScalarQuickHeap")
    )
    legend_methods = ax.legend(
        handles=method_handles, loc="upper left", fontsize=8, ncol=4
    )
    ax.add_artist(legend_methods)

    # Push / pop legend (top-right)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, ncol=1)

    fig.tight_layout()
    fig.savefig(f"plots/{benchname}.svg", bbox_inches="tight")
    fig.savefig(f"plots/{benchname}.pdf", bbox_inches="tight")
    fig.savefig(f"plots/{benchname}.png", bbox_inches="tight", dpi=300)

elif "table" in benchname:
    # pivot table
    # - rows: "name"
    # - columns: 'n'
    # - values: ['nanos', 'l3_cache_misses'] / 'ops'
    # print(df.columns)
    # print(df)
    assert len(df.elem.unique()) == 1
    assert len(df.workload.unique()) == 1

    vals = [
        "nanos",
        "l1_cache_misses",
        "hw_cache_misses",
        "hw_cache_references",
        "l3_cache_misses",
    ]
    # NOTE: Only normalized per element,
    for val in vals:
        df[val] /= df["ops"]

    pivot = df.pivot_table(
        index=["name"],
        columns="n",
        values=vals,
        aggfunc="median",
        sort=False,
    )
    pivot.columns = [f"{val}_{n}" for val, n in pivot.columns]
    print(pivot)
    pivot.to_csv("table.dat")
else:
    # Bench Plot

    # Take median over repeats, then normalize each metric by n*log2(n)
    metrics = [
        # ("nanos", r"$(\mathsf{ns} \:/\: \#(\mathsf{pop}\circ\mathsf{push})) \:/\: \lg n$"),
        ("nanos", r"Run time ($\mathsf{ns} \:/\: (N \cdot \lg n)$)"),
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

    df["pop_comparisons"] /= df["ops"]
    df["push_comparisons"] /= df["ops"]
    metric_cols = [m for m, _ in metrics] + [
        "push_comparisons",
        "pop_comparisons",
    ]

    if df["nanos"].max() == 0:
        metrics = [
            (
                "comparisons",
                r"comparisons / ($\mathsf{push}\circ\mathsf{pop}) / \lg n$",
            )
        ]

    df = (
        df.groupby(["elem", "name", "type", "n", "workload"])[metric_cols + ["ops"]]
        .median()
        .reset_index()
    )

    df["order"] = pd.Categorical(df["type"], categories=type_order, ordered=True)
    df = df.sort_values(["order", "name"])
    ops = df["ops"] * np.log2(df["n"])
    for m in metric_cols:
        df[m] = df[m] / ops

    elems = df["elem"].unique()

    all_types = df["type"].unique()
    all_names_by_type = {
        tp: list(df[df["type"] == tp]["name"].unique()) for tp in all_types
    }

    def width_for_type(tp):
        # if tp == "RadixHeapMap":
        #     return 1.0
        return 1.5

    def style_for_name(tp, name):
        if "Boost" in name:
            return "--"
        if tp == "ScalarQuickHeap":
            return ":"
        # Match the graph-plot hatching: Avx512 solid, Avx2 patterned
        if tp == "SimdQuickHeap":
            return "-" if "Avx512" in name else "--"

        names = all_names_by_type[tp]
        if len(names) >= 2 and names.index(name) == 0:
            return "--"
        return "-"

    # workloads = df["workload"].unique()
    workloads = [
        "HeapSort",
        "MonotoneConstantSize",
        "MonotoneWiggle",
        "RandomWiggle",
    ]
    # workloads = ["MonotoneConstantSize"]

    plt.close("all")
    p_width = 2
    fig, axs = plt.subplots(
        len(workloads),
        len(elems),
        figsize=((21 / 2.54, 29.7 / 2.54)),  # Size for A4 PDF
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for metric, label in metrics:
        for j, elem in enumerate(elems):
            edf = df[df["elem"] == elem]
            for i, workload in enumerate(workloads):
                wdf = edf[edf["workload"] == workload]
                for tp, tgroup in wdf.groupby("type", sort=False):
                    for name, ngroup in tgroup.groupby("name", sort=False):
                        c = type_colour.get(tp, "gray")
                        if benchname == "ablation":
                            if "Avx2" in name:
                                c = "red"
                                if "RandomPivot" in name:
                                    c = "orange"
                                if "<5" in name:
                                    c = "brown"

                            if "Avx512" in name:
                                c = "blue"
                                if "RandomPivot" in name:
                                    c = "green"
                                if "<5" in name:
                                    c = "cyan"

                        lw = width_for_type(tp)
                        ngroup.plot(
                            kind="line",
                            x="n",
                            y=metric,
                            ax=axs[i][j],
                            title=elem if i == 0 else None,
                            label=rewrite_legend(name) if j == 0 else None,
                            color=c,
                            ls=style_for_name(tp, name),
                            lw=lw,
                        )

                axs[i][j].set_yscale("log", base=2)
                axs[i][j].set_xscale("log", base=2)
                axs[i][j].set_xticks([2**i for i in [10, 15, 20, 25]])
                axs[i][j].set_xlabel(None)
                if benchname != "ablation":
                    axs[i][j].set_yticks([2**i for i in [0, 1, 2, 3, 4]])
                axs[i][j].set_ylabel(None)
                # axs[i][j].set_ylim(0.1, 16)
                axs[i][j].grid(axis="both", which="both", linestyle="-")

                # Remove all intermediate legends
                axs[i][j].get_legend().remove()
                axs[i][j].set_ylabel(workload, fontsize=8)

    # handles, labels_leg = axs[len(workloads) - 1][0].get_legend_handles_labels()
    handles, labels_leg = axs[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_leg,
        loc="center",
        ncol=4,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.020),
    )

    fig.supylabel(label)
    fig.supxlabel("$n$", y=0.02)
    fig.subplots_adjust(left=0.12, bottom=0.055, wspace=0.12, hspace=0.08)

    fig.savefig(f"plots/{benchname}.svg", bbox_inches="tight")
    fig.savefig(f"plots/{benchname}.pdf", bbox_inches="tight")
    fig.savefig(f"plots/{benchname}.png", bbox_inches="tight", dpi=300)
