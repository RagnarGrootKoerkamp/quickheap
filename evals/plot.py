#!/usr/bin/env python3
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import sys

benchname = sys.argv[1]


all = False
if len(sys.argv) > 2 and sys.argv[2] == "all":
    all = True

suff = "-all" if all else ""

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

df["type"] = df["name"].str.split("<").str[0]

# Sort by type
type_order = [
    # Dary
    "BinaryHeap",
    "DaryHeapOrx",
    "BoostDary4Heap",
    # "CustomBinaryHeap",
    # "CustomDaryHeap",
   
    # Amortized
    "WeakHeap",
    "BoostBinomialHeap",
    "FibonacciHeap",
    "BoostFibHeap",
    "PairingHeap",
    "BoostPairingHeap",
    "BoostSkewHeap",
    # Actual Competitors 
    "RadixHeapMap",
    "SequenceHeap",
    "S3qHeap",
    
    "OriginalQuickHeap",
    "ScalarQuickHeap",
    "SimdQuickHeap",
]

# Filtered out for the graph plot
graph_method_filter = ["FibonacciHeap<T>", "BoostBinomialHeap<T>", "BoostFibHeap<T>", "PairingHeap<T>", "BoostPairingHeap<T>", "BoostSkewHeap<T>"]
graph_instance_filter = ["NY"]

nanos_filter = ["BoostDary4Heap", "FibonacciHeap", "BoostPairingHeap", "BoostBinomialHeap", "BoostFibHeap", "PairingHeap", "BoostSkewHeap", "DaryHeapOrx<T, 4>"]

colours = (
    list(plt.get_cmap("tab20").colors)
    + list(plt.get_cmap("tab20b").colors)
    + list(plt.get_cmap("tab20c").colors)
)

# Assign colours from the fixed type_order so they are consistent across all plots
type_colour = {tp: colours[k % len(colours)] for k, tp in enumerate(type_order)}
type_colour["ScalarQuickHeap"] = "black"
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

    df["is_rhg"] = df["graph_name"].str.startswith("rhg").astype(int)
    graph_names = df.sort_values(["is_rhg", "vertices"])["graph_name"].unique()

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
    
    # Old: Graphs just sorted by name: graph_names = sorted(df["graph_name"].unique()
    def filter_graph(name):
        return not name in graph_method_filter
    
    methods = list(filter(filter_graph, df["name"].unique())) # already sorted by type/name above

    # hatches = ["", "//", "--", "xx", "++", "\\\\", "oo", ".."]
    hatches = [""]
    graph_hatch = {gn: hatches[k % len(hatches)] for k, gn in enumerate(graph_names)}
    graph_alpha = {
        gn: (1.0 if k == 1 or k == 5 else 0.5) for k, gn in enumerate(graph_names) # Change here if graphs are added or removed
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
            if (gn in graph_instance_filter):
                continue

            gdf = wdf[wdf["graph_name"] == gn].set_index("name")
            heights = [
                gdf.loc[m, "rel"] if m in gdf.index else float("nan") for m in methods # Relative
                # gdf.loc[m, "millis"] if m in gdf.index else float("nan") for m in methods # Nanos
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
        
        # Now using linear scale
        # Old (log scale): ax.set_yscale("log")

        ymax = df[df["workload"] == workload]["rel"].max()
        
        ax.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

        ax.set_ylim(0.8, 4)
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.3g}×"))
        ax.set_title(workload)
        ax.grid(axis="y", which="major", linestyle="-", alpha=0.4)
        if workload == workloads[0]:
            ax.legend(title="Graph", loc="upper right")

    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(methods, rotation=-45, ha="left", fontsize=8)

    # Color-coded type labels along x-axis
    for tick, method in zip(axs[-1].get_xticklabels(), methods):
        tick.set_color(type_colour.get(method_type[method], "black"))

    fig.supylabel("rel time on instance (compared to SIMD QuickHeap)")
    fig.tight_layout()
    fig.savefig(f"plots/{benchname}.svg", bbox_inches="tight")


elif "comparisons" in benchname:
    n_max = df["n"].max()
    df = df[df["n"] == n_max].copy()

    df["push_comparisons"] /= df["normalization"]
    df["pop_comparisons"] /= df["normalization"]
    df["comparisons"] = df["push_comparisons"] + df["pop_comparisons"]
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
    sqh_gap = bar_width * 0.5  # gap size
    offsets = []
    extra = 0.0
    for mi in range(n_methods):
        if mi in [3, 6, 8, 11]:
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
    ax.set_xticklabels(workloads, rotation=0, ha="center", fontsize=9)

    # ax.set_ylabel(r"$\# \mathsf{comparisons} / (\mathsf{pop} \circ \mathsf{push}) / \lg n$")
    ax.set_ylabel(r"$\frac{\# \mathsf{comparisons}}{I \cdot \lg n}$")
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
    fig.savefig(f"plots/{benchname}.pdf", bbox_inches="tight")
    fig.savefig(f"plots/{benchname}.png", bbox_inches="tight", dpi=300)

else:
    # Bench Plot

    # Take median over repeats, then normalize each metric by n*log2(n)
    metrics = [
        # ("nanos", r"$(\mathsf{ns} \:/\: \#(\mathsf{pop}\circ\mathsf{push})) \:/\: \lg n$"),
        ("nanos", r"Runtime / $ \frac{\mathsf{ns}}{I \cdot \lg n}$"),
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

    df["pop_comparisons"] /= df["normalization"]
    df["push_comparisons"] /= df["normalization"]
    metric_cols = [m for m, _ in metrics] + [
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
        if tp == "RadixHeapMap":
            return 1.0
        return 1.5

    def style_for_name(tp, name):
        if "Boost" in name:
            return "--"
        if tp == "ScalarQuickHeap":
            return ":"
        
        names = all_names_by_type[tp]
        if len(names) >= 2 and names.index(name) == 0:
            return "--"
        return "-"

    for metric, label in metrics:
        plt.close("all")
        p_width = 2
        fig, axs = plt.subplots(
            len(workloads),
            len(elems),
            figsize=(21/2.54, 29.7/2.54), # Size for A4 PDF
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        for j, elem in enumerate(elems):
            edf = df[df["elem"] == elem]
            for i, workload in enumerate(workloads):
                wdf = edf[edf["workload"] == workload]
                for tp, tgroup in wdf.groupby("type", sort=False):
                    c = type_colour[tp]
                    if not all and tp in nanos_filter:
                        continue

                    for name, ngroup in tgroup.groupby("name", sort=False):                        
                        if not all and name in nanos_filter:
                            continue

                        lw = width_for_type(tp)

                        ngroup.plot(
                            kind="line",
                            x="n",
                            y=metric,
                            ax=axs[i][j],
                            title=elem if i == 0 else None, # TODO Is this right?
                            label=name if j == 0 else None, # TODO Is this right?
                            color=c,
                            ls=style_for_name(tp, name),
                            lw=lw,
                        )

                axs[i][j].set_yscale("log", base=2)
                axs[i][j].set_xscale("log", base=2)
                axs[i][j].set_xticks([2**i for i in [10, 15, 20, 25]])
                axs[i][j].set_xlabel(None)
                axs[i][j].set_yticks([2**i for i in [0, 1, 2, 3, 4]])
                axs[i][j].set_ylabel(None)
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
            ncol=3,
            fontsize=8,
            bbox_to_anchor=(0.5, -0.020),
        )

        fig.supylabel(label)
        fig.supxlabel("$n$", y=0.02)
        fig.subplots_adjust(left=0.12, bottom=0.055, wspace=0.12, hspace=0.08)

        fig.savefig(f"plots/{benchname}-{metric}{suff}.svg", bbox_inches="tight")
        fig.savefig(f"plots/{benchname}-{metric}{suff}.pdf", bbox_inches="tight")
        fig.savefig(f"plots/{benchname}-{metric}{suff}.png", bbox_inches="tight", dpi=300)
    
    # Filter out the workload column, only keep constant size
    workload = "ConstantSize"
    df = df[df["workload"] == "ConstantSize"]
    df.drop('workload', axis='columns', inplace=True)
    
    for metric, label in metrics:
        plt.close("all")
        fig, axs = plt.subplots(
            1, # only one workload
            len(elems), # ['i32', 'i64']
            figsize=(10, 4),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        fig.suptitle(label)

        for j, elem in enumerate(elems):
            edf = df[df["elem"] == elem]
            wdf = edf
            for tp, tgroup in wdf.groupby("type", sort=False):
                c = type_colour[tp]
                if not all and tp in nanos_filter:
                    continue
                for name, ngroup in tgroup.groupby("name", sort=False):
                    lw = width_for_type(tp)
                    ngroup.plot(
                        kind="line",
                        x="n",
                        y=metric,
                        ax=axs[0, j],
                        title=elem,
                        # label=name if i == 0 and j == 0 else None,
                        label=name,
                        color=c,
                        ls=style_for_name(tp, name),
                        lw=lw,
                    )

            
            # cache_bytes_laptop = [32 * 1024, 256 * 1024, 12 * 1024 * 1024]
            # cache_bytes = [64 * 1024, 1024 * 1024, 96 * 1024 * 1024]
            # cache_n = [c // (4 if elem == "i32" else 8) for c in cache_bytes]
            # for cn in cache_n:
            #     axs[0, j].axvline(cn, color="blue", ls="-", lw=0.5, alpha=0.5)

            axs[0, j].set_yscale("log", base=2)
            axs[0, j].set_xscale("log", base=2)
            axs[0, j].set_xticks([2**i for i in [10, 15, 20, 25]])
            axs[0, j].legend().remove()
            axs[0, j].set_xlabel(None)
            axs[0, j].grid(axis="both", which="both", linestyle="-")
            axs[0, j].set_ylabel(None)
        
        # fig.supxlabel("$n$ = max # elements in heap", y=-0.01)
        fig.supxlabel("$n$", y=-0.2)
        handles, labels_leg = axs[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels_leg,
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, -0.27),
        )

        fig.savefig(f"plots/small-{benchname}-{metric}{suff}.svg", bbox_inches="tight")
        fig.savefig(f"plots/small-{benchname}-{metric}{suff}.pdf", bbox_inches="tight")
        fig.savefig(f"plots/small-{benchname}-{metric}{suff}.png", bbox_inches="tight", dpi=300)
