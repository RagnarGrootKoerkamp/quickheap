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

df["workload"] = df["workload"].str.replace(r"RandomConstantSize", "rcs", regex=True)
df["workload"] = df["workload"].str.replace(r"ConstantSize", "MonotoneConstantSize", regex=True)
df["workload"] = df["workload"].str.replace(r"rcs", "ConstantSize", regex=True)


df = df[~df["name"].str.contains("SimdQuickHeap<T, Avx512>")]
df["name"] = df["name"].str.replace(r"<true>", "", regex=True)

df["type"] = df["name"].str.split("<").str[0]

# Sort by type
type_order = [
    # Dary
    "BinaryHeap",
    "DaryHeapOrx",
    # "BoostDary4Heap",
    "WeakHeap",

    # Amortized
    
    # "BoostBinomialHeap",
    # "FibonacciHeap",
    # "BoostFibHeap",
    # "PairingHeap",
    # "BoostPairingHeap",
    # "BoostSkewHeap",

    # Actual Competitors 
    "RadixHeapMap",
    "SequenceHeap",
    "S3qHeap",
    
    "OriginalQuickHeap",
    "ScalarQuickHeap",
    "SimdQuickHeap",
]

# Filtered out for the graph plot
graph_instance_filter = ["NY"] # , "rhg_22"

nanos_filter = ["BoostDary4Heap", "FibonacciHeap", "BoostPairingHeap", "BoostBinomialHeap", "BoostFibHeap", "PairingHeap", "BoostSkewHeap", "DaryHeapOrx<T, 4>", "FibonacciHeap<T>", "BoostBinomialHeap<T>", "BoostFibHeap<T>", "PairingHeap<T>", "BoostPairingHeap<T>", "BoostSkewHeap<T>"]

unnormalized_colours = [(152, 78, 163), (55, 126, 184), (77, 175, 74), (228, 26, 28), (255, 127, 0), (166, 86, 40), (247, 129, 191), (153, 153, 153)]
# [(228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163), (255, 127, 0), (255, 255, 51), (166, 86, 40), (247, 129, 191), (153, 153, 153)]

colours = [(x / 255, y / 255, z / 255) for (x, y, z) in unnormalized_colours]

# Assign colours from the fixed type_order so they are consistent across all plots
type_colour = {tp: colours[k % len(colours)] for k, tp in enumerate(type_order)}
type_colour["ScalarQuickHeap"] = "black"
type_colour["SimdQuickHeap"] = "black"

is_categorical = "graph" in df.columns


# TODO: Mixed normalization factors because the bench data is mixed between different runs
df["normalization"] = df["workload"].apply(lambda w: 3 if "Wiggle" in w else (10 if "ConstantSize" in w  else 1)) # TODO!!!

# df["normalization"] = df["workload"].apply(lambda w: 3 if "Wiggle" in w else 1)

# mask = ((df["name"] == "SimdQuickHeap<T, Avx2, MedianOfM<3>, 16, true>") | (df["name"] == "SimdQuickHeap<T, Avx512, MedianOfM<3>, 16, true>")) & (df["workload"] == "MonotoneConstantSize")
# df.loc[mask, "normalization"] = df["normalization"].apply(lambda x: 10)


df["nanos"] /= df["normalization"]


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

    def filter_graph(name):
        cleaned = re.sub(r'<[^>]*>', '', name)
        return all or not (cleaned in nanos_filter or name in nanos_filter)
    
    methods = list(filter(filter_graph, df["name"].unique())) # already sorted by type/name above

    # hatches = ["", "//", "--", "xx", "++", "\\\\", "oo", ".."]
    hatches = [""]
    graph_hatch = {gn: hatches[k % len(hatches)] for k, gn in enumerate(graph_names)}
    graph_alpha = {
        gn: (1.0 if k < 5 else 0.5) for k, gn in enumerate(graph_names) # Change here if graphs are added or removed
    }

    filtered_graph_names = []
    for g in graph_names:
        if not g in graph_instance_filter:
            filtered_graph_names.append(g)

    graph_names = filtered_graph_names

    n_methods = len(methods)
    bar_width = 0.8 / len(graph_names)
    method_type = {m: df[df["name"] == m]["type"].iloc[0] for m in methods}
    bar_colors = [type_colour.get(method_type[m], "gray") for m in methods]
    x = np.arange(n_methods)

    plt.close("all")
    fig, axs = plt.subplots(
        len(workloads),
        1,
        # figsize=(max(8, n_methods * 0.7), 4 * len(workloads)),
        # figsize=(21/2.54, 29.7/2.54), # Size for A4 PDF
        figsize=(21/2.54, 4 * len(workloads)), # Width for A$ PDF
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
        ymax = df[df["workload"] == workload]["rel"].max()
        ax.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
        ax.set_ylim(0.8, 5)
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.3g}×"))
        ax.set_title(workload)
        ax.grid(axis="y", which="major", linestyle="-", alpha=0.4)

    short_methods = [rewrite_legend(method) for method in methods]
    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(short_methods, rotation=-45, ha="left", fontsize=8)

    # Color-coded type labels along x-axis
    for tick, method in zip(axs[-1].get_xticklabels(), methods):
        tick.set_color(type_colour.get(method_type[method], "black"))

    # Get legend handles and labels from the last subplot
    handles, labels = axs[-1].get_legend_handles_labels()
    
    # Filter to keep only the desired graph types
    filtered_handles = []
    filtered_labels = []
    for handle, label in zip(handles, labels):
        if label.startswith("rhg"):
            filtered_labels.append("Random Hyperbolic Graphs: 20, 22, 24")
            filtered_handles.append(handle)
        else:
            filtered_labels.append(f"Road Networks: CAL, CTR, GER, USA")
            filtered_handles.append(handle)
    
    # Deduplicate labels (keep first occurrence of each unique label)
    seen = set()
    final_handles = []
    final_labels = []
    for handle, label in zip(filtered_handles, filtered_labels):
        if label not in seen:
            final_handles.append(handle)
            final_labels.append(label)
            seen.add(label)



    # Add a single grey proxy for the legend (if you want one grey legend entry)
    # For example, if you want "Random Hyperbolic Graphs" to be grey in legend only:
    grey_patch_rn = mpatches.Patch(linewidth=0, color='#333333FF', label="Road Networks: CAL, CTR, GER, USA")
    grey_patch_rhg = mpatches.Patch(linewidth=0, color="#33333399", label="Random Hyperbolic Graphs: 20, 22, 24")

    # Also, keep the existing road‑networks label
    road_label = "Road Networks: CAL, CTR, GER, USA"
    road_handle = [h for h, l in zip(final_handles, final_labels) if l == road_label][0]

    # Replace or adjust final_handles/labels as needed
    # Example: keep existing road handle, add a grey proxy for RHG
    new_final_handles = [grey_patch_rn, grey_patch_rhg]
    new_final_labels = [road_label, "Random Hyperbolic Graphs: 20, 22, 24"]

    # fig.legend(handles=new_final_handles, labels=new_final_labels)
    
    fig.legend(new_final_handles, new_final_labels, loc='upper right', bbox_to_anchor=(0.93, 0.96))

    fig.supylabel(r"Relative Time ($t \:/\: t_{min}$)")
    fig.tight_layout()
    fig.savefig(f"plots/{benchname}{suff}.pdf", bbox_inches="tight")
    fig.savefig(f"plots/{benchname}{suff}.png", bbox_inches="tight")
    fig.savefig(f"plots/{benchname}{suff}.svg", bbox_inches="tight")


elif "comparisons" in benchname:
    n_max = df["n"].max()
    df = df[df["n"] == n_max].copy()

    df["push_comparisons"] /= df["normalization"]
    df["pop_comparisons"] /= df["normalization"]
    df["comparisons"] = df["push_comparisons"] + df["pop_comparisons"]
    ops = n_max * np.log2(n_max)

    # workloads = list(df["workload"].unique())
    # workloads = ['HeapSort', 'MonotoneConstantSize', 'MonotoneWiggle', 'ConstantSize', 'Wiggle']
    workloads = ['HeapSort', 'MonotoneConstantSize', 'MonotoneWiggle', 'Wiggle']

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

    filtered_methods = []

    for method in methods:
        if not method in nanos_filter:
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
    fig, ax = plt.subplots(figsize=(21/2.54, 5))
    x = np.arange(n_workloads)  # one tick per workload group

    legend_handles = [
        mpatches.Patch(
            facecolor="#333333FF", edgecolor="white", label="pop"
        ),
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

    ax.set_ylabel(r"Comparisons ($1 \:/\: (\mathsf{Ops} \cdot \lg n)$)")
    ax.grid(axis="y", linestyle="-", alpha=0.4)

    # Coloured method legend (top-left)
    method_handles = [
        mpatches.Patch(facecolor=bar_colors[mi], label=rewrite_legend(method))
        for mi, method in enumerate(methods)
    ]
    legend_methods = ax.legend(
        handles=method_handles, loc="upper left", fontsize=8, ncol=4
    )
    ax.add_artist(legend_methods)

    # Push / pop legend (top-right)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, ncol=1)

    fig.tight_layout()
    fig.savefig(f"plots/{benchname}{suff}.svg", bbox_inches="tight")
    fig.savefig(f"plots/{benchname}{suff}.pdf", bbox_inches="tight")
    fig.savefig(f"plots/{benchname}{suff}.png", bbox_inches="tight", dpi=300)

else:
    # Bench Plot

    # Take median over repeats, then normalize each metric by n*log2(n)
    metrics = [
        # ("nanos", r"$(\mathsf{ns} \:/\: \#(\mathsf{pop}\circ\mathsf{push})) \:/\: \lg n$"),
        ("nanos", r"Runtime ($\mathsf{ns} \:/\: (\mathsf{Ops} \cdot \lg n)$)"),
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

    # workloads = df["workload"].unique()
    workloads = ['HeapSort', 'MonotoneConstantSize', 'MonotoneWiggle', 'Wiggle']


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
        # if tp == "RadixHeapMap":
        #     return 1.0
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
                    if not all and tp in nanos_filter:
                        continue

                    c = type_colour[tp]

                    for name, ngroup in tgroup.groupby("name", sort=False):                        
                        if not all and name in nanos_filter:
                            continue
                        
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
                if not all and tp in nanos_filter:
                    continue
                c = type_colour[tp]
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

            axs[0, j].set_yscale("log", base=2)
            axs[0, j].set_xscale("log", base=2)
            axs[0, j].set_xticks([2**i for i in [10, 15, 20, 25]])
            axs[0, j].legend().remove()
            axs[0, j].set_xlabel(None)
            axs[0, j].grid(axis="both", which="both", linestyle="-")
            axs[0, j].set_ylabel(None)
        
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
