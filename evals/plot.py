#!/usr/bin/env python3
import matplotlib.pyplot as plt
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
df["name"] = df["name"].str.replace(r"core::cmp::Reverse<i(32|64)>", "T", regex=True)
df["name"] = df["name"].str.replace(
    "orx_priority_queue::dary::daryheap::DaryHeap", "DaryHeapOrx", regex=False
)
df["name"] = df["name"].str.replace("dary_heap::", "", regex=False)
df["name"] = df["name"].str.replace("pheap::ph::", "", regex=False)
df["name"] = df["name"].str.replace("fibonacci_heap::", "", regex=False)
df["name"] = df["name"].str.replace("weakheap::", "", regex=False)
df["name"] = df["name"].str.replace("radix_heap::", "", regex=False)
df["name"] = df["name"].str.replace(r"\bi(32|64)\b", "T", regex=True)
df["name"] = df["name"].str.replace(r"I(32|64)", "<T>", regex=True)
df["name"] = df["name"].str.replace(r"<\(\), T", "<T", regex=True)
df["name"] = df["name"].str.replace(r", \(\)", "", regex=True)
df["name"] = df["name"].str.replace(r", 16, 1", "", regex=True)
df["type"] = df["name"].str.split("<").str[0]

# Shorten workload names
df["workload"] = df["workload"].str.split("::").str[-1]

# Filter out some overly slow heaps
# df = df[
#     ~df["name"].str.contains(
#         "FibonacciHeap|PairingHeap|WeakHeap|DaryHeap<T, (2|4|8|16)>|DaryHeapOrx<T(|, 16)>"
#     )
# ]

df = df[~df["name"].str.contains("SimdQuickHeap<T, Avx512>")]
df["name"] = df["name"].str.replace(r"<true>", "", regex=True)


# Take median over repeats, then normalize each metric by n*log2(n)
metrics = [
    ("nanos", r"ns / ($\mathsf{push}\circ\mathsf{pop}) / \lg n$"),
    # ("branch_misses", r"branch misses / ($\mathsf{push}\circ\mathsf{pop}) / \lg n$"),
    # (
    #     "l1_cache_misses",
    #     r"L1 cache misses / ($\mathsf{push}\circ\mathsf{pop}) / \lg n$",
    # ),
    # # (
    # #     "hw_cache_misses",
    # #     r"HW cache misses / ($\mathsf{push}\circ\mathsf{pop}) / \lg n$",
    # # ),
    # (
    #     "l3_cache_misses",
    #     r"L3 cache misses / ($\mathsf{push}\circ\mathsf{pop}) / \lg n$",
    # ),
]

df["push_comparisons"], df["comparisons"] = df["comparisons"], df["push_comparisons"]
df["pop_comparisons"] = df["comparisons"] - df["push_comparisons"]
metric_cols = [m for m, _ in metrics] + [
    "comparisons",
    "push_comparisons",
    "pop_comparisons",
]


if df["nanos"].max() == 0:
    # only report comparisons
    metrics = [
        ("comparisons", r"comparisons / ($\mathsf{push}\circ\mathsf{pop}) / \lg n$")
    ]


df = (
    df.groupby(["elem", "name", "type", "n", "workload"])[metric_cols]
    .median()
    .reset_index()
)


# Sort by type
type_order = [
    "BinaryHeap",
    "RadixHeapMap",
    "DaryHeapOrx",
    "SequenceHeap",
    "S3qHeap",
    "SimdQuickHeap",
]
df["order"] = pd.Categorical(df["type"], categories=type_order, ordered=True)
df = df.sort_values(["order", "name"])
ops = df["n"] * np.log2(df["n"])
for m in metric_cols:
    df[m] = df[m] / ops

workloads = ["HeapSort", "ConstantSize", "Decreasing"]
elems = ["i32", "i64"]

colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
styles = ["-", "--", "-.", ":"]
widths = [1.5, 1.5, 1.5]


all_types = df["type"].unique()
type_colour = {tp: colours[k % len(colours)] for k, tp in enumerate(all_types)}
# type_colour["DaryHeapOrx"] = "black"
# blogyellow = "#fcc007"
type_colour["SimdQuickHeap"] = "black"
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
        ncol=4,
        bbox_to_anchor=(0.5, -0.10),
    )
    fig.supxlabel("n = max #elements in heap", y=0.02)

    fig.savefig(f"plot-{benchname}-{metric}.svg", bbox_inches="tight")
    fig.savefig(f"plot-{benchname}-{metric}.png", bbox_inches="tight", dpi=300)
