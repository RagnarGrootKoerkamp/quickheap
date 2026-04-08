#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("test.csv")

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
df["name"] = df["name"].str.replace(r"<\(\), T", "<T", regex=True)
df["name"] = df["name"].str.replace(r", \(\)>", ">", regex=True)
df["type"] = df["name"].str.split("<").str[0]

# Shorten workload names
df["workload"] = df["workload"].str.split("::").str[-1]

# Take median over repeats, then normalize each metric by n*log2(n)
metrics = [
    ("nanos", r"ns / ($\mathsf{push}\circ\mathsf{pop}) / \lg n$"),
    ("comparisons", r"comparisons / ($\mathsf{push}\circ\mathsf{pop}) / \lg n$"),
    ("branch_misses", r"branch misses / ($\mathsf{push}\circ\mathsf{pop}) / \lg n$"),
    (
        "l1_cache_misses",
        r"L1 cache misses / ($\mathsf{push}\circ\mathsf{pop}) / \lg n$",
    ),
    # (
    #     "hw_cache_misses",
    #     r"HW cache misses / ($\mathsf{push}\circ\mathsf{pop}) / \lg n$",
    # ),
    # (
    #     "l3_cache_misses",
    #     r"L3 cache misses / ($\mathsf{push}\circ\mathsf{pop}) / \lg n$",
    # ),
]
metric_cols = [m for m, _ in metrics]
df = (
    df.groupby(["elem", "name", "type", "n", "workload"])[metric_cols]
    .median()
    .reset_index()
)
ops = df["n"] * np.log2(df["n"])
for m in metric_cols:
    df[m] = df[m] / ops

workloads = ["HeapSort", "ConstantSize", "Decreasing"]
elems = ["i32", "i64"]

colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
styles = ["-", "--", "-.", ":"]

plt.close("all")
fig = plt.figure(figsize=(10, 8 * len(metrics)))
subfigs = fig.subfigures(len(metrics), 1, hspace=0.15)

for m_idx, (subfig, (metric, label)) in enumerate(zip(subfigs, metrics)):
    subfig.suptitle(label)
    axs = subfig.subplots(2, 3, sharex=True, sharey=True)

    for j, elem in enumerate(elems):
        edf = df[df["elem"] == elem]
        for i, workload in enumerate(workloads):
            wdf = edf[edf["workload"] == workload]
            for c, (tp, tgroup) in zip(colours, wdf.groupby("type")):
                for ls, (name, ngroup) in zip(
                    styles, tgroup.groupby("name", sort=False)
                ):
                    ngroup.plot(
                        kind="line",
                        x="n",
                        y=metric,
                        logx=True,
                        ax=axs[j][i],
                        title=workload if j == 0 else None,
                        label=name if m_idx == 0 and i == 0 and j == 0 else None,
                        color=c,
                        ls=ls,
                    )
            axs[j][i].set_yscale("log", base=2)
            axs[j][i].set_xscale("log", base=2)
            axs[j][i].legend().remove()
            axs[j][i].set_xlabel(None)
            axs[j][i].grid(axis="y", which="both", linestyle="-")
            if j > 0:
                axs[j][i].set_title(None)

        axs[j][0].set_ylabel(elem)

    axs[0][0].set_xticks([2**i for i in [10, 15, 20]])

handles, labels = subfigs[0].get_axes()[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.05),
)
fig.supxlabel("n = max #elements in heap", y=0.01)

fig.savefig("plot.svg", bbox_inches="tight")
