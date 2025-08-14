#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd

cols = [
    "Linear",
    "Increasing",
    "Heapsort",
    "Random",
]
displaycols = [
    "Heapsort",
    "Random",
    "Linear",
    "Increasing",
]

dfs = []

for suf in ["32", "64"]:
    df = pd.read_csv(
        f"data{suf}.tsv",
        header=None,
        names=["name", "T", "n"] + cols,
        sep="\t",
    )

    df["name"] = df["name"].str.replace("alloc::collections::binary_heap::", "")
    df["name"] = df["name"].str.replace("core::cmp::Reverse<u32>", "u32")
    df["name"] = df["name"].str.replace("radix_heap::", "")
    df["name"] = df["name"].str.replace("dary::daryheap::", "")
    df["name"] = df["name"].str.replace("quickheap::", "")
    df["name"] = df["name"].str.replace("simd_heap::", "")
    df["name"] = df["name"].str.replace("BucketHeap", "QuickHeap")
    df["name"] = df["name"].str.replace("u32", "T")
    df["name"] = df["name"].str.replace("dary_heap::DaryHeap", "DaryHeap")
    df["name"] = df["name"].str.replace("orx_priority_queue::DaryHeap", "DaryHeapOrx")
    df["type"] = df["name"].str.split("<").str[0]
    dfs.append(df)


# Figure with 4 subplots side by side
plt.close()
fig, axs = plt.subplots(2, 4, figsize=(10, 6), sharex=True, sharey=True)

colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
styles = ["-", "--", "-.", ":"]

for j in range(2):

    for i, col in enumerate(displaycols):
        groups = dfs[j].groupby("type")
        for c, (tp, group) in zip(colours, groups):
            groups = group.groupby("name", sort=False)

            # Next colour of the default palette
            for ls, (name, group) in zip(styles, groups):
                group.plot(
                    kind="line",
                    x="n",
                    y=col,
                    logx=True,
                    ax=axs[j][i],
                    title=col,
                    label=name if i == 0 and j == 0 else None,
                    color=c,
                    ls=ls,
                )
        axs[j][i].set_yscale("log", base=2)
        # axs[j][i].set_ylim(0, 120)
        axs[j][i].set_xscale("log", base=2)
        # if i > 0:
        axs[j][i].legend().remove()
        axs[j][i].set_xlabel(None)

        # show x grid
        axs[j][i].grid(axis="y", which="both", linestyle="-")
        if j > 0:
            axs[j][i].set_title(None)
    axs[0][0].set_ylabel("u32")
    axs[1][0].set_ylabel("u64")

axs[0][0].set_xticks([2**i for i in [10, 15, 20, 25]])

handles, labels = axs[0][0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2,
    bbox_to_anchor=(0.5, -0.11),
)
fig.supxlabel("n = max #elements in heap", y=0.03)
fig.supylabel("ns / ($\mathsf{push}\circ\mathsf{pop}$)", x=0.06)

# fig.show()
fig.savefig(f"plot.svg", bbox_inches="tight")
