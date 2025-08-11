#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd


df = pd.read_csv(
    "data.tsv",
    header=None,
    names=["name", "incr", "n", "r0", "r4"],
    sep="\t",
)

for i in [0, 4]:
    df[f"r{i}"] = df[f"r{i}"].apply(lambda x: x / (1 + 2 * i))

df["type"] = df["name"].str.split("<").str[0]


# Figure with 4 subplots side by side
fig, axs = plt.subplots(2, 2, figsize=(20, 5), sharey=True)

for j, incr in enumerate([False, True]):
    df2 = df[df.incr == incr].pivot_table(
        index=["n"], columns="type", values=["r0", "r4"], aggfunc="min", sort=False
    )

    for i, col in enumerate([0, 4]):
        df2.plot(
            kind="line",
            y=f"r{col}",
            logx=True,
            logy=True,
            ax=axs[j][i],
            title=f"(ins (del ins)^{i})^m (del (ins del)^{i})^m",
        )
        axs[j][i].set_yscale("log", base=2)
        axs[j][i].set_xscale("log", base=2)
        if i > 0:
            axs[j][i].legend().remove()

plt.show()
