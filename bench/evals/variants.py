#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

print("Table creation for all SIMD QH variants.")
print("Reading all files.")

directory = Path("./data/variants")

for file_path in sorted(directory.glob("*.csv")):
    filename = file_path.name.split("_", 1)[1].replace(".csv", "")
    df = pd.read_csv(file_path)
    workloads = df["workload"].unique()
    ns = df["n"].unique()
    sep = " & "

    result = "\\multirow{4}{*}{" + filename + "} &"

    wl_sep = ""
    for workload in workloads:
        workload_line = wl_sep
        wl_sep = "\\cmidrule(lr){2-11} &\n"
        df_wl = df[df["workload"] == workload]
        workload_line += "\\multirow{2}{*}{" + workload + "} & Time (ns) "

        time = ""
        for t in df_wl["nanoseconds"]:
            time += "& "
            time += str(t)
        time += "\\\\\n"
        workload_line += time


        cache = "& & Cache Misses "
        for c in df_wl["cache_misses"]:
            cache += "& "
            cache += str(c)
        cache += "\\\\\n"
        workload_line += cache
        result += workload_line

    result += "\\cmidrule(lr){1-11}\n"

    print(result)
