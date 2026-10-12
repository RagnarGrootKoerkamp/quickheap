#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

directory = Path("./data/variants")

for file_path in sorted(directory.glob("*.csv")):
    filename = file_path.name.split("_", 1)[1].replace(".csv", "")
    df = pd.read_csv(file_path)
    workloads = df["workload"].unique()
    ns = df["n"].unique()
    sep = " & "

    num_workloads = len(workloads)
    num_features = len(df.columns) - 2 # Num of columns minus col n and col workload


    result_col_str = str(num_workloads * num_features)
    result = "\\multirow{" + result_col_str + "}{*}{" + filename + "} &"

    wl_sep = ""
    for workload in workloads:
        workload_line = wl_sep
        wl_sep = "\\cmidrule(lr){2-11} &\n"
        df_wl = df[df["workload"] == workload]
        workload_line += "\\multirow{" + str(num_features) + "}{*}{" + workload + "}"

        time = " & Time (ns) "
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

        ipc = "& & Instructions per Cycle "
        for c in df_wl["inst_per_cycle"]:
            ipc += "& "
            ipc += str(c)
        ipc += "\\\\\n"
        workload_line += ipc

        mem = "& & Bytes per Elem "
        for c in df_wl["bytes_per_elem"]:
            mem += "& "
            mem += str(c)
        mem += "\\\\\n"
        workload_line += mem

        result += workload_line

    result += "\\cmidrule(lr){1-11}\n"

    print(result)
