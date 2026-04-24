#!/usr/bin/env python3

import numpy as np

graph_names = ["GER"]
# graph_names = ["CAL", "CTR", "GER", "USA", "rhg_20", "rhg_22", "rhg_24"]

for graph in graph_names:
    f = open("./input/" + graph + "_graph.gr")
    vals = []
    max = 0
    min = 100000000000

    for x in f:
        parts = x.split(" ")
        if parts[0] != "a":
            continue
        w = parts[3]
        vals.append(int(w))

    print(graph, "MAX:", np.max(vals), "MEDIAN", np.median(vals))



    


