#!/usr/bin/env python3

import sys

if len(sys.argv) < 2:
    print("Please specify a graph instance in the input dir.")
    exit()

gn = sys.argv[1]

f = open("./input/" + gn)

max = 0
min = 1000000

for x in f:
    parts = x.split(" ")
    if parts[0] != "a":
        continue
    w = parts[3]
    if int(w) > max:
        max = int(w)
    if int(w) < min:
        min = int(w)

print(gn, "MAX:", max, "MIN:", min)
