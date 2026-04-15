#!/bin/sh
cargo run -r --example bench -F avx512 > ./test-diffie.csv
cargo run -r --example bench -F avx512 -- --comparisons > ./test-comparisons.csv
cargo run -r --example bench_graph -F avx512 > ./test-graphs-diffie.csv