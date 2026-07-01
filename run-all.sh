#!/bin/sh
cargo run -r --package bench --example bench_graph -F avx512 >./bench/evals/data/graphs.csv
cargo run -r --package bench --example bench -F avx512 -- --memory 2>./bench/evals/data/memory.txt
cargo run -r --package bench --example bench -F avx512 -- --ablation >./bench/evals/data/ablation.csv
cargo run -r --package bench --example bench -F avx512 -- --comparisons >./bench/evals/data/comparisons.csv
cargo run -r --package bench --example bench -F avx512 >./bench/evals/data/plot.csv
cargo run -r --package bench --example bench -F avx512,boost,perf -- --table >./bench/evals/data/table.csv
