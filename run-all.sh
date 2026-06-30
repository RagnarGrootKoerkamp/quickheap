#!/bin/sh
cargo run -r --package bench --example bench_graph -F avx512 >./bench/evals/data/results-graphs.csv
cargo run -r --package bench --example bench -F avx512 -- --ablation >./bench/evals/data/results-ablation.csv
cargo run -r --package bench --example bench -F avx512 -- --comparisons >./bench/evals/data/results-comparisons.csv
cargo run -r --package bench --example bench -F avx512,perf,boost -- --table >./bench/evals/data/results-table.csv
cargo run -r --package bench --example bench -F avx512 >./bench/evals/data/results-plot.csv
