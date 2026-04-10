default:


bench-all:
    cargo run -r --example bench -F avx512 > evals/test-diffie.csv
    cargo run -r --example bench -F avx512 -- --comparisons > evals/test-comparisons.csv
    cargo run -r --example bench_graph -F avx512 > evals/test-graphs-diffie.csv
