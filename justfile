default:


bench-all:
    cargo run -r --example bench -F avx512 > evals/test-diffie.csv
    cargo run -r --example bench -F avx512 -- --comparisons > evals/test-comparisons.csv
    cargo run -r --example bench_graph -F avx512 > evals/test-graphs-diffie.csv

run:
    # cargo run -r --example run > evals/run.csv
    cargo run -r --example run

plot-all:
    cd evals && ./plot.py diffie
    cd evals && ./plot.py diffie all
    cd evals && ./plot.py comparisons
    cd evals && ./plot.py graphs-diffie
