default:


bench-all:
    cd bench && cargo run -r --example bench -F avx512 > evals/data/nanos.csv
    cd bench && cargo run -r --example bench -F avx512 -- --comparisons > evals/data/comparisons.csv
    cd bench && cargo run -r --example bench_graph -F avx512 > evals/data/graphs.csv

run:
    cd bench && cargo run -r --example run

plot-all:
    cd bench/evals && ./plot.py nanos
    cd bench/evals && ./plot.py nanos all
    cd bench/evals && ./plot.py comparisons
    cd bench/evals && ./plot.py graphs

plot-nanos:
    cd bench/evals && ./plot.py nanos
    cd bench/evals && ./plot.py nanos all
