default:

# Run the one-off testing binary.
run:
    cd bench && cargo run -r --example run

bench-simd:
    cd bench && cargo run -r --example bench_simd > evals/data/simd.csv

pivots:
    cd bench && cargo run -r --example pivots -F pivots > evals/data/pivots.csv

rebalancing:
    cd bench && cargo run -r --example rebalancing -F rebalancing > evals/data/rebal.csv

# Run all benchmarks. Takes a bunch of hours.
bench-all:
    cd bench && cargo run -r --example bench       -F avx512                  > evals/data/nanos.csv
    cd bench && cargo run -r --example bench       -F avx512 -- --comparisons > evals/data/comparisons.csv
    cd bench && cargo run -r --example bench_graph -F avx512                  > evals/data/graphs.csv

# A smaller benchmark for testing.
bench-small:
    cd bench && cargo run -r --example bench       -- --max 18               > evals/data/nanos.csv
    cd bench && cargo run -r --example bench       -- --max 18 --comparisons > evals/data/comparisons.csv
    # cd bench && cargo run -r --example bench_graph                           > evals/data/graphs.csv

# Generate all plots.
# See bench/evals/plots/
plot-all:
    cd bench/evals && ./plot.py nanos
    cd bench/evals && ./plot.py nanos all
    cd bench/evals && ./plot.py comparisons
    cd bench/evals && ./plot.py graphs

plot-nanos:
    cd bench/evals && ./plot.py nanos
    cd bench/evals && ./plot.py nanos all
