#!/bin/sh

# Setup for the specific architectures
arch=$(uname -m)
export RUSTUP_HOME=$HOME/.rustup-$arch
export CARGO_HOME=$HOME/.cargo-$arch
export PATH=$CARGO_HOME/bin:$PATH
export CARGO_TARGET_DIR=$HOME/target-$(uname -m)

# Run all experiments
# cargo run -r -q  --package bench --example experiments -F implicit_equal_buckets
# cargo run -r -q  --package bench --example experiments -F pivots
# cargo run -r -q --package bench --example experiments -F buckets -F perf
# cargo run -r -q --package bench --example experiments -F simd_qh
# cargo run -r -q  --package bench --example experiments -F multiway -F perf
# cargo run -r -q  --package bench --example experiments -F radix -F perf
# cargo run -r -q  --package bench --example experiments -F rebalancing

cargo run -r -q --package bench --example partitions -F only_equals
cargo run -r -q --package bench --example partitions

# cargo run -r --package bench --example experiments -F graphs
# cargo run -r --package bench --example experiments -F decrease_key
# cargo run -r --package bench --example experiments -F competitors
