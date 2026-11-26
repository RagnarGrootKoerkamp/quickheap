#!/bin/sh
cargo run -r --package bench --example experiments -F implicit_equal_buckets
cargo run -r --package bench --example experiments -F pivots
cargo run -r --package bench --example experiments -F rebalancing
# cargo run -r --package bench --example experiments -F multiway
# cargo run -r --package bench --example experiments -F radix

# cargo run -r --package bench --example experiments -F graphs
# cargo run -r --package bench --example experiments -F decrease_key
# cargo run -r --package bench --example experiments -F competitors
# cargo run -r --package bench --example experiments -F simd_qh
# cargo run -r --package bench --example experiments -F buckets
