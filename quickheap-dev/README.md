## Running the benchmarks

The main entrypoint for running all benchmarks is the [`bench/run-all.sh`](bench/run-all.sh)
script (or `just bench-all`). This runs `bench/examples/{bench,bench_graph}.rs`
and writes to `bench/evals/data/*.csv`. (`just bench-small` runs a
smaller/faster subset.)

After that is done, `just plot` creates plots in `bench/evals/plots/*.{png,svg,pdf}`.

We benchmark both Rust libraries and (LLM-generated) FFI wrappers around
external C++ implementations that can be found in [ext/](ext/).
Submodules
containing the original C++ sources are in  nested `third_party` directories.
(Run `git submodule update --init --recursive` to initialize them.)

### Build errors
- The benchmarks need nightly Rust for some unstable convenience features.
- If you get linker errors, comment out the `.flag("-flto")` in `ext/{boost-heap-sys,s3q-sys,sequence-heap-sys,original-quickheap-sys}/build.rs`.
- To include boost implementations, add `-F boost`. In `spack`:
  - `spack env activate .` to activate the environment.
  - `spack find --path | grep boost` to get the path to the boost libraries
  - `export BOOST_INCLUDE_DIR=<result>/include`
  - `export CXX_FLAGS="-I$BOOST_INCLUDE_DIR"`
