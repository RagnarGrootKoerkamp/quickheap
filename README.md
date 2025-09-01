# QuickHeap

An implementation of the quickheap.
See the corresponding blog post for details:
https://curiouscoding.nl/posts/quickheap.

The quickheap was first published in:

> On Sorting, Heaps, and Minimum Spanning Trees\
> Gonzalo Navarro and Rodrigo Paredes, Algorithmica, 2010\
> http://doi.org/10.1007/s00453-010-9400-6

with some improvements to avoid worst-case linear updates in

> Stronger Quickheaps\
> Gonzalo Navarro and Rodrigo Paredes and Patricia V. Poblete and Peter Sanders\
> International Journal of Foundations of Computer Science, 2011\
> http://doi.org/10.1142/S0129054111008507

## Crate

The v0.0.1 crates.io library is very preliminary, and mostly a placeholder.
For now, it depends on unstable Rust for `portable-simd`, and requires AVX2 instructions.
By default, it only supports queues of `u32` values.
Enable the `u64` feature to store `u64` values instead.
Making this into a generic API and supporting more types is future work.
Please submit an issue if your specific type is not supported.
