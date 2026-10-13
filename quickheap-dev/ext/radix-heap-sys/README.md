# radix-heap-sys

Rust FFI bindings for the radix heap implementation from https://github.com/iwiwi/radix-heap

## Structure

This crate follows the same structure as `randomized-quickheap-sys`:

- `third_party/radix-heap/` - The original C++ header-only library (git submodule)
- `csrc/` - C++ wrapper files that provide a C-compatible FFI interface
  - `wrapper.hpp` - Header file declaring the C API
  - `wrapper.cpp` - Implementation wrapping the C++ radix_heap template
- `src/lib.rs` - Rust FFI declarations and safe wrappers
- `build.rs` - Build script to compile the C++ wrapper using the `cc` crate

## API

The bindings provide monotone priority queues (radix heaps) for:
- `RadixHeapI32` - signed 32-bit integers
- `RadixHeapI64` - signed 64-bit integers
- `RadixHeapU32` - unsigned 32-bit integers
- `RadixHeapU64` - unsigned 64-bit integers

Each type provides:
- `*_new()` - allocate a new heap
- `*_free()` - deallocate the heap
- `*_push()` - insert an element (must be >= last popped element)
- `*_pop()` - remove and return the minimum element
- `*_top()` - return the minimum element without removing it
- `*_size()` - return the number of elements
- `*_empty()` - check if the heap is empty
- `*_clear()` - remove all elements

## Note

Radix heaps are monotone priority queues - inserted elements must be >= the last popped element. This makes them very efficient for applications like Dijkstra's algorithm where distances are non-decreasing.
