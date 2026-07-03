# QuickHeap C++ bindings

This directory provides C++ bindings for `SimdQuickHeap`, built on top of a
thin `extern "C"` layer (`src/c.rs`, enabled via the `c` Cargo feature).

There are two headers:

- [`quickheap.h`](quickheap.h): the raw C-ABI, generated from `src/c.rs` with
  [`cbindgen`](https://github.com/mozilla/cbindgen). One `quickheap_<ty>_*`
  family of functions per supported element type (`i32`, `u32`, `i64`, `u64`).
- [`quickheap.hpp`](quickheap.hpp): a small header-only RAII wrapper around
  `quickheap.h`, exposing a `quickheap::Heap<T>` class with an interface
  similar to `std::priority_queue` (but returning the *smallest* element
  first, like the underlying Rust type).

Prefer including `quickheap.hpp` unless you specifically need the raw C API.

## Building

First build the Rust library with the `c` feature enabled, which produces a
shared (and static) library exposing the `quickheap_*` symbols:

```bash
cargo build --release --features c
```

This produces `target/release/libquickheap.{so,dylib}` (and
`libquickheap.a`).

Then compile [`example.cpp`](example.cpp) against it:

```bash
g++ -std=c++17 -I cpp cpp/example.cpp \
    -L target/release -lquickheap \
    -o quickheap_example
```

And run it (adjusting the dynamic linker path for your platform):

```bash
# Linux
LD_LIBRARY_PATH=target/release ./quickheap_example
# macOS
DYLD_LIBRARY_PATH=target/release ./quickheap_example
```

## Example usage

```cpp
#include "quickheap.hpp"

int main() {
    quickheap::Heap<std::uint64_t> q;
    q.push(4);
    q.push(1);
    q.push(7);
    assert(q.pop() == 1);   // smallest first
    q.push(7);
    q.push(3);
    assert(q.pop() == 3);
    assert(q.size() == 3);
    assert(!q.empty());

    // try_pop() returns std::optional<T> instead of asserting on an
    // empty heap.
    while (auto v = q.try_pop()) {
        // ...
    }
}
```

`Heap<T>` is move-only (copying would require deep-cloning the underlying
Rust heap, which isn't currently exposed) and supports `int32_t`,
`uint32_t`, `int64_t`, and `uint64_t`.

## Regenerating `quickheap.h`

`quickheap.h` is checked in, but can be regenerated from `src/c.rs` with:

```bash
cargo install cbindgen
cbindgen --config cbindgen.toml --output cpp/quickheap.h
```

## Notes

- Like the rest of the crate, this requires AVX2 or AVX-512 to be available
  at compile time (see the top-level [README](../README.md)).
- All functions in `quickheap.h` assert (Rust panic) rather than segfault on
  null-pointer misuse; linking against a `panic = "abort"` profile is
  recommended for C/C++ consumers so a Rust panic doesn't unwind across the
  FFI boundary.
