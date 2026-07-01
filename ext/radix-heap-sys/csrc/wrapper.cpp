#include "wrapper.hpp"

#include "../third_party/radix-heap/radix_heap.h"

#include <cstdint>
#include <new>

using I32Heap = radix_heap::radix_heap<int32_t>;
using I64Heap = radix_heap::radix_heap<int64_t>;
using U32Heap = radix_heap::radix_heap<uint32_t>;
using U64Heap = radix_heap::radix_heap<uint64_t>;

extern "C" {

// I32 Heap
RadixHeapI32 *radix_heap_i32_new() {
    return reinterpret_cast<RadixHeapI32 *>(new (std::nothrow) I32Heap());
}
void radix_heap_i32_free(RadixHeapI32 *heap) {
    delete reinterpret_cast<I32Heap *>(heap);
}
void radix_heap_i32_push(RadixHeapI32 *heap, int32_t item) {
    reinterpret_cast<I32Heap *>(heap)->push(item);
}
int32_t radix_heap_i32_pop(RadixHeapI32 *heap) {
    auto h = reinterpret_cast<I32Heap *>(heap);
    int32_t result = h->top();
    h->pop();
    return result;
}
int32_t radix_heap_i32_top(RadixHeapI32 *heap) {
    return reinterpret_cast<I32Heap *>(heap)->top();
}
size_t radix_heap_i32_size(const RadixHeapI32 *heap) {
    return reinterpret_cast<const I32Heap *>(heap)->size();
}
bool radix_heap_i32_empty(const RadixHeapI32 *heap) {
    return reinterpret_cast<const I32Heap *>(heap)->empty();
}
void radix_heap_i32_clear(RadixHeapI32 *heap) {
    reinterpret_cast<I32Heap *>(heap)->clear();
}

// I64 Heap
RadixHeapI64 *radix_heap_i64_new() {
    return reinterpret_cast<RadixHeapI64 *>(new (std::nothrow) I64Heap());
}
void radix_heap_i64_free(RadixHeapI64 *heap) {
    delete reinterpret_cast<I64Heap *>(heap);
}
void radix_heap_i64_push(RadixHeapI64 *heap, int64_t item) {
    reinterpret_cast<I64Heap *>(heap)->push(item);
}
int64_t radix_heap_i64_pop(RadixHeapI64 *heap) {
    auto h = reinterpret_cast<I64Heap *>(heap);
    int64_t result = h->top();
    h->pop();
    return result;
}
int64_t radix_heap_i64_top(RadixHeapI64 *heap) {
    return reinterpret_cast<I64Heap *>(heap)->top();
}
size_t radix_heap_i64_size(const RadixHeapI64 *heap) {
    return reinterpret_cast<const I64Heap *>(heap)->size();
}
bool radix_heap_i64_empty(const RadixHeapI64 *heap) {
    return reinterpret_cast<const I64Heap *>(heap)->empty();
}
void radix_heap_i64_clear(RadixHeapI64 *heap) {
    reinterpret_cast<I64Heap *>(heap)->clear();
}

// U32 Heap
RadixHeapU32 *radix_heap_u32_new() {
    return reinterpret_cast<RadixHeapU32 *>(new (std::nothrow) U32Heap());
}
void radix_heap_u32_free(RadixHeapU32 *heap) {
    delete reinterpret_cast<U32Heap *>(heap);
}
void radix_heap_u32_push(RadixHeapU32 *heap, uint32_t item) {
    reinterpret_cast<U32Heap *>(heap)->push(item);
}
uint32_t radix_heap_u32_pop(RadixHeapU32 *heap) {
    auto h = reinterpret_cast<U32Heap *>(heap);
    uint32_t result = h->top();
    h->pop();
    return result;
}
uint32_t radix_heap_u32_top(RadixHeapU32 *heap) {
    return reinterpret_cast<U32Heap *>(heap)->top();
}
size_t radix_heap_u32_size(const RadixHeapU32 *heap) {
    return reinterpret_cast<const U32Heap *>(heap)->size();
}
bool radix_heap_u32_empty(const RadixHeapU32 *heap) {
    return reinterpret_cast<const U32Heap *>(heap)->empty();
}
void radix_heap_u32_clear(RadixHeapU32 *heap) {
    reinterpret_cast<U32Heap *>(heap)->clear();
}

// U64 Heap
RadixHeapU64 *radix_heap_u64_new() {
    return reinterpret_cast<RadixHeapU64 *>(new (std::nothrow) U64Heap());
}
void radix_heap_u64_free(RadixHeapU64 *heap) {
    delete reinterpret_cast<U64Heap *>(heap);
}
void radix_heap_u64_push(RadixHeapU64 *heap, uint64_t item) {
    reinterpret_cast<U64Heap *>(heap)->push(item);
}
uint64_t radix_heap_u64_pop(RadixHeapU64 *heap) {
    auto h = reinterpret_cast<U64Heap *>(heap);
    uint64_t result = h->top();
    h->pop();
    return result;
}
uint64_t radix_heap_u64_top(RadixHeapU64 *heap) {
    return reinterpret_cast<U64Heap *>(heap)->top();
}
size_t radix_heap_u64_size(const RadixHeapU64 *heap) {
    return reinterpret_cast<const U64Heap *>(heap)->size();
}
bool radix_heap_u64_empty(const RadixHeapU64 *heap) {
    return reinterpret_cast<const U64Heap *>(heap)->empty();
}
void radix_heap_u64_clear(RadixHeapU64 *heap) {
    reinterpret_cast<U64Heap *>(heap)->clear();
}

} // extern "C"
