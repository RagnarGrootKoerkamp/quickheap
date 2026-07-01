#pragma once

#include <cstddef>
#include <cstdint>

typedef struct RadixHeapI32 RadixHeapI32;
typedef struct RadixHeapI64 RadixHeapI64;
typedef struct RadixHeapU32 RadixHeapU32;
typedef struct RadixHeapU64 RadixHeapU64;

extern "C" {

RadixHeapI32 *radix_heap_i32_new();
void radix_heap_i32_free(RadixHeapI32 *heap);
void radix_heap_i32_push(RadixHeapI32 *heap, int32_t item);
int32_t radix_heap_i32_pop(RadixHeapI32 *heap);
int32_t radix_heap_i32_top(RadixHeapI32 *heap);
size_t radix_heap_i32_size(const RadixHeapI32 *heap);
bool radix_heap_i32_empty(const RadixHeapI32 *heap);
void radix_heap_i32_clear(RadixHeapI32 *heap);

RadixHeapI64 *radix_heap_i64_new();
void radix_heap_i64_free(RadixHeapI64 *heap);
void radix_heap_i64_push(RadixHeapI64 *heap, int64_t item);
int64_t radix_heap_i64_pop(RadixHeapI64 *heap);
int64_t radix_heap_i64_top(RadixHeapI64 *heap);
size_t radix_heap_i64_size(const RadixHeapI64 *heap);
bool radix_heap_i64_empty(const RadixHeapI64 *heap);
void radix_heap_i64_clear(RadixHeapI64 *heap);

RadixHeapU32 *radix_heap_u32_new();
void radix_heap_u32_free(RadixHeapU32 *heap);
void radix_heap_u32_push(RadixHeapU32 *heap, uint32_t item);
uint32_t radix_heap_u32_pop(RadixHeapU32 *heap);
uint32_t radix_heap_u32_top(RadixHeapU32 *heap);
size_t radix_heap_u32_size(const RadixHeapU32 *heap);
bool radix_heap_u32_empty(const RadixHeapU32 *heap);
void radix_heap_u32_clear(RadixHeapU32 *heap);

RadixHeapU64 *radix_heap_u64_new();
void radix_heap_u64_free(RadixHeapU64 *heap);
void radix_heap_u64_push(RadixHeapU64 *heap, uint64_t item);
uint64_t radix_heap_u64_pop(RadixHeapU64 *heap);
uint64_t radix_heap_u64_top(RadixHeapU64 *heap);
size_t radix_heap_u64_size(const RadixHeapU64 *heap);
bool radix_heap_u64_empty(const RadixHeapU64 *heap);
void radix_heap_u64_clear(RadixHeapU64 *heap);

} // extern "C"
