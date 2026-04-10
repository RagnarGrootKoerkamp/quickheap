#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SeqHeapI32 SeqHeapI32;

SeqHeapI32 *seq_heap_i32_new(void);
void seq_heap_i32_free(SeqHeapI32 *pq);
void seq_heap_i32_push(SeqHeapI32 *pq, int32_t key);
int32_t seq_heap_i32_pop(SeqHeapI32 *pq);
int32_t seq_heap_i32_top(SeqHeapI32 *pq);
int seq_heap_i32_size(const SeqHeapI32 *pq);
bool seq_heap_i32_empty(const SeqHeapI32 *pq);

typedef struct SeqHeapI64 SeqHeapI64;

SeqHeapI64 *seq_heap_i64_new(void);
void seq_heap_i64_free(SeqHeapI64 *pq);
void seq_heap_i64_push(SeqHeapI64 *pq, int64_t key);
int64_t seq_heap_i64_pop(SeqHeapI64 *pq);
int64_t seq_heap_i64_top(SeqHeapI64 *pq);
int seq_heap_i64_size(const SeqHeapI64 *pq);
bool seq_heap_i64_empty(const SeqHeapI64 *pq);

typedef struct SeqHeapU32 SeqHeapU32;

SeqHeapU32 *seq_heap_u32_new(void);
void seq_heap_u32_free(SeqHeapU32 *pq);
void seq_heap_u32_push(SeqHeapU32 *pq, uint32_t key);
uint32_t seq_heap_u32_pop(SeqHeapU32 *pq);
uint32_t seq_heap_u32_top(SeqHeapU32 *pq);
int seq_heap_u32_size(const SeqHeapU32 *pq);
bool seq_heap_u32_empty(const SeqHeapU32 *pq);

typedef struct SeqHeapU64 SeqHeapU64;

SeqHeapU64 *seq_heap_u64_new(void);
void seq_heap_u64_free(SeqHeapU64 *pq);
void seq_heap_u64_push(SeqHeapU64 *pq, uint64_t key);
uint64_t seq_heap_u64_pop(SeqHeapU64 *pq);
uint64_t seq_heap_u64_top(SeqHeapU64 *pq);
int seq_heap_u64_size(const SeqHeapU64 *pq);
bool seq_heap_u64_empty(const SeqHeapU64 *pq);

#ifdef __cplusplus
}
#endif
