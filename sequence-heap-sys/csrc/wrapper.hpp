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

#ifdef __cplusplus
}
#endif
