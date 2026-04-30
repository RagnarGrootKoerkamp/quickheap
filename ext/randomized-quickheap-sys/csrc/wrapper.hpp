#pragma once

#include <cstddef>
#include <cstdint>

typedef struct Rqh2I32Pq Rqh2I32Pq;
typedef struct Rqh2I64Pq Rqh2I64Pq;
typedef struct Rqh2U32Pq Rqh2U32Pq;
typedef struct Rqh2U64Pq Rqh2U64Pq;

extern "C" {

Rqh2I32Pq *rqh2_i32_pq_new(int capacity);
void rqh2_i32_pq_free(Rqh2I32Pq *pq);
bool rqh2_i32_pq_push(Rqh2I32Pq *pq, int32_t item);
int32_t rqh2_i32_pq_pop(Rqh2I32Pq *pq);
int32_t rqh2_i32_pq_top(Rqh2I32Pq *pq);
int rqh2_i32_pq_size(const Rqh2I32Pq *pq);
bool rqh2_i32_pq_empty(const Rqh2I32Pq *pq);

Rqh2I64Pq *rqh2_i64_pq_new(int capacity);
void rqh2_i64_pq_free(Rqh2I64Pq *pq);
bool rqh2_i64_pq_push(Rqh2I64Pq *pq, int64_t item);
int64_t rqh2_i64_pq_pop(Rqh2I64Pq *pq);
int64_t rqh2_i64_pq_top(Rqh2I64Pq *pq);
int rqh2_i64_pq_size(const Rqh2I64Pq *pq);
bool rqh2_i64_pq_empty(const Rqh2I64Pq *pq);

Rqh2U32Pq *rqh2_u32_pq_new(int capacity);
void rqh2_u32_pq_free(Rqh2U32Pq *pq);
bool rqh2_u32_pq_push(Rqh2U32Pq *pq, uint32_t item);
uint32_t rqh2_u32_pq_pop(Rqh2U32Pq *pq);
uint32_t rqh2_u32_pq_top(Rqh2U32Pq *pq);
int rqh2_u32_pq_size(const Rqh2U32Pq *pq);
bool rqh2_u32_pq_empty(const Rqh2U32Pq *pq);

Rqh2U64Pq *rqh2_u64_pq_new(int capacity);
void rqh2_u64_pq_free(Rqh2U64Pq *pq);
bool rqh2_u64_pq_push(Rqh2U64Pq *pq, uint64_t item);
uint64_t rqh2_u64_pq_pop(Rqh2U64Pq *pq);
uint64_t rqh2_u64_pq_top(Rqh2U64Pq *pq);
int rqh2_u64_pq_size(const Rqh2U64Pq *pq);
bool rqh2_u64_pq_empty(const Rqh2U64Pq *pq);

} // extern "C"
