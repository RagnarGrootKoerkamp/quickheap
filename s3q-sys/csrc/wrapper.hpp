#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct S3qI32Pq S3qI32Pq;

S3qI32Pq *s3q_i32_pq_new(void);
void s3q_i32_pq_free(S3qI32Pq *pq);
void s3q_i32_pq_push(S3qI32Pq *pq, int32_t item);
int32_t s3q_i32_pq_pop(S3qI32Pq *pq);
int32_t s3q_i32_pq_top(const S3qI32Pq *pq);
size_t s3q_i32_pq_size(const S3qI32Pq *pq);
bool s3q_i32_pq_empty(const S3qI32Pq *pq);

typedef struct S3qI64Pq S3qI64Pq;

S3qI64Pq *s3q_i64_pq_new(void);
void s3q_i64_pq_free(S3qI64Pq *pq);
void s3q_i64_pq_push(S3qI64Pq *pq, int64_t item);
int64_t s3q_i64_pq_pop(S3qI64Pq *pq);
int64_t s3q_i64_pq_top(const S3qI64Pq *pq);
size_t s3q_i64_pq_size(const S3qI64Pq *pq);
bool s3q_i64_pq_empty(const S3qI64Pq *pq);

typedef struct S3qU32Pq S3qU32Pq;

S3qU32Pq *s3q_u32_pq_new(void);
void s3q_u32_pq_free(S3qU32Pq *pq);
void s3q_u32_pq_push(S3qU32Pq *pq, uint32_t item);
uint32_t s3q_u32_pq_pop(S3qU32Pq *pq);
uint32_t s3q_u32_pq_top(const S3qU32Pq *pq);
size_t s3q_u32_pq_size(const S3qU32Pq *pq);
bool s3q_u32_pq_empty(const S3qU32Pq *pq);

typedef struct S3qU64Pq S3qU64Pq;

S3qU64Pq *s3q_u64_pq_new(void);
void s3q_u64_pq_free(S3qU64Pq *pq);
void s3q_u64_pq_push(S3qU64Pq *pq, uint64_t item);
uint64_t s3q_u64_pq_pop(S3qU64Pq *pq);
uint64_t s3q_u64_pq_top(const S3qU64Pq *pq);
size_t s3q_u64_pq_size(const S3qU64Pq *pq);
bool s3q_u64_pq_empty(const S3qU64Pq *pq);

#ifdef __cplusplus
}
#endif
