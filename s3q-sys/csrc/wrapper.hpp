#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct S3qU32Pq S3qU32Pq;

S3qU32Pq *s3q_u32_pq_new(void);
void s3q_u32_pq_free(S3qU32Pq *pq);
void s3q_u32_pq_push(S3qU32Pq *pq, uint32_t item);
uint32_t s3q_u32_pq_pop(S3qU32Pq *pq);
uint32_t s3q_u32_pq_top(const S3qU32Pq *pq);
size_t s3q_u32_pq_size(const S3qU32Pq *pq);
bool s3q_u32_pq_empty(const S3qU32Pq *pq);

#ifdef __cplusplus
}
#endif
