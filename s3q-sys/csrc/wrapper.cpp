#include "wrapper.hpp"

#include <s3q/s3q.hpp>

#include <cstddef>
#include <cstdint>
#include <new>

struct I32Cfg : s3q::DefaultCfg {
    using Item = int32_t;
};

using I32Pq = s3q::PriorityQueue<I32Cfg>;

struct I64Cfg : s3q::DefaultCfg {
    using Item = int64_t;
};

using I64Pq = s3q::PriorityQueue<I64Cfg>;

struct U32Cfg : s3q::DefaultCfg {
    using Item = uint32_t;
};

using U32Pq = s3q::PriorityQueue<U32Cfg>;

struct U64Cfg : s3q::DefaultCfg {
    using Item = uint64_t;
};

using U64Pq = s3q::PriorityQueue<U64Cfg>;

extern "C" {

S3qI32Pq *s3q_i32_pq_new() {
    return reinterpret_cast<S3qI32Pq *>(new (std::nothrow) I32Pq());
}

void s3q_i32_pq_free(S3qI32Pq *pq) {
    delete reinterpret_cast<I32Pq *>(pq);
}

void s3q_i32_pq_push(S3qI32Pq *pq, int32_t item) {
    reinterpret_cast<I32Pq *>(pq)->push(item);
}

int32_t s3q_i32_pq_pop(S3qI32Pq *pq) {
    return reinterpret_cast<I32Pq *>(pq)->pop();
}

int32_t s3q_i32_pq_top(const S3qI32Pq *pq) {
    return reinterpret_cast<const I32Pq *>(pq)->top();
}

size_t s3q_i32_pq_size(const S3qI32Pq *pq) {
    return reinterpret_cast<const I32Pq *>(pq)->size();
}

bool s3q_i32_pq_empty(const S3qI32Pq *pq) {
    return reinterpret_cast<const I32Pq *>(pq)->empty();
}

S3qI64Pq *s3q_i64_pq_new() {
    return reinterpret_cast<S3qI64Pq *>(new (std::nothrow) I64Pq());
}

void s3q_i64_pq_free(S3qI64Pq *pq) {
    delete reinterpret_cast<I64Pq *>(pq);
}

void s3q_i64_pq_push(S3qI64Pq *pq, int64_t item) {
    reinterpret_cast<I64Pq *>(pq)->push(item);
}

int64_t s3q_i64_pq_pop(S3qI64Pq *pq) {
    return reinterpret_cast<I64Pq *>(pq)->pop();
}

int64_t s3q_i64_pq_top(const S3qI64Pq *pq) {
    return reinterpret_cast<const I64Pq *>(pq)->top();
}

size_t s3q_i64_pq_size(const S3qI64Pq *pq) {
    return reinterpret_cast<const I64Pq *>(pq)->size();
}

bool s3q_i64_pq_empty(const S3qI64Pq *pq) {
    return reinterpret_cast<const I64Pq *>(pq)->empty();
}

S3qU32Pq *s3q_u32_pq_new() {
    return reinterpret_cast<S3qU32Pq *>(new (std::nothrow) U32Pq());
}

void s3q_u32_pq_free(S3qU32Pq *pq) {
    delete reinterpret_cast<U32Pq *>(pq);
}

void s3q_u32_pq_push(S3qU32Pq *pq, uint32_t item) {
    reinterpret_cast<U32Pq *>(pq)->push(item);
}

uint32_t s3q_u32_pq_pop(S3qU32Pq *pq) {
    return reinterpret_cast<U32Pq *>(pq)->pop();
}

uint32_t s3q_u32_pq_top(const S3qU32Pq *pq) {
    return reinterpret_cast<const U32Pq *>(pq)->top();
}

size_t s3q_u32_pq_size(const S3qU32Pq *pq) {
    return reinterpret_cast<const U32Pq *>(pq)->size();
}

bool s3q_u32_pq_empty(const S3qU32Pq *pq) {
    return reinterpret_cast<const U32Pq *>(pq)->empty();
}

S3qU64Pq *s3q_u64_pq_new() {
    return reinterpret_cast<S3qU64Pq *>(new (std::nothrow) U64Pq());
}

void s3q_u64_pq_free(S3qU64Pq *pq) {
    delete reinterpret_cast<U64Pq *>(pq);
}

void s3q_u64_pq_push(S3qU64Pq *pq, uint64_t item) {
    reinterpret_cast<U64Pq *>(pq)->push(item);
}

uint64_t s3q_u64_pq_pop(S3qU64Pq *pq) {
    return reinterpret_cast<U64Pq *>(pq)->pop();
}

uint64_t s3q_u64_pq_top(const S3qU64Pq *pq) {
    return reinterpret_cast<const U64Pq *>(pq)->top();
}

size_t s3q_u64_pq_size(const S3qU64Pq *pq) {
    return reinterpret_cast<const U64Pq *>(pq)->size();
}

bool s3q_u64_pq_empty(const S3qU64Pq *pq) {
    return reinterpret_cast<const U64Pq *>(pq)->empty();
}

} // extern "C"
