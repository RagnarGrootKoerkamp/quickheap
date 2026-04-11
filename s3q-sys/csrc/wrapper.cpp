#include "wrapper.hpp"

#include <s3q/s3q.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
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

// ---- Counting variant for i64 ----

thread_local uint64_t s3q_push_cmp = 0;
thread_local uint64_t s3q_pop_cmp = 0;
thread_local uint64_t* s3q_cmp_target = nullptr;

struct CntI64 {
    int64_t v;
    CntI64 operator-()       const noexcept { return {-v}; }
    CntI64 operator+(int64_t n) const noexcept { return {v + n}; }
    CntI64 operator-(int64_t n) const noexcept { return {v - n}; }
    bool operator<(CntI64 o) const {if (s3q_cmp_target) ++(*s3q_cmp_target); return v < o.v; }
    bool operator>(CntI64 o) const {if (s3q_cmp_target) ++(*s3q_cmp_target); return v > o.v; }
    bool operator<=(CntI64 o) const {if (s3q_cmp_target) ++(*s3q_cmp_target); return v <= o.v; }
    bool operator>=(CntI64 o) const {if (s3q_cmp_target) ++(*s3q_cmp_target); return v >= o.v; }
    bool operator==(CntI64 o) const { return v == o.v; }
    bool operator!=(CntI64 o) const { return v != o.v; }
};

namespace std {
template <> struct numeric_limits<CntI64> {
    static constexpr bool is_specialized = true;
    static constexpr CntI64 min()      noexcept { return {INT64_MIN}; }
    static constexpr CntI64 max()      noexcept { return {INT64_MAX}; }
    static constexpr CntI64 lowest()   noexcept { return {INT64_MIN}; }
    static constexpr CntI64 infinity() noexcept { return {INT64_MAX}; }
    static constexpr bool has_infinity = false;
};
} // namespace std

struct I64CountingCfg {
    using BucketIdx = std::ptrdiff_t;
    using Item = CntI64;
    static constexpr std::ptrdiff_t kBufBaseSize = (1l << 15) / sizeof(CntI64);
    static constexpr int kLogMaxDegree = 6;

    // Perfect-forwarding GetKey so Key = CntI64 (not const CntI64).
    struct GetKey {
        template <class T>
        constexpr decltype(auto) operator()(T&& item) const noexcept {
            return std::forward<T>(item);
        }
    };
};

using I64CountingPq = s3q::PriorityQueue<I64CountingCfg>;

struct S3qI64CountingOpaque {
    I64CountingPq inner;
};

extern "C" {

S3qI64CountingPq* s3q_i64_counting_pq_new() {
    return reinterpret_cast<S3qI64CountingPq*>(new (std::nothrow) S3qI64CountingOpaque());
}

void s3q_i64_counting_pq_free(S3qI64CountingPq* pq) {
    delete reinterpret_cast<S3qI64CountingOpaque*>(pq);
}

void s3q_i64_counting_pq_push(S3qI64CountingPq* pq, int64_t item) {
    s3q_cmp_target = &s3q_push_cmp;
    // fprintf(stderr, "push: cmp_target=%p, &push_cmp=%p, push_cmp_val=%lu\n",
    //         (void*)s3q_cmp_target, (void*)&s3q_push_cmp, s3q_push_cmp);
    reinterpret_cast<S3qI64CountingOpaque*>(pq)->inner.push({item});
    // fprintf(stderr, "push done: push_cmp=%lu, *cmp_target=%lu\n",
    //         s3q_push_cmp, s3q_cmp_target ? *s3q_cmp_target : 999ULL);
    s3q_cmp_target = nullptr;
}

int64_t s3q_i64_counting_pq_pop(S3qI64CountingPq* pq) {
    s3q_cmp_target = &s3q_pop_cmp;
    CntI64 result = reinterpret_cast<S3qI64CountingOpaque*>(pq)->inner.pop();
    s3q_cmp_target = nullptr;
    return result.v;
}

bool s3q_i64_counting_pq_empty(const S3qI64CountingPq* pq) {
    return reinterpret_cast<const S3qI64CountingOpaque*>(pq)->inner.empty();
}

void s3q_i64_counting_pq_reset_comparisons() {
    s3q_push_cmp = 0;
    s3q_pop_cmp = 0;
}

uint64_t s3q_i64_counting_pq_push_comparisons() { return s3q_push_cmp; }
uint64_t s3q_i64_counting_pq_pop_comparisons()  { return s3q_pop_cmp; }

} // extern "C" (counting)
