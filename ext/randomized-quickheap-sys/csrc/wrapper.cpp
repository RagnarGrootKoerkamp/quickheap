#include "wrapper.hpp"

#include "MinRandQH2.hpp"

#include <cstdint>
#include <new>

using I32Pq = MinRandQH2<int32_t>;
using I64Pq = MinRandQH2<int64_t>;
using U32Pq = MinRandQH2<uint32_t>;
using U64Pq = MinRandQH2<uint64_t>;

extern "C" {

Rqh2I32Pq *rqh2_i32_pq_new(int capacity) {
    return reinterpret_cast<Rqh2I32Pq *>(new (std::nothrow) I32Pq(capacity));
}
void rqh2_i32_pq_free(Rqh2I32Pq *pq) {
    delete reinterpret_cast<I32Pq *>(pq);
}
bool rqh2_i32_pq_push(Rqh2I32Pq *pq, int32_t item) {
    return reinterpret_cast<I32Pq *>(pq)->insert(item);
}
int32_t rqh2_i32_pq_pop(Rqh2I32Pq *pq) {
    return reinterpret_cast<I32Pq *>(pq)->extractMin();
}
int32_t rqh2_i32_pq_top(Rqh2I32Pq *pq) {
    return reinterpret_cast<I32Pq *>(pq)->findMin();
}
int rqh2_i32_pq_size(const Rqh2I32Pq *pq) {
    return reinterpret_cast<const I32Pq *>(pq)->size();
}
bool rqh2_i32_pq_empty(const Rqh2I32Pq *pq) {
    return reinterpret_cast<const I32Pq *>(pq)->isEmpty();
}

Rqh2I64Pq *rqh2_i64_pq_new(int capacity) {
    return reinterpret_cast<Rqh2I64Pq *>(new (std::nothrow) I64Pq(capacity));
}
void rqh2_i64_pq_free(Rqh2I64Pq *pq) {
    delete reinterpret_cast<I64Pq *>(pq);
}
bool rqh2_i64_pq_push(Rqh2I64Pq *pq, int64_t item) {
    return reinterpret_cast<I64Pq *>(pq)->insert(item);
}
int64_t rqh2_i64_pq_pop(Rqh2I64Pq *pq) {
    return reinterpret_cast<I64Pq *>(pq)->extractMin();
}
int64_t rqh2_i64_pq_top(Rqh2I64Pq *pq) {
    return reinterpret_cast<I64Pq *>(pq)->findMin();
}
int rqh2_i64_pq_size(const Rqh2I64Pq *pq) {
    return reinterpret_cast<const I64Pq *>(pq)->size();
}
bool rqh2_i64_pq_empty(const Rqh2I64Pq *pq) {
    return reinterpret_cast<const I64Pq *>(pq)->isEmpty();
}

Rqh2U32Pq *rqh2_u32_pq_new(int capacity) {
    return reinterpret_cast<Rqh2U32Pq *>(new (std::nothrow) U32Pq(capacity));
}
void rqh2_u32_pq_free(Rqh2U32Pq *pq) {
    delete reinterpret_cast<U32Pq *>(pq);
}
bool rqh2_u32_pq_push(Rqh2U32Pq *pq, uint32_t item) {
    return reinterpret_cast<U32Pq *>(pq)->insert(item);
}
uint32_t rqh2_u32_pq_pop(Rqh2U32Pq *pq) {
    return reinterpret_cast<U32Pq *>(pq)->extractMin();
}
uint32_t rqh2_u32_pq_top(Rqh2U32Pq *pq) {
    return reinterpret_cast<U32Pq *>(pq)->findMin();
}
int rqh2_u32_pq_size(const Rqh2U32Pq *pq) {
    return reinterpret_cast<const U32Pq *>(pq)->size();
}
bool rqh2_u32_pq_empty(const Rqh2U32Pq *pq) {
    return reinterpret_cast<const U32Pq *>(pq)->isEmpty();
}

Rqh2U64Pq *rqh2_u64_pq_new(int capacity) {
    return reinterpret_cast<Rqh2U64Pq *>(new (std::nothrow) U64Pq(capacity));
}
void rqh2_u64_pq_free(Rqh2U64Pq *pq) {
    delete reinterpret_cast<U64Pq *>(pq);
}
bool rqh2_u64_pq_push(Rqh2U64Pq *pq, uint64_t item) {
    return reinterpret_cast<U64Pq *>(pq)->insert(item);
}
uint64_t rqh2_u64_pq_pop(Rqh2U64Pq *pq) {
    return reinterpret_cast<U64Pq *>(pq)->extractMin();
}
uint64_t rqh2_u64_pq_top(Rqh2U64Pq *pq) {
    return reinterpret_cast<U64Pq *>(pq)->findMin();
}
int rqh2_u64_pq_size(const Rqh2U64Pq *pq) {
    return reinterpret_cast<const U64Pq *>(pq)->size();
}
bool rqh2_u64_pq_empty(const Rqh2U64Pq *pq) {
    return reinterpret_cast<const U64Pq *>(pq)->isEmpty();
}

} // extern "C"
