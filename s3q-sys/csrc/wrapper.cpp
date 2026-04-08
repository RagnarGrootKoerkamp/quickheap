#include "wrapper.hpp"

#include <s3q/s3q.hpp>

#include <cstddef>
#include <cstdint>
#include <new>

struct U32Cfg : s3q::DefaultCfg {
    using Item = uint32_t;
    // Recompute kBufBaseSize for the new Item size (default uses sizeof(DefaultCfg::Item) == 8)
    static constexpr std::ptrdiff_t kBufBaseSize = (1l << 15) / sizeof(Item);
};

using Pq = s3q::PriorityQueue<U32Cfg>;

extern "C" {

S3qU32Pq *s3q_u32_pq_new() {
    return reinterpret_cast<S3qU32Pq *>(new (std::nothrow) Pq());
}

void s3q_u32_pq_free(S3qU32Pq *pq) {
    delete reinterpret_cast<Pq *>(pq);
}

void s3q_u32_pq_push(S3qU32Pq *pq, uint32_t item) {
    reinterpret_cast<Pq *>(pq)->push(item);
}

uint32_t s3q_u32_pq_pop(S3qU32Pq *pq) {
    return reinterpret_cast<Pq *>(pq)->pop();
}

uint32_t s3q_u32_pq_top(const S3qU32Pq *pq) {
    return reinterpret_cast<const Pq *>(pq)->top();
}

size_t s3q_u32_pq_size(const S3qU32Pq *pq) {
    return reinterpret_cast<const Pq *>(pq)->size();
}

bool s3q_u32_pq_empty(const S3qU32Pq *pq) {
    return reinterpret_cast<const Pq *>(pq)->empty();
}

} // extern "C"
