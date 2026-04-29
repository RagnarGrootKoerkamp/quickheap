#include "wrapper.hpp"

#include <climits>
#include <cstdint>
#include <new>

// Pull in the full KNHeap template.
// knheap.C uses `#include "knheap.h"` (caught by include guard after
// our include below) and `#include "multiMergeUnrolled.C"` (found
// relative to knheap.C's own directory in third_party/SequenceHeap/spq/).
//
// util.h (pulled in by knheap.h) defines Assert macros that use unqualified
// `cout`/`endl`. We disable all assertions before including the template
// implementations to avoid the compile error.
#define DEBUGLEVEL 0 // suppress Debug1..Debug5 and their Assert variants
#include "spq/knheap.h"
// Assert(c) used directly in dead stub functions — redefine as no-op
#undef Assert
#define Assert(c) ((void)0)
#include "spq/knheap.C"
#undef Assert

// KNHeap requires a concrete value type; use a zero-size dummy to model void.
struct NoValue {};
using I32Heap = KNHeap<int32_t, NoValue>;
using I64Heap = KNHeap<int64_t, NoValue>;
using U32Heap = KNHeap<uint32_t, NoValue>;
using U64Heap = KNHeap<uint64_t, NoValue>;

extern "C" {

SeqHeapI32 *seq_heap_i32_new() {
    return reinterpret_cast<SeqHeapI32 *>(
        new (std::nothrow) I32Heap(INT32_MAX, INT32_MIN));
}

void seq_heap_i32_free(SeqHeapI32 *pq) {
    delete reinterpret_cast<I32Heap *>(pq);
}

void seq_heap_i32_push(SeqHeapI32 *pq, int32_t key) {
    reinterpret_cast<I32Heap *>(pq)->insert(key, NoValue{});
}

int32_t seq_heap_i32_pop(SeqHeapI32 *pq) {
    int32_t key; NoValue value;
    reinterpret_cast<I32Heap *>(pq)->deleteMin(&key, &value);
    return key;
}

int32_t seq_heap_i32_top(SeqHeapI32 *pq) {
    int32_t key; NoValue value;
    reinterpret_cast<I32Heap *>(pq)->getMin(&key, &value);
    return key;
}

int seq_heap_i32_size(const SeqHeapI32 *pq) {
    return reinterpret_cast<const I32Heap *>(pq)->getSize();
}

bool seq_heap_i32_empty(const SeqHeapI32 *pq) {
    return reinterpret_cast<const I32Heap *>(pq)->getSize() == 0;
}

SeqHeapI64 *seq_heap_i64_new() {
    return reinterpret_cast<SeqHeapI64 *>(
        new (std::nothrow) I64Heap(INT64_MAX, INT64_MIN));
}

void seq_heap_i64_free(SeqHeapI64 *pq) {
    delete reinterpret_cast<I64Heap *>(pq);
}

void seq_heap_i64_push(SeqHeapI64 *pq, int64_t key) {
    reinterpret_cast<I64Heap *>(pq)->insert(key, NoValue{});
}

int64_t seq_heap_i64_pop(SeqHeapI64 *pq) {
    int64_t key; NoValue value;
    reinterpret_cast<I64Heap *>(pq)->deleteMin(&key, &value);
    return key;
}

int64_t seq_heap_i64_top(SeqHeapI64 *pq) {
    int64_t key; NoValue value;
    reinterpret_cast<I64Heap *>(pq)->getMin(&key, &value);
    return key;
}

int seq_heap_i64_size(const SeqHeapI64 *pq) {
    return reinterpret_cast<const I64Heap *>(pq)->getSize();
}

bool seq_heap_i64_empty(const SeqHeapI64 *pq) {
    return reinterpret_cast<const I64Heap *>(pq)->getSize() == 0;
}

SeqHeapU32 *seq_heap_u32_new() {
    return reinterpret_cast<SeqHeapU32 *>(
        new (std::nothrow) U32Heap(UINT32_MAX, 0));
}

void seq_heap_u32_free(SeqHeapU32 *pq) {
    delete reinterpret_cast<U32Heap *>(pq);
}

void seq_heap_u32_push(SeqHeapU32 *pq, uint32_t key) {
    reinterpret_cast<U32Heap *>(pq)->insert(key, NoValue{});
}

uint32_t seq_heap_u32_pop(SeqHeapU32 *pq) {
    uint32_t key; NoValue value;
    reinterpret_cast<U32Heap *>(pq)->deleteMin(&key, &value);
    return key;
}

uint32_t seq_heap_u32_top(SeqHeapU32 *pq) {
    uint32_t key; NoValue value;
    reinterpret_cast<U32Heap *>(pq)->getMin(&key, &value);
    return key;
}

int seq_heap_u32_size(const SeqHeapU32 *pq) {
    return reinterpret_cast<const U32Heap *>(pq)->getSize();
}

bool seq_heap_u32_empty(const SeqHeapU32 *pq) {
    return reinterpret_cast<const U32Heap *>(pq)->getSize() == 0;
}

SeqHeapU64 *seq_heap_u64_new() {
    return reinterpret_cast<SeqHeapU64 *>(
        new (std::nothrow) U64Heap(UINT64_MAX, 0));
}

void seq_heap_u64_free(SeqHeapU64 *pq) {
    delete reinterpret_cast<U64Heap *>(pq);
}

void seq_heap_u64_push(SeqHeapU64 *pq, uint64_t key) {
    reinterpret_cast<U64Heap *>(pq)->insert(key, NoValue{});
}

uint64_t seq_heap_u64_pop(SeqHeapU64 *pq) {
    uint64_t key; NoValue value;
    reinterpret_cast<U64Heap *>(pq)->deleteMin(&key, &value);
    return key;
}

uint64_t seq_heap_u64_top(SeqHeapU64 *pq) {
    uint64_t key; NoValue value;
    reinterpret_cast<U64Heap *>(pq)->getMin(&key, &value);
    return key;
}

int seq_heap_u64_size(const SeqHeapU64 *pq) {
    return reinterpret_cast<const U64Heap *>(pq)->getSize();
}

bool seq_heap_u64_empty(const SeqHeapU64 *pq) {
    return reinterpret_cast<const U64Heap *>(pq)->getSize() == 0;
}

} // extern "C"

// ---- Counting variant for i64 ----

thread_local uint64_t seq_push_cmp = 0;
thread_local uint64_t seq_pop_cmp = 0;
thread_local uint64_t* seq_cmp_target = nullptr;

struct CntI64 {
    int64_t v;
    CntI64 operator-() const noexcept { return {-v}; }
    bool operator<(CntI64 o) const { if (seq_cmp_target) ++(*seq_cmp_target); return v < o.v; }
    bool operator>(CntI64 o) const { if (seq_cmp_target) ++(*seq_cmp_target); return v > o.v; }
    bool operator<=(CntI64 o) const { if (seq_cmp_target) ++(*seq_cmp_target); return v <= o.v; }
    bool operator>=(CntI64 o) const { if (seq_cmp_target) ++(*seq_cmp_target); return v >= o.v; }
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

using I64CntHeap = KNHeap<CntI64, NoValue>;

struct SeqI64CountingOpaque {
    I64CntHeap inner;
    SeqI64CountingOpaque() : inner({INT64_MAX}, {INT64_MIN}) {}
};

extern "C" {

SeqHeapI64Counting* seq_heap_i64_counting_new() {
    return reinterpret_cast<SeqHeapI64Counting*>(new (std::nothrow) SeqI64CountingOpaque());
}
void seq_heap_i64_counting_free(SeqHeapI64Counting* pq) {
    delete reinterpret_cast<SeqI64CountingOpaque*>(pq);
}
void seq_heap_i64_counting_push(SeqHeapI64Counting* pq, int64_t key) {
    seq_cmp_target = &seq_push_cmp;
    reinterpret_cast<SeqI64CountingOpaque*>(pq)->inner.insert({key}, NoValue{});
    seq_cmp_target = nullptr;
}
int64_t seq_heap_i64_counting_pop(SeqHeapI64Counting* pq) {
    seq_cmp_target = &seq_pop_cmp;
    CntI64 key; NoValue value;
    reinterpret_cast<SeqI64CountingOpaque*>(pq)->inner.deleteMin(&key, &value);
    seq_cmp_target = nullptr;
    return key.v;
}
bool seq_heap_i64_counting_empty(const SeqHeapI64Counting* pq) {
    return reinterpret_cast<const SeqI64CountingOpaque*>(pq)->inner.getSize() == 0;
}
void seq_heap_i64_counting_reset_comparisons() { seq_push_cmp = 0; seq_pop_cmp = 0; }
uint64_t seq_heap_i64_counting_push_comparisons() { return seq_push_cmp; }
uint64_t seq_heap_i64_counting_pop_comparisons()  { return seq_pop_cmp; }

} // extern "C" (counting)
