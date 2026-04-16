#include "wrapper.hpp"

#include <boost/heap/d_ary_heap.hpp>
#include <boost/heap/fibonacci_heap.hpp>
#include <boost/heap/pairing_heap.hpp>
#include <boost/heap/binomial_heap.hpp>
#include <boost/heap/skew_heap.hpp>

#include <cstdint>
#include <functional>
#include <new>

/* ── implementation macro ──────────────────────────────────────────── */

#define IMPL_BOOST_HEAP(PREFIX, HEAP_TYPE, CTYPE)                              \
    PREFIX *PREFIX##_new() {                                                    \
        return reinterpret_cast<PREFIX *>(new (std::nothrow) HEAP_TYPE());      \
    }                                                                           \
    void PREFIX##_free(PREFIX *pq) {                                           \
        delete reinterpret_cast<HEAP_TYPE *>(pq);                              \
    }                                                                           \
    void PREFIX##_push(PREFIX *pq, CTYPE item) {                               \
        reinterpret_cast<HEAP_TYPE *>(pq)->push(item);                         \
    }                                                                           \
    CTYPE PREFIX##_pop(PREFIX *pq) {                                           \
        auto *h = reinterpret_cast<HEAP_TYPE *>(pq);                           \
        CTYPE val = h->top();                                                  \
        h->pop();                                                              \
        return val;                                                            \
    }                                                                           \
    CTYPE PREFIX##_top(const PREFIX *pq) {                                     \
        return reinterpret_cast<const HEAP_TYPE *>(pq)->top();                 \
    }                                                                           \
    size_t PREFIX##_size(const PREFIX *pq) {                                   \
        return reinterpret_cast<const HEAP_TYPE *>(pq)->size();                \
    }                                                                           \
    bool PREFIX##_empty(const PREFIX *pq) {                                    \
        return reinterpret_cast<const HEAP_TYPE *>(pq)->empty();               \
    }

/* ── type aliases ──────────────────────────────────────────────────── */

template <typename T>
using Dary4 = boost::heap::d_ary_heap<T, boost::heap::arity<4>,
                                       boost::heap::compare<std::greater<T>>>;

template <typename T>
using Fib = boost::heap::fibonacci_heap<T, boost::heap::compare<std::greater<T>>>;

template <typename T>
using Pairing = boost::heap::pairing_heap<T, boost::heap::compare<std::greater<T>>>;

template <typename T>
using Binomial = boost::heap::binomial_heap<T, boost::heap::compare<std::greater<T>>>;

template <typename T>
using Skew = boost::heap::skew_heap<T, boost::heap::compare<std::greater<T>>>;

/* ── instantiations ────────────────────────────────────────────────── */

extern "C" {

IMPL_BOOST_HEAP(boost_dary4_i32_pq, Dary4<int32_t>, int32_t)
IMPL_BOOST_HEAP(boost_dary4_i64_pq, Dary4<int64_t>, int64_t)
IMPL_BOOST_HEAP(boost_dary4_u32_pq, Dary4<uint32_t>, uint32_t)
IMPL_BOOST_HEAP(boost_dary4_u64_pq, Dary4<uint64_t>, uint64_t)

IMPL_BOOST_HEAP(boost_fib_i32_pq, Fib<int32_t>, int32_t)
IMPL_BOOST_HEAP(boost_fib_i64_pq, Fib<int64_t>, int64_t)
IMPL_BOOST_HEAP(boost_fib_u32_pq, Fib<uint32_t>, uint32_t)
IMPL_BOOST_HEAP(boost_fib_u64_pq, Fib<uint64_t>, uint64_t)

IMPL_BOOST_HEAP(boost_pairing_i32_pq, Pairing<int32_t>, int32_t)
IMPL_BOOST_HEAP(boost_pairing_i64_pq, Pairing<int64_t>, int64_t)
IMPL_BOOST_HEAP(boost_pairing_u32_pq, Pairing<uint32_t>, uint32_t)
IMPL_BOOST_HEAP(boost_pairing_u64_pq, Pairing<uint64_t>, uint64_t)

IMPL_BOOST_HEAP(boost_binomial_i32_pq, Binomial<int32_t>, int32_t)
IMPL_BOOST_HEAP(boost_binomial_i64_pq, Binomial<int64_t>, int64_t)
IMPL_BOOST_HEAP(boost_binomial_u32_pq, Binomial<uint32_t>, uint32_t)
IMPL_BOOST_HEAP(boost_binomial_u64_pq, Binomial<uint64_t>, uint64_t)

IMPL_BOOST_HEAP(boost_skew_i32_pq, Skew<int32_t>, int32_t)
IMPL_BOOST_HEAP(boost_skew_i64_pq, Skew<int64_t>, int64_t)
IMPL_BOOST_HEAP(boost_skew_u32_pq, Skew<uint32_t>, uint32_t)
IMPL_BOOST_HEAP(boost_skew_u64_pq, Skew<uint64_t>, uint64_t)

} // extern "C"
