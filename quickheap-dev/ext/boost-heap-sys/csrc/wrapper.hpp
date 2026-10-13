#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── generic declaration macro ─────────────────────────────────────── */

#define DECLARE_BOOST_HEAP(PREFIX, CTYPE)                                      \
    typedef struct PREFIX PREFIX;                                               \
    PREFIX *PREFIX##_new(void);                                                 \
    void PREFIX##_free(PREFIX *pq);                                            \
    void PREFIX##_push(PREFIX *pq, CTYPE item);                                \
    CTYPE PREFIX##_pop(PREFIX *pq);                                            \
    CTYPE PREFIX##_top(const PREFIX *pq);                                      \
    size_t PREFIX##_size(const PREFIX *pq);                                    \
    bool PREFIX##_empty(const PREFIX *pq);

/* ── d_ary_heap (arity 4) ─────────────────────────────────────────── */

DECLARE_BOOST_HEAP(boost_dary4_i32_pq, int32_t)
DECLARE_BOOST_HEAP(boost_dary4_i64_pq, int64_t)
DECLARE_BOOST_HEAP(boost_dary4_u32_pq, uint32_t)
DECLARE_BOOST_HEAP(boost_dary4_u64_pq, uint64_t)

/* ── fibonacci_heap ────────────────────────────────────────────────── */

DECLARE_BOOST_HEAP(boost_fib_i32_pq, int32_t)
DECLARE_BOOST_HEAP(boost_fib_i64_pq, int64_t)
DECLARE_BOOST_HEAP(boost_fib_u32_pq, uint32_t)
DECLARE_BOOST_HEAP(boost_fib_u64_pq, uint64_t)

/* ── pairing_heap ──────────────────────────────────────────────────── */

DECLARE_BOOST_HEAP(boost_pairing_i32_pq, int32_t)
DECLARE_BOOST_HEAP(boost_pairing_i64_pq, int64_t)
DECLARE_BOOST_HEAP(boost_pairing_u32_pq, uint32_t)
DECLARE_BOOST_HEAP(boost_pairing_u64_pq, uint64_t)

/* ── binomial_heap ─────────────────────────────────────────────────── */

DECLARE_BOOST_HEAP(boost_binomial_i32_pq, int32_t)
DECLARE_BOOST_HEAP(boost_binomial_i64_pq, int64_t)
DECLARE_BOOST_HEAP(boost_binomial_u32_pq, uint32_t)
DECLARE_BOOST_HEAP(boost_binomial_u64_pq, uint64_t)

/* ── skew_heap ─────────────────────────────────────────────────────── */

DECLARE_BOOST_HEAP(boost_skew_i32_pq, int32_t)
DECLARE_BOOST_HEAP(boost_skew_i64_pq, int64_t)
DECLARE_BOOST_HEAP(boost_skew_u32_pq, uint32_t)
DECLARE_BOOST_HEAP(boost_skew_u64_pq, uint64_t)

#undef DECLARE_BOOST_HEAP

#ifdef __cplusplus
}
#endif
