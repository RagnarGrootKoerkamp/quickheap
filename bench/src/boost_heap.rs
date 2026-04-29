use crate::Heap;
use crate::impls::NoHeap;

macro_rules! impl_boost_heap {
    ($heap:ident, $t:ty, $pq:ty, $new:ident, $free:ident, $push:ident, $pop:ident, $empty:ident) => {
        pub struct $heap(*mut $pq);

        impl Drop for $heap {
            fn drop(&mut self) {
                unsafe { $free(self.0) }
            }
        }

        impl Heap<$t> for $heap {
            type CountedHeap = NoHeap;

            #[inline(always)]
            fn default() -> Self {
                let pq = unsafe { $new() };
                assert!(
                    !pq.is_null(),
                    concat!(stringify!($new), ": allocation failed")
                );
                $heap(pq)
            }

            #[inline(always)]
            fn push(&mut self, t: $t) {
                unsafe { $push(self.0, t) }
            }

            #[inline(always)]
            fn pop(&mut self) -> Option<$t> {
                unsafe {
                    if $empty(self.0) {
                        None
                    } else {
                        Some($pop(self.0))
                    }
                }
            }
        }
    };
}

// ── d_ary_heap (arity 4) ──────────────────────────────────────────

use boost_heap_sys::{
    BoostDary4I32Pq, BoostDary4I64Pq, BoostDary4U32Pq, BoostDary4U64Pq,
    boost_dary4_i32_pq_empty, boost_dary4_i32_pq_free, boost_dary4_i32_pq_new,
    boost_dary4_i32_pq_pop, boost_dary4_i32_pq_push,
    boost_dary4_i64_pq_empty, boost_dary4_i64_pq_free, boost_dary4_i64_pq_new,
    boost_dary4_i64_pq_pop, boost_dary4_i64_pq_push,
    boost_dary4_u32_pq_empty, boost_dary4_u32_pq_free, boost_dary4_u32_pq_new,
    boost_dary4_u32_pq_pop, boost_dary4_u32_pq_push,
    boost_dary4_u64_pq_empty, boost_dary4_u64_pq_free, boost_dary4_u64_pq_new,
    boost_dary4_u64_pq_pop, boost_dary4_u64_pq_push,
};

impl_boost_heap!(BoostDary4HeapI32, i32, BoostDary4I32Pq,
    boost_dary4_i32_pq_new, boost_dary4_i32_pq_free,
    boost_dary4_i32_pq_push, boost_dary4_i32_pq_pop, boost_dary4_i32_pq_empty);
impl_boost_heap!(BoostDary4HeapI64, i64, BoostDary4I64Pq,
    boost_dary4_i64_pq_new, boost_dary4_i64_pq_free,
    boost_dary4_i64_pq_push, boost_dary4_i64_pq_pop, boost_dary4_i64_pq_empty);
impl_boost_heap!(BoostDary4HeapU32, u32, BoostDary4U32Pq,
    boost_dary4_u32_pq_new, boost_dary4_u32_pq_free,
    boost_dary4_u32_pq_push, boost_dary4_u32_pq_pop, boost_dary4_u32_pq_empty);
impl_boost_heap!(BoostDary4HeapU64, u64, BoostDary4U64Pq,
    boost_dary4_u64_pq_new, boost_dary4_u64_pq_free,
    boost_dary4_u64_pq_push, boost_dary4_u64_pq_pop, boost_dary4_u64_pq_empty);

// ── fibonacci_heap ────────────────────────────────────────────────

use boost_heap_sys::{
    BoostFibI32Pq, BoostFibI64Pq, BoostFibU32Pq, BoostFibU64Pq,
    boost_fib_i32_pq_empty, boost_fib_i32_pq_free, boost_fib_i32_pq_new,
    boost_fib_i32_pq_pop, boost_fib_i32_pq_push,
    boost_fib_i64_pq_empty, boost_fib_i64_pq_free, boost_fib_i64_pq_new,
    boost_fib_i64_pq_pop, boost_fib_i64_pq_push,
    boost_fib_u32_pq_empty, boost_fib_u32_pq_free, boost_fib_u32_pq_new,
    boost_fib_u32_pq_pop, boost_fib_u32_pq_push,
    boost_fib_u64_pq_empty, boost_fib_u64_pq_free, boost_fib_u64_pq_new,
    boost_fib_u64_pq_pop, boost_fib_u64_pq_push,
};

impl_boost_heap!(BoostFibHeapI32, i32, BoostFibI32Pq,
    boost_fib_i32_pq_new, boost_fib_i32_pq_free,
    boost_fib_i32_pq_push, boost_fib_i32_pq_pop, boost_fib_i32_pq_empty);
impl_boost_heap!(BoostFibHeapI64, i64, BoostFibI64Pq,
    boost_fib_i64_pq_new, boost_fib_i64_pq_free,
    boost_fib_i64_pq_push, boost_fib_i64_pq_pop, boost_fib_i64_pq_empty);
impl_boost_heap!(BoostFibHeapU32, u32, BoostFibU32Pq,
    boost_fib_u32_pq_new, boost_fib_u32_pq_free,
    boost_fib_u32_pq_push, boost_fib_u32_pq_pop, boost_fib_u32_pq_empty);
impl_boost_heap!(BoostFibHeapU64, u64, BoostFibU64Pq,
    boost_fib_u64_pq_new, boost_fib_u64_pq_free,
    boost_fib_u64_pq_push, boost_fib_u64_pq_pop, boost_fib_u64_pq_empty);

// ── pairing_heap ──────────────────────────────────────────────────

use boost_heap_sys::{
    BoostPairingI32Pq, BoostPairingI64Pq, BoostPairingU32Pq, BoostPairingU64Pq,
    boost_pairing_i32_pq_empty, boost_pairing_i32_pq_free, boost_pairing_i32_pq_new,
    boost_pairing_i32_pq_pop, boost_pairing_i32_pq_push,
    boost_pairing_i64_pq_empty, boost_pairing_i64_pq_free, boost_pairing_i64_pq_new,
    boost_pairing_i64_pq_pop, boost_pairing_i64_pq_push,
    boost_pairing_u32_pq_empty, boost_pairing_u32_pq_free, boost_pairing_u32_pq_new,
    boost_pairing_u32_pq_pop, boost_pairing_u32_pq_push,
    boost_pairing_u64_pq_empty, boost_pairing_u64_pq_free, boost_pairing_u64_pq_new,
    boost_pairing_u64_pq_pop, boost_pairing_u64_pq_push,
};

impl_boost_heap!(BoostPairingHeapI32, i32, BoostPairingI32Pq,
    boost_pairing_i32_pq_new, boost_pairing_i32_pq_free,
    boost_pairing_i32_pq_push, boost_pairing_i32_pq_pop, boost_pairing_i32_pq_empty);
impl_boost_heap!(BoostPairingHeapI64, i64, BoostPairingI64Pq,
    boost_pairing_i64_pq_new, boost_pairing_i64_pq_free,
    boost_pairing_i64_pq_push, boost_pairing_i64_pq_pop, boost_pairing_i64_pq_empty);
impl_boost_heap!(BoostPairingHeapU32, u32, BoostPairingU32Pq,
    boost_pairing_u32_pq_new, boost_pairing_u32_pq_free,
    boost_pairing_u32_pq_push, boost_pairing_u32_pq_pop, boost_pairing_u32_pq_empty);
impl_boost_heap!(BoostPairingHeapU64, u64, BoostPairingU64Pq,
    boost_pairing_u64_pq_new, boost_pairing_u64_pq_free,
    boost_pairing_u64_pq_push, boost_pairing_u64_pq_pop, boost_pairing_u64_pq_empty);

// ── binomial_heap ─────────────────────────────────────────────────

use boost_heap_sys::{
    BoostBinomialI32Pq, BoostBinomialI64Pq, BoostBinomialU32Pq, BoostBinomialU64Pq,
    boost_binomial_i32_pq_empty, boost_binomial_i32_pq_free, boost_binomial_i32_pq_new,
    boost_binomial_i32_pq_pop, boost_binomial_i32_pq_push,
    boost_binomial_i64_pq_empty, boost_binomial_i64_pq_free, boost_binomial_i64_pq_new,
    boost_binomial_i64_pq_pop, boost_binomial_i64_pq_push,
    boost_binomial_u32_pq_empty, boost_binomial_u32_pq_free, boost_binomial_u32_pq_new,
    boost_binomial_u32_pq_pop, boost_binomial_u32_pq_push,
    boost_binomial_u64_pq_empty, boost_binomial_u64_pq_free, boost_binomial_u64_pq_new,
    boost_binomial_u64_pq_pop, boost_binomial_u64_pq_push,
};

impl_boost_heap!(BoostBinomialHeapI32, i32, BoostBinomialI32Pq,
    boost_binomial_i32_pq_new, boost_binomial_i32_pq_free,
    boost_binomial_i32_pq_push, boost_binomial_i32_pq_pop, boost_binomial_i32_pq_empty);
impl_boost_heap!(BoostBinomialHeapI64, i64, BoostBinomialI64Pq,
    boost_binomial_i64_pq_new, boost_binomial_i64_pq_free,
    boost_binomial_i64_pq_push, boost_binomial_i64_pq_pop, boost_binomial_i64_pq_empty);
impl_boost_heap!(BoostBinomialHeapU32, u32, BoostBinomialU32Pq,
    boost_binomial_u32_pq_new, boost_binomial_u32_pq_free,
    boost_binomial_u32_pq_push, boost_binomial_u32_pq_pop, boost_binomial_u32_pq_empty);
impl_boost_heap!(BoostBinomialHeapU64, u64, BoostBinomialU64Pq,
    boost_binomial_u64_pq_new, boost_binomial_u64_pq_free,
    boost_binomial_u64_pq_push, boost_binomial_u64_pq_pop, boost_binomial_u64_pq_empty);

// ── skew_heap ─────────────────────────────────────────────────────

use boost_heap_sys::{
    BoostSkewI32Pq, BoostSkewI64Pq, BoostSkewU32Pq, BoostSkewU64Pq,
    boost_skew_i32_pq_empty, boost_skew_i32_pq_free, boost_skew_i32_pq_new,
    boost_skew_i32_pq_pop, boost_skew_i32_pq_push,
    boost_skew_i64_pq_empty, boost_skew_i64_pq_free, boost_skew_i64_pq_new,
    boost_skew_i64_pq_pop, boost_skew_i64_pq_push,
    boost_skew_u32_pq_empty, boost_skew_u32_pq_free, boost_skew_u32_pq_new,
    boost_skew_u32_pq_pop, boost_skew_u32_pq_push,
    boost_skew_u64_pq_empty, boost_skew_u64_pq_free, boost_skew_u64_pq_new,
    boost_skew_u64_pq_pop, boost_skew_u64_pq_push,
};

impl_boost_heap!(BoostSkewHeapI32, i32, BoostSkewI32Pq,
    boost_skew_i32_pq_new, boost_skew_i32_pq_free,
    boost_skew_i32_pq_push, boost_skew_i32_pq_pop, boost_skew_i32_pq_empty);
impl_boost_heap!(BoostSkewHeapI64, i64, BoostSkewI64Pq,
    boost_skew_i64_pq_new, boost_skew_i64_pq_free,
    boost_skew_i64_pq_push, boost_skew_i64_pq_pop, boost_skew_i64_pq_empty);
impl_boost_heap!(BoostSkewHeapU32, u32, BoostSkewU32Pq,
    boost_skew_u32_pq_new, boost_skew_u32_pq_free,
    boost_skew_u32_pq_push, boost_skew_u32_pq_pop, boost_skew_u32_pq_empty);
impl_boost_heap!(BoostSkewHeapU64, u64, BoostSkewU64Pq,
    boost_skew_u64_pq_new, boost_skew_u64_pq_free,
    boost_skew_u64_pq_push, boost_skew_u64_pq_pop, boost_skew_u64_pq_empty);
