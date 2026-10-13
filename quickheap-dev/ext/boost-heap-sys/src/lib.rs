use std::ffi::c_void;

macro_rules! boost_heap_ffi {
    (
        $Pq:ident, $t:ty,
        $new:ident, $free:ident, $push:ident, $pop:ident, $top:ident, $size:ident, $empty:ident
    ) => {
        #[repr(C)]
        pub struct $Pq(c_void);

        unsafe extern "C" {
            pub fn $new() -> *mut $Pq;
            pub fn $free(pq: *mut $Pq);
            pub fn $push(pq: *mut $Pq, item: $t);
            pub fn $pop(pq: *mut $Pq) -> $t;
            pub fn $top(pq: *const $Pq) -> $t;
            pub fn $size(pq: *const $Pq) -> usize;
            pub fn $empty(pq: *const $Pq) -> bool;
        }
    };
}

// ── d_ary_heap (arity 4) ──────────────────────────────────────────

boost_heap_ffi!(
    BoostDary4I32Pq, i32,
    boost_dary4_i32_pq_new, boost_dary4_i32_pq_free,
    boost_dary4_i32_pq_push, boost_dary4_i32_pq_pop, boost_dary4_i32_pq_top,
    boost_dary4_i32_pq_size, boost_dary4_i32_pq_empty
);
boost_heap_ffi!(
    BoostDary4I64Pq, i64,
    boost_dary4_i64_pq_new, boost_dary4_i64_pq_free,
    boost_dary4_i64_pq_push, boost_dary4_i64_pq_pop, boost_dary4_i64_pq_top,
    boost_dary4_i64_pq_size, boost_dary4_i64_pq_empty
);
boost_heap_ffi!(
    BoostDary4U32Pq, u32,
    boost_dary4_u32_pq_new, boost_dary4_u32_pq_free,
    boost_dary4_u32_pq_push, boost_dary4_u32_pq_pop, boost_dary4_u32_pq_top,
    boost_dary4_u32_pq_size, boost_dary4_u32_pq_empty
);
boost_heap_ffi!(
    BoostDary4U64Pq, u64,
    boost_dary4_u64_pq_new, boost_dary4_u64_pq_free,
    boost_dary4_u64_pq_push, boost_dary4_u64_pq_pop, boost_dary4_u64_pq_top,
    boost_dary4_u64_pq_size, boost_dary4_u64_pq_empty
);

// ── fibonacci_heap ────────────────────────────────────────────────

boost_heap_ffi!(
    BoostFibI32Pq, i32,
    boost_fib_i32_pq_new, boost_fib_i32_pq_free,
    boost_fib_i32_pq_push, boost_fib_i32_pq_pop, boost_fib_i32_pq_top,
    boost_fib_i32_pq_size, boost_fib_i32_pq_empty
);
boost_heap_ffi!(
    BoostFibI64Pq, i64,
    boost_fib_i64_pq_new, boost_fib_i64_pq_free,
    boost_fib_i64_pq_push, boost_fib_i64_pq_pop, boost_fib_i64_pq_top,
    boost_fib_i64_pq_size, boost_fib_i64_pq_empty
);
boost_heap_ffi!(
    BoostFibU32Pq, u32,
    boost_fib_u32_pq_new, boost_fib_u32_pq_free,
    boost_fib_u32_pq_push, boost_fib_u32_pq_pop, boost_fib_u32_pq_top,
    boost_fib_u32_pq_size, boost_fib_u32_pq_empty
);
boost_heap_ffi!(
    BoostFibU64Pq, u64,
    boost_fib_u64_pq_new, boost_fib_u64_pq_free,
    boost_fib_u64_pq_push, boost_fib_u64_pq_pop, boost_fib_u64_pq_top,
    boost_fib_u64_pq_size, boost_fib_u64_pq_empty
);

// ── pairing_heap ──────────────────────────────────────────────────

boost_heap_ffi!(
    BoostPairingI32Pq, i32,
    boost_pairing_i32_pq_new, boost_pairing_i32_pq_free,
    boost_pairing_i32_pq_push, boost_pairing_i32_pq_pop, boost_pairing_i32_pq_top,
    boost_pairing_i32_pq_size, boost_pairing_i32_pq_empty
);
boost_heap_ffi!(
    BoostPairingI64Pq, i64,
    boost_pairing_i64_pq_new, boost_pairing_i64_pq_free,
    boost_pairing_i64_pq_push, boost_pairing_i64_pq_pop, boost_pairing_i64_pq_top,
    boost_pairing_i64_pq_size, boost_pairing_i64_pq_empty
);
boost_heap_ffi!(
    BoostPairingU32Pq, u32,
    boost_pairing_u32_pq_new, boost_pairing_u32_pq_free,
    boost_pairing_u32_pq_push, boost_pairing_u32_pq_pop, boost_pairing_u32_pq_top,
    boost_pairing_u32_pq_size, boost_pairing_u32_pq_empty
);
boost_heap_ffi!(
    BoostPairingU64Pq, u64,
    boost_pairing_u64_pq_new, boost_pairing_u64_pq_free,
    boost_pairing_u64_pq_push, boost_pairing_u64_pq_pop, boost_pairing_u64_pq_top,
    boost_pairing_u64_pq_size, boost_pairing_u64_pq_empty
);

// ── binomial_heap ─────────────────────────────────────────────────

boost_heap_ffi!(
    BoostBinomialI32Pq, i32,
    boost_binomial_i32_pq_new, boost_binomial_i32_pq_free,
    boost_binomial_i32_pq_push, boost_binomial_i32_pq_pop, boost_binomial_i32_pq_top,
    boost_binomial_i32_pq_size, boost_binomial_i32_pq_empty
);
boost_heap_ffi!(
    BoostBinomialI64Pq, i64,
    boost_binomial_i64_pq_new, boost_binomial_i64_pq_free,
    boost_binomial_i64_pq_push, boost_binomial_i64_pq_pop, boost_binomial_i64_pq_top,
    boost_binomial_i64_pq_size, boost_binomial_i64_pq_empty
);
boost_heap_ffi!(
    BoostBinomialU32Pq, u32,
    boost_binomial_u32_pq_new, boost_binomial_u32_pq_free,
    boost_binomial_u32_pq_push, boost_binomial_u32_pq_pop, boost_binomial_u32_pq_top,
    boost_binomial_u32_pq_size, boost_binomial_u32_pq_empty
);
boost_heap_ffi!(
    BoostBinomialU64Pq, u64,
    boost_binomial_u64_pq_new, boost_binomial_u64_pq_free,
    boost_binomial_u64_pq_push, boost_binomial_u64_pq_pop, boost_binomial_u64_pq_top,
    boost_binomial_u64_pq_size, boost_binomial_u64_pq_empty
);

// ── skew_heap ─────────────────────────────────────────────────────

boost_heap_ffi!(
    BoostSkewI32Pq, i32,
    boost_skew_i32_pq_new, boost_skew_i32_pq_free,
    boost_skew_i32_pq_push, boost_skew_i32_pq_pop, boost_skew_i32_pq_top,
    boost_skew_i32_pq_size, boost_skew_i32_pq_empty
);
boost_heap_ffi!(
    BoostSkewI64Pq, i64,
    boost_skew_i64_pq_new, boost_skew_i64_pq_free,
    boost_skew_i64_pq_push, boost_skew_i64_pq_pop, boost_skew_i64_pq_top,
    boost_skew_i64_pq_size, boost_skew_i64_pq_empty
);
boost_heap_ffi!(
    BoostSkewU32Pq, u32,
    boost_skew_u32_pq_new, boost_skew_u32_pq_free,
    boost_skew_u32_pq_push, boost_skew_u32_pq_pop, boost_skew_u32_pq_top,
    boost_skew_u32_pq_size, boost_skew_u32_pq_empty
);
boost_heap_ffi!(
    BoostSkewU64Pq, u64,
    boost_skew_u64_pq_new, boost_skew_u64_pq_free,
    boost_skew_u64_pq_push, boost_skew_u64_pq_pop, boost_skew_u64_pq_top,
    boost_skew_u64_pq_size, boost_skew_u64_pq_empty
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dary4_u32_basic() {
        unsafe {
            let pq = boost_dary4_u32_pq_new();
            assert!(!pq.is_null());
            assert!(boost_dary4_u32_pq_empty(pq));

            for &v in &[5u32, 3, 8, 1, 4] {
                boost_dary4_u32_pq_push(pq, v);
            }
            assert_eq!(boost_dary4_u32_pq_size(pq), 5);
            assert_eq!(boost_dary4_u32_pq_top(pq), 1);

            let mut out = Vec::new();
            while !boost_dary4_u32_pq_empty(pq) {
                out.push(boost_dary4_u32_pq_pop(pq));
            }
            assert_eq!(out, vec![1, 3, 4, 5, 8]);

            boost_dary4_u32_pq_free(pq);
        }
    }

    #[test]
    fn fib_i64_basic() {
        unsafe {
            let pq = boost_fib_i64_pq_new();
            assert!(!pq.is_null());

            for &v in &[5i64, -3, 8, -1, 4] {
                boost_fib_i64_pq_push(pq, v);
            }
            assert_eq!(boost_fib_i64_pq_size(pq), 5);

            let mut out = Vec::new();
            while !boost_fib_i64_pq_empty(pq) {
                out.push(boost_fib_i64_pq_pop(pq));
            }
            assert_eq!(out, vec![-3, -1, 4, 5, 8]);

            boost_fib_i64_pq_free(pq);
        }
    }
}
