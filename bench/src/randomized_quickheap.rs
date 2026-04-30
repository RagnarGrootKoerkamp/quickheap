use crate::Heap;
use crate::impls::NoHeap;
use crate::workloads::CountingHeapT;
use randomized_quickheap_sys::{
    Rqh2I32Pq, Rqh2I64CountingPq, Rqh2I64Pq, Rqh2U32Pq, Rqh2U64Pq, rqh2_i32_pq_empty,
    rqh2_i32_pq_free, rqh2_i32_pq_new, rqh2_i32_pq_pop, rqh2_i32_pq_push,
    rqh2_i64_counting_pq_empty, rqh2_i64_counting_pq_free, rqh2_i64_counting_pq_new,
    rqh2_i64_counting_pq_pop, rqh2_i64_counting_pq_pop_comparisons,
    rqh2_i64_counting_pq_push, rqh2_i64_counting_pq_push_comparisons,
    rqh2_i64_counting_pq_reset_comparisons, rqh2_i64_pq_empty, rqh2_i64_pq_free,
    rqh2_i64_pq_new, rqh2_i64_pq_pop, rqh2_i64_pq_push, rqh2_u32_pq_empty, rqh2_u32_pq_free,
    rqh2_u32_pq_new, rqh2_u32_pq_pop, rqh2_u32_pq_push, rqh2_u64_pq_empty, rqh2_u64_pq_free,
    rqh2_u64_pq_new, rqh2_u64_pq_pop, rqh2_u64_pq_push,
};

/// Default capacity for the randomized quickheap. Must cover the maximum n used by any
/// benchmark workload (bench goes up to 2^25 = ~33M elements in HeapSort).
const DEFAULT_CAPACITY: i32 = 1 << 25;

macro_rules! impl_rqh2_heap {
    ($heap:ident, $t:ty, $pq:ty, $new:ident, $free:ident, $push:ident, $pop:ident, $empty:ident, $counting:ty) => {
        pub struct $heap(*mut $pq);

        impl Drop for $heap {
            fn drop(&mut self) {
                unsafe { $free(self.0) }
            }
        }

        impl Heap<$t> for $heap {
            type CountedHeap = $counting;

            #[inline(always)]
            fn default() -> Self {
                let pq = unsafe { $new(DEFAULT_CAPACITY) };
                assert!(
                    !pq.is_null(),
                    concat!(stringify!($new), ": allocation failed")
                );
                $heap(pq)
            }

            #[inline(always)]
            fn push(&mut self, t: $t) {
                let ok = unsafe { $push(self.0, t) };
                assert!(ok, "randomized quickheap is full");
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

impl_rqh2_heap!(
    RandQH2HeapI32,
    i32,
    Rqh2I32Pq,
    rqh2_i32_pq_new,
    rqh2_i32_pq_free,
    rqh2_i32_pq_push,
    rqh2_i32_pq_pop,
    rqh2_i32_pq_empty,
    NoHeap
);
impl_rqh2_heap!(
    RandQH2HeapI64,
    i64,
    Rqh2I64Pq,
    rqh2_i64_pq_new,
    rqh2_i64_pq_free,
    rqh2_i64_pq_push,
    rqh2_i64_pq_pop,
    rqh2_i64_pq_empty,
    RandQH2HeapI64Counting
);
impl_rqh2_heap!(
    RandQH2HeapU32,
    u32,
    Rqh2U32Pq,
    rqh2_u32_pq_new,
    rqh2_u32_pq_free,
    rqh2_u32_pq_push,
    rqh2_u32_pq_pop,
    rqh2_u32_pq_empty,
    NoHeap
);
impl_rqh2_heap!(
    RandQH2HeapU64,
    u64,
    Rqh2U64Pq,
    rqh2_u64_pq_new,
    rqh2_u64_pq_free,
    rqh2_u64_pq_push,
    rqh2_u64_pq_pop,
    rqh2_u64_pq_empty,
    NoHeap
);

pub struct RandQH2HeapI64Counting(*mut Rqh2I64CountingPq);

impl Drop for RandQH2HeapI64Counting {
    fn drop(&mut self) {
        unsafe { rqh2_i64_counting_pq_free(self.0) }
    }
}

impl Heap<i64> for RandQH2HeapI64Counting {
    type CountedHeap = NoHeap;

    fn default() -> Self {
        let pq = unsafe { rqh2_i64_counting_pq_new(DEFAULT_CAPACITY) };
        assert!(!pq.is_null(), "rqh2_i64_counting_pq_new: allocation failed");
        RandQH2HeapI64Counting(pq)
    }

    #[inline(always)]
    fn push(&mut self, t: i64) {
        let ok = unsafe { rqh2_i64_counting_pq_push(self.0, t) };
        assert!(ok, "randomized quickheap is full");
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<i64> {
        unsafe {
            if rqh2_i64_counting_pq_empty(self.0) {
                None
            } else {
                Some(rqh2_i64_counting_pq_pop(self.0))
            }
        }
    }
}

impl CountingHeapT<i64> for RandQH2HeapI64Counting {
    fn reset_comparisons() {
        unsafe { rqh2_i64_counting_pq_reset_comparisons() }
    }
    fn get_comparisons() -> (u64, u64) {
        unsafe {
            (
                rqh2_i64_counting_pq_push_comparisons(),
                rqh2_i64_counting_pq_pop_comparisons(),
            )
        }
    }
}
