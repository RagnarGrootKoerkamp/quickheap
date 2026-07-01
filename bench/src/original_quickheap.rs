use crate::Heap;
use crate::impls::NoHeap;
use crate::workloads::CountingHeapT;
use original_quickheap_sys::{
    rqh2_i32_pq_empty, rqh2_i32_pq_free, rqh2_i32_pq_new, rqh2_i32_pq_pop, rqh2_i32_pq_push,
    rqh2_i64_counting_pq_empty, rqh2_i64_counting_pq_free, rqh2_i64_counting_pq_new,
    rqh2_i64_counting_pq_pop, rqh2_i64_counting_pq_pop_comparisons, rqh2_i64_counting_pq_push,
    rqh2_i64_counting_pq_push_comparisons, rqh2_i64_counting_pq_reset_comparisons,
    rqh2_i64_pq_empty, rqh2_i64_pq_free, rqh2_i64_pq_new, rqh2_i64_pq_pop, rqh2_i64_pq_push,
    rqh2_u32_pq_empty, rqh2_u32_pq_free, rqh2_u32_pq_new, rqh2_u32_pq_pop, rqh2_u32_pq_push,
    rqh2_u64_pq_empty, rqh2_u64_pq_free, rqh2_u64_pq_new, rqh2_u64_pq_pop, rqh2_u64_pq_push,
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
    (counting: $heap:ident, $t:ty, $pq:ty, $new:ident, $free:ident, $push:ident, $pop:ident, $empty:ident, $reset:ident, $push_cmp:ident, $pop_cmp:ident) => {
        impl_rqh2_heap!($heap, $t, $pq, $new, $free, $push, $pop, $empty, NoHeap);

        impl CountingHeapT<$t> for $heap {
            fn reset_comparisons() {
                unsafe { $reset() }
            }
            fn get_comparisons() -> (u64, u64) {
                unsafe { ($push_cmp(), $pop_cmp()) }
            }
        }
    };
}

impl_rqh2_heap!(
    OriginalQuickHeapI32,
    i32,
    original_quickheap_sys::OriginalQuickHeapI32,
    rqh2_i32_pq_new,
    rqh2_i32_pq_free,
    rqh2_i32_pq_push,
    rqh2_i32_pq_pop,
    rqh2_i32_pq_empty,
    NoHeap
);
impl_rqh2_heap!(
    OriginalQuickHeapI64,
    i64,
    original_quickheap_sys::OriginalQuickHeapI64,
    rqh2_i64_pq_new,
    rqh2_i64_pq_free,
    rqh2_i64_pq_push,
    rqh2_i64_pq_pop,
    rqh2_i64_pq_empty,
    OriginalQuickHeapI64Counting
);
impl_rqh2_heap!(
    OriginalQuickHeapU32,
    u32,
    original_quickheap_sys::OriginalQuickHeapU32,
    rqh2_u32_pq_new,
    rqh2_u32_pq_free,
    rqh2_u32_pq_push,
    rqh2_u32_pq_pop,
    rqh2_u32_pq_empty,
    NoHeap
);
impl_rqh2_heap!(
    OriginalQuickHeapU64,
    u64,
    original_quickheap_sys::OriginalQuickHeapU64,
    rqh2_u64_pq_new,
    rqh2_u64_pq_free,
    rqh2_u64_pq_push,
    rqh2_u64_pq_pop,
    rqh2_u64_pq_empty,
    NoHeap
);

impl_rqh2_heap!(
    counting:
    OriginalQuickHeapI64Counting, i64, original_quickheap_sys::OriginalQuickHeapI64Counting,
    rqh2_i64_counting_pq_new, rqh2_i64_counting_pq_free,
    rqh2_i64_counting_pq_push, rqh2_i64_counting_pq_pop, rqh2_i64_counting_pq_empty,
    rqh2_i64_counting_pq_reset_comparisons,
    rqh2_i64_counting_pq_push_comparisons, rqh2_i64_counting_pq_pop_comparisons
);
