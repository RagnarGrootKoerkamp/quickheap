use crate::Heap;
use crate::impls::NoHeap;
use crate::workloads::CountingHeapT;
use s3q_sys::{
    S3qI32Pq, S3qI64Pq, S3qU32Pq, S3qU64Pq, s3q_i32_pq_empty, s3q_i32_pq_free, s3q_i32_pq_new,
    s3q_i32_pq_pop, s3q_i32_pq_push, s3q_i64_pq_empty, s3q_i64_pq_free, s3q_i64_pq_new,
    s3q_i64_pq_pop, s3q_i64_pq_push, s3q_u32_pq_empty, s3q_u32_pq_free, s3q_u32_pq_new,
    s3q_u32_pq_pop, s3q_u32_pq_push, s3q_u64_pq_empty, s3q_u64_pq_free, s3q_u64_pq_new,
    s3q_u64_pq_pop, s3q_u64_pq_push,
};

macro_rules! impl_s3q_heap {
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
                let pq = unsafe { $new() };
                assert!(
                    !pq.is_null(),
                    concat!(stringify!($new), ": allocation failed")
                );
                $heap(pq)
            }

            #[inline(always)]
            fn push(&mut self, t: $t) {
                unsafe { $push(self.0, t + 1) }
            }

            #[inline(always)]
            fn pop(&mut self) -> Option<$t> {
                unsafe {
                    if $empty(self.0) {
                        None
                    } else {
                        Some($pop(self.0) - 1)
                    }
                }
            }
        }
    };
    (counting: $heap:ident, $t:ty, $pq:ty, $new:ident, $free:ident, $push:ident, $pop:ident, $empty:ident, $reset:ident, $push_cmp:ident, $pop_cmp:ident) => {
        impl_s3q_heap!($heap, $t, $pq, $new, $free, $push, $pop, $empty, NoHeap);

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

impl_s3q_heap!(
    S3qHeapI32,
    i32,
    S3qI32Pq,
    s3q_i32_pq_new,
    s3q_i32_pq_free,
    s3q_i32_pq_push,
    s3q_i32_pq_pop,
    s3q_i32_pq_empty,
    NoHeap
);
impl_s3q_heap!(
    S3qHeapI64,
    i64,
    S3qI64Pq,
    s3q_i64_pq_new,
    s3q_i64_pq_free,
    s3q_i64_pq_push,
    s3q_i64_pq_pop,
    s3q_i64_pq_empty,
    S3qHeapI64Counting
);
impl_s3q_heap!(
    S3qHeapU32,
    u32,
    S3qU32Pq,
    s3q_u32_pq_new,
    s3q_u32_pq_free,
    s3q_u32_pq_push,
    s3q_u32_pq_pop,
    s3q_u32_pq_empty,
    NoHeap
);
impl_s3q_heap!(
    S3qHeapU64,
    u64,
    S3qU64Pq,
    s3q_u64_pq_new,
    s3q_u64_pq_free,
    s3q_u64_pq_push,
    s3q_u64_pq_pop,
    s3q_u64_pq_empty,
    NoHeap
);

use s3q_sys::{
    S3qI64CountingPq, s3q_i64_counting_pq_empty, s3q_i64_counting_pq_free, s3q_i64_counting_pq_new,
    s3q_i64_counting_pq_pop, s3q_i64_counting_pq_pop_comparisons, s3q_i64_counting_pq_push,
    s3q_i64_counting_pq_push_comparisons, s3q_i64_counting_pq_reset_comparisons,
};

impl_s3q_heap!(
    counting:
    S3qHeapI64Counting, i64, S3qI64CountingPq,
    s3q_i64_counting_pq_new, s3q_i64_counting_pq_free,
    s3q_i64_counting_pq_push, s3q_i64_counting_pq_pop, s3q_i64_counting_pq_empty,
    s3q_i64_counting_pq_reset_comparisons,
    s3q_i64_counting_pq_push_comparisons, s3q_i64_counting_pq_pop_comparisons
);
