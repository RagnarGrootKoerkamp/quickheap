use crate::Heap;
use crate::impls::NoHeap;
use crate::workloads::Elem;
use s3q_sys::{
    S3qI32Pq, S3qI64Pq, S3qU32Pq, S3qU64Pq, s3q_i32_pq_empty, s3q_i32_pq_free, s3q_i32_pq_new,
    s3q_i32_pq_pop, s3q_i32_pq_push, s3q_i64_pq_empty, s3q_i64_pq_free, s3q_i64_pq_new,
    s3q_i64_pq_pop, s3q_i64_pq_push, s3q_u32_pq_empty, s3q_u32_pq_free, s3q_u32_pq_new,
    s3q_u32_pq_pop, s3q_u32_pq_push, s3q_u64_pq_empty, s3q_u64_pq_free, s3q_u64_pq_new,
    s3q_u64_pq_pop, s3q_u64_pq_push,
};

macro_rules! impl_s3q_heap {
    ($heap:ident, $t:ty, $pq:ty, $new:ident, $free:ident, $push:ident, $pop:ident, $empty:ident) => {
        pub struct $heap(*mut $pq);

        impl Drop for $heap {
            fn drop(&mut self) {
                unsafe { $free(self.0) }
            }
        }

        impl Heap<$t> for $heap {
            type Casted<T2: Elem> = NoHeap;

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
}

impl_s3q_heap!(
    S3qHeapI32,
    i32,
    S3qI32Pq,
    s3q_i32_pq_new,
    s3q_i32_pq_free,
    s3q_i32_pq_push,
    s3q_i32_pq_pop,
    s3q_i32_pq_empty
);
impl_s3q_heap!(
    S3qHeapI64,
    i64,
    S3qI64Pq,
    s3q_i64_pq_new,
    s3q_i64_pq_free,
    s3q_i64_pq_push,
    s3q_i64_pq_pop,
    s3q_i64_pq_empty
);
impl_s3q_heap!(
    S3qHeapU32,
    u32,
    S3qU32Pq,
    s3q_u32_pq_new,
    s3q_u32_pq_free,
    s3q_u32_pq_push,
    s3q_u32_pq_pop,
    s3q_u32_pq_empty
);
impl_s3q_heap!(
    S3qHeapU64,
    u64,
    S3qU64Pq,
    s3q_u64_pq_new,
    s3q_u64_pq_free,
    s3q_u64_pq_push,
    s3q_u64_pq_pop,
    s3q_u64_pq_empty
);
