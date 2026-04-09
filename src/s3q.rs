use crate::impls::NoHeap;
use crate::workloads::Elem;
use crate::Heap;
use s3q_sys::{
    s3q_i32_pq_empty, s3q_i32_pq_free, s3q_i32_pq_new, s3q_i32_pq_pop, s3q_i32_pq_push,
    s3q_i64_pq_empty, s3q_i64_pq_free, s3q_i64_pq_new, s3q_i64_pq_pop, s3q_i64_pq_push, S3qI32Pq,
    S3qI64Pq,
};

pub struct S3qHeapI32(*mut S3qI32Pq);

impl Drop for S3qHeapI32 {
    fn drop(&mut self) {
        unsafe { s3q_i32_pq_free(self.0) }
    }
}

impl Heap<i32> for S3qHeapI32 {
    type Casted<T2: Elem> = NoHeap;

    #[inline(always)]
    fn default() -> Self {
        let pq = unsafe { s3q_i32_pq_new() };
        assert!(!pq.is_null(), "s3q_i32_pq_new: allocation failed");
        S3qHeapI32(pq)
    }

    #[inline(always)]
    fn push(&mut self, t: i32) {
        unsafe { s3q_i32_pq_push(self.0, t) }
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<i32> {
        unsafe {
            if s3q_i32_pq_empty(self.0) {
                None
            } else {
                Some(s3q_i32_pq_pop(self.0))
            }
        }
    }
}

pub struct S3qHeapI64(*mut S3qI64Pq);

impl Drop for S3qHeapI64 {
    fn drop(&mut self) {
        unsafe { s3q_i64_pq_free(self.0) }
    }
}

impl Heap<i64> for S3qHeapI64 {
    type Casted<T2: Elem> = NoHeap;

    #[inline(always)]
    fn default() -> Self {
        let pq = unsafe { s3q_i64_pq_new() };
        assert!(!pq.is_null(), "s3q_i64_pq_new: allocation failed");
        S3qHeapI64(pq)
    }

    #[inline(always)]
    fn push(&mut self, t: i64) {
        unsafe { s3q_i64_pq_push(self.0, t) }
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<i64> {
        unsafe {
            if s3q_i64_pq_empty(self.0) {
                None
            } else {
                Some(s3q_i64_pq_pop(self.0))
            }
        }
    }
}
