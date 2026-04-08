use crate::workloads::Elem;
use crate::Heap;
use crate::impls::NoHeap;
use s3q_sys::{
    S3qI32Pq, S3qI64Pq,
    s3q_i32_pq_empty, s3q_i32_pq_free, s3q_i32_pq_new, s3q_i32_pq_pop, s3q_i32_pq_push,
    s3q_i64_pq_empty, s3q_i64_pq_free, s3q_i64_pq_new, s3q_i64_pq_pop, s3q_i64_pq_push,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn heap_sort_i32<H: Heap<i32>>(inputs: &[i32]) -> Vec<i32> {
        let mut h = H::default();
        for &x in inputs {
            h.push(x);
        }
        let mut out = Vec::new();
        while let Some(v) = h.pop() {
            out.push(v);
        }
        out
    }

    #[test]
    fn i32_sorts_correctly() {
        let input = vec![5, -3, 8, -1, 4, 0, -100, 99];
        let mut expected = input.clone();
        expected.sort();
        assert_eq!(heap_sort_i32::<S3qHeapI32>(&input), expected);
    }

    #[test]
    fn i64_sorts_correctly() {
        // s3q uses an open key range (exclusive of i64::MIN and i64::MAX)
        let input: Vec<i64> = vec![5, -3, 8, -1, 4, 0, i64::MIN + 1, i64::MAX - 1];
        let mut h = S3qHeapI64::default();
        for &x in &input {
            h.push(x);
        }
        let mut out = Vec::new();
        while let Some(v) = h.pop() {
            out.push(v);
        }
        let mut expected = input.clone();
        expected.sort();
        assert_eq!(out, expected);
    }

    #[test]
    fn i32_pop_empty_returns_none() {
        let mut h = S3qHeapI32::default();
        assert_eq!(h.pop(), None);
        h.push(42);
        assert_eq!(h.pop(), Some(42));
        assert_eq!(h.pop(), None);
    }
}
