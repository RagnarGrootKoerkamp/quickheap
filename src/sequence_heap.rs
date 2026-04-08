use crate::impls::NoHeap;
use crate::workloads::Elem;
use crate::Heap;
use sequence_heap_sys::{
    seq_heap_i32_empty, seq_heap_i32_free, seq_heap_i32_new, seq_heap_i32_pop, seq_heap_i32_push,
    seq_heap_i64_empty, seq_heap_i64_free, seq_heap_i64_new, seq_heap_i64_pop, seq_heap_i64_push,
    SeqHeapI32, SeqHeapI64,
};

pub struct SequenceHeapI32(*mut SeqHeapI32);

impl Drop for SequenceHeapI32 {
    fn drop(&mut self) {
        unsafe { seq_heap_i32_free(self.0) }
    }
}

impl Heap<i32> for SequenceHeapI32 {
    type Casted<T2: Elem> = NoHeap;

    #[inline(always)]
    fn default() -> Self {
        let pq = unsafe { seq_heap_i32_new() };
        assert!(!pq.is_null(), "seq_heap_i32_new: allocation failed");
        SequenceHeapI32(pq)
    }

    #[inline(always)]
    fn push(&mut self, t: i32) {
        unsafe { seq_heap_i32_push(self.0, t) }
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<i32> {
        unsafe {
            if seq_heap_i32_empty(self.0) {
                None
            } else {
                Some(seq_heap_i32_pop(self.0))
            }
        }
    }
}

pub struct SequenceHeapI64(*mut SeqHeapI64);

impl Drop for SequenceHeapI64 {
    fn drop(&mut self) {
        unsafe { seq_heap_i64_free(self.0) }
    }
}

impl Heap<i64> for SequenceHeapI64 {
    type Casted<T2: Elem> = NoHeap;

    #[inline(always)]
    fn default() -> Self {
        let pq = unsafe { seq_heap_i64_new() };
        assert!(!pq.is_null(), "seq_heap_i64_new: allocation failed");
        SequenceHeapI64(pq)
    }

    #[inline(always)]
    fn push(&mut self, t: i64) {
        unsafe { seq_heap_i64_push(self.0, t) }
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<i64> {
        unsafe {
            if seq_heap_i64_empty(self.0) {
                None
            } else {
                Some(seq_heap_i64_pop(self.0))
            }
        }
    }
}
