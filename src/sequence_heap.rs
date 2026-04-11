use crate::Heap;
use crate::impls::NoHeap;
use crate::workloads::CountingHeapT;
use sequence_heap_sys::{
    SeqHeapI32, SeqHeapI64, SeqHeapU32, SeqHeapU64, seq_heap_i32_empty, seq_heap_i32_free,
    seq_heap_i32_new, seq_heap_i32_pop, seq_heap_i32_push, seq_heap_i64_empty, seq_heap_i64_free,
    seq_heap_i64_new, seq_heap_i64_pop, seq_heap_i64_push, seq_heap_u32_empty, seq_heap_u32_free,
    seq_heap_u32_new, seq_heap_u32_pop, seq_heap_u32_push, seq_heap_u64_empty, seq_heap_u64_free,
    seq_heap_u64_new, seq_heap_u64_pop, seq_heap_u64_push,
};

macro_rules! impl_sequence_heap {
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
}

impl_sequence_heap!(
    SequenceHeapI32,
    i32,
    SeqHeapI32,
    seq_heap_i32_new,
    seq_heap_i32_free,
    seq_heap_i32_push,
    seq_heap_i32_pop,
    seq_heap_i32_empty,
    NoHeap
);
impl_sequence_heap!(
    SequenceHeapI64,
    i64,
    SeqHeapI64,
    seq_heap_i64_new,
    seq_heap_i64_free,
    seq_heap_i64_push,
    seq_heap_i64_pop,
    seq_heap_i64_empty,
    SequenceHeapI64Counting
);
impl_sequence_heap!(
    SequenceHeapU32,
    u32,
    SeqHeapU32,
    seq_heap_u32_new,
    seq_heap_u32_free,
    seq_heap_u32_push,
    seq_heap_u32_pop,
    seq_heap_u32_empty,
    NoHeap
);
impl_sequence_heap!(
    SequenceHeapU64,
    u64,
    SeqHeapU64,
    seq_heap_u64_new,
    seq_heap_u64_free,
    seq_heap_u64_push,
    seq_heap_u64_pop,
    seq_heap_u64_empty,
    NoHeap
);

use sequence_heap_sys::{
    SeqHeapI64Counting, seq_heap_i64_counting_empty, seq_heap_i64_counting_free,
    seq_heap_i64_counting_new, seq_heap_i64_counting_pop, seq_heap_i64_counting_pop_comparisons,
    seq_heap_i64_counting_push, seq_heap_i64_counting_push_comparisons,
    seq_heap_i64_counting_reset_comparisons,
};

pub struct SequenceHeapI64Counting(*mut SeqHeapI64Counting);

impl Drop for SequenceHeapI64Counting {
    fn drop(&mut self) {
        unsafe { seq_heap_i64_counting_free(self.0) }
    }
}

impl Heap<i64> for SequenceHeapI64Counting {
    const MONOTONE: bool = false;
    type CountedHeap = NoHeap;

    fn default() -> Self {
        let pq = unsafe { seq_heap_i64_counting_new() };
        assert!(
            !pq.is_null(),
            "seq_heap_i64_counting_new: allocation failed"
        );
        SequenceHeapI64Counting(pq)
    }

    #[inline(always)]
    fn push(&mut self, t: i64) {
        unsafe { seq_heap_i64_counting_push(self.0, t + 1) }
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<i64> {
        unsafe {
            if seq_heap_i64_counting_empty(self.0) {
                None
            } else {
                Some(seq_heap_i64_counting_pop(self.0) - 1)
            }
        }
    }
}

impl CountingHeapT<i64> for SequenceHeapI64Counting {
    fn reset_comparisons() {
        unsafe { seq_heap_i64_counting_reset_comparisons() }
    }
    fn get_comparisons() -> (u64, u64) {
        unsafe {
            (
                seq_heap_i64_counting_push_comparisons(),
                seq_heap_i64_counting_pop_comparisons(),
            )
        }
    }
}
