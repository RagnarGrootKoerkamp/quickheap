use crate::Heap;
use crate::impls::NoHeap;
use crate::workloads::Elem;
use sequence_heap_sys::{
    SeqHeapI32, SeqHeapI64, SeqHeapU32, SeqHeapU64, seq_heap_i32_empty, seq_heap_i32_free,
    seq_heap_i32_new, seq_heap_i32_pop, seq_heap_i32_push, seq_heap_i64_empty, seq_heap_i64_free,
    seq_heap_i64_new, seq_heap_i64_pop, seq_heap_i64_push, seq_heap_u32_empty, seq_heap_u32_free,
    seq_heap_u32_new, seq_heap_u32_pop, seq_heap_u32_push, seq_heap_u64_empty, seq_heap_u64_free,
    seq_heap_u64_new, seq_heap_u64_pop, seq_heap_u64_push,
};

macro_rules! impl_sequence_heap {
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

impl_sequence_heap!(
    SequenceHeapI32,
    i32,
    SeqHeapI32,
    seq_heap_i32_new,
    seq_heap_i32_free,
    seq_heap_i32_push,
    seq_heap_i32_pop,
    seq_heap_i32_empty
);
impl_sequence_heap!(
    SequenceHeapI64,
    i64,
    SeqHeapI64,
    seq_heap_i64_new,
    seq_heap_i64_free,
    seq_heap_i64_push,
    seq_heap_i64_pop,
    seq_heap_i64_empty
);
impl_sequence_heap!(
    SequenceHeapU32,
    u32,
    SeqHeapU32,
    seq_heap_u32_new,
    seq_heap_u32_free,
    seq_heap_u32_push,
    seq_heap_u32_pop,
    seq_heap_u32_empty
);
impl_sequence_heap!(
    SequenceHeapU64,
    u64,
    SeqHeapU64,
    seq_heap_u64_new,
    seq_heap_u64_free,
    seq_heap_u64_push,
    seq_heap_u64_pop,
    seq_heap_u64_empty
);
