use crate::Heap;
use crate::impls::NoHeap;
use radix_heap_sys::{
    radix_heap_i32_empty, radix_heap_i32_free, radix_heap_i32_new, radix_heap_i32_pop,
    radix_heap_i32_push, radix_heap_i64_empty, radix_heap_i64_free, radix_heap_i64_new,
    radix_heap_i64_pop, radix_heap_i64_push, radix_heap_u32_empty, radix_heap_u32_free,
    radix_heap_u32_new, radix_heap_u32_pop, radix_heap_u32_push, radix_heap_u64_empty,
    radix_heap_u64_free, radix_heap_u64_new, radix_heap_u64_pop, radix_heap_u64_push,
};

macro_rules! impl_radix_heap_ffi {
    ($heap:ident, $t:ty, $pq:ty, $new:ident, $free:ident, $push:ident, $pop:ident, $empty:ident) => {
        pub struct $heap(*mut $pq);

        impl Drop for $heap {
            fn drop(&mut self) {
                unsafe { $free(self.0) }
            }
        }

        impl Heap<$t> for $heap {
            type CountedHeap = NoHeap;
            const MONOTONE: bool = true;

            #[inline(always)]
            fn default() -> Self {
                let heap = unsafe { $new() };
                assert!(
                    !heap.is_null(),
                    concat!(stringify!($new), ": allocation failed")
                );
                $heap(heap)
            }

            #[inline(always)]
            fn push(&mut self, t: $t) {
                unsafe { $push(self.0, t) };
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

impl_radix_heap_ffi!(
    RadixHeapI32,
    i32,
    radix_heap_sys::RadixHeapI32,
    radix_heap_i32_new,
    radix_heap_i32_free,
    radix_heap_i32_push,
    radix_heap_i32_pop,
    radix_heap_i32_empty
);

impl_radix_heap_ffi!(
    RadixHeapI64,
    i64,
    radix_heap_sys::RadixHeapI64,
    radix_heap_i64_new,
    radix_heap_i64_free,
    radix_heap_i64_push,
    radix_heap_i64_pop,
    radix_heap_i64_empty
);

impl_radix_heap_ffi!(
    RadixHeapU32,
    u32,
    radix_heap_sys::RadixHeapU32,
    radix_heap_u32_new,
    radix_heap_u32_free,
    radix_heap_u32_push,
    radix_heap_u32_pop,
    radix_heap_u32_empty
);

impl_radix_heap_ffi!(
    RadixHeapU64,
    u64,
    radix_heap_sys::RadixHeapU64,
    radix_heap_u64_new,
    radix_heap_u64_free,
    radix_heap_u64_push,
    radix_heap_u64_pop,
    radix_heap_u64_empty
);
