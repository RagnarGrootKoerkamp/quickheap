//! C-ABI bindings for [`SimdQuickHeap`], enabled with the `c` feature.
//!
//! These are the low-level building blocks used by the C++ wrapper in
//! `cpp/quickheap.hpp`; C or C++ code can also link against them directly.
//! One set of `quickheap_<ty>_*` functions is generated for each of the four
//! supported element types (`i32`, `u32`, `i64`, `u64`).
//!
//! Each heap must be created with `quickheap_<ty>_new` and released with
//! `quickheap_<ty>_free`. Passing a null pointer to any function is
//! undefined behaviour (checked with an assertion in debug builds via the
//! panics below).

#![allow(non_snake_case)]

use crate::SimdQuickHeap;

/// Defines the opaque type and `extern "C"` functions for one element type.
///
/// `$Elem`: the Rust element type (e.g. `i32`).
/// `$Opaque`: name of the opaque handle type alias (e.g. `QuickHeapI32`).
/// `$new`/`$free`/`$push`/`$pop`/`$len`/`$is_empty`/`$capacity`: exported
/// function names, following the `quickheap_<ty>_<op>` convention.
macro_rules! define_heap_c_api {
    (
        $Elem:ty,
        $Opaque:ident,
        $new:ident,
        $free:ident,
        $push:ident,
        $pop:ident,
        $len:ident,
        $is_empty:ident,
        $capacity:ident
    ) => {
        #[doc = concat!("Opaque `SimdQuickHeap<", stringify!($Elem), ">` handle.")]
        pub type $Opaque = SimdQuickHeap<$Elem>;

        #[doc = concat!("Create a new, empty `SimdQuickHeap<", stringify!($Elem), ">`.\n\nMust be freed with [`", stringify!($free), "`].")]
        #[unsafe(no_mangle)]
        pub extern "C" fn $new() -> *mut $Opaque {
            Box::into_raw(Box::new(SimdQuickHeap::<$Elem>::default()))
        }

        #[doc = concat!("Free a heap previously created with [`", stringify!($new), "`].")]
        #[unsafe(no_mangle)]
        pub extern "C" fn $free(ptr: *mut $Opaque) {
            assert!(!ptr.is_null(), "Heap pointer must not be null");
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }

        #[doc = "Push `value` onto the heap."]
        #[unsafe(no_mangle)]
        pub extern "C" fn $push(ptr: *mut $Opaque, value: $Elem) {
            assert!(!ptr.is_null(), "Heap pointer must not be null");
            let heap = unsafe { &mut *ptr };
            heap.push(value);
        }

        #[doc = "Pop the smallest value from the heap into `*out_value`.\n\nReturns `true` and writes the popped value on success, or returns\n`false` (leaving `*out_value` untouched) if the heap is empty."]
        #[unsafe(no_mangle)]
        pub extern "C" fn $pop(ptr: *mut $Opaque, out_value: *mut $Elem) -> bool {
            assert!(!ptr.is_null(), "Heap pointer must not be null");
            assert!(!out_value.is_null(), "Output pointer must not be null");
            let heap = unsafe { &mut *ptr };
            match heap.pop() {
                Some(v) => {
                    unsafe { *out_value = v };
                    true
                }
                None => false,
            }
        }

        #[doc = "Return the number of elements currently in the heap."]
        #[unsafe(no_mangle)]
        pub extern "C" fn $len(ptr: *const $Opaque) -> usize {
            assert!(!ptr.is_null(), "Heap pointer must not be null");
            unsafe { &*ptr }.len()
        }

        #[doc = "Return whether the heap is empty."]
        #[unsafe(no_mangle)]
        pub extern "C" fn $is_empty(ptr: *const $Opaque) -> bool {
            assert!(!ptr.is_null(), "Heap pointer must not be null");
            unsafe { &*ptr }.is_empty()
        }

        #[doc = "Return the total capacity currently allocated over all internal buckets."]
        #[unsafe(no_mangle)]
        pub extern "C" fn $capacity(ptr: *const $Opaque) -> usize {
            assert!(!ptr.is_null(), "Heap pointer must not be null");
            unsafe { &*ptr }.capacity()
        }
    };
}

define_heap_c_api!(
    i32,
    QuickHeapI32,
    quickheap_i32_new,
    quickheap_i32_free,
    quickheap_i32_push,
    quickheap_i32_pop,
    quickheap_i32_len,
    quickheap_i32_is_empty,
    quickheap_i32_capacity
);

define_heap_c_api!(
    u32,
    QuickHeapU32,
    quickheap_u32_new,
    quickheap_u32_free,
    quickheap_u32_push,
    quickheap_u32_pop,
    quickheap_u32_len,
    quickheap_u32_is_empty,
    quickheap_u32_capacity
);

define_heap_c_api!(
    i64,
    QuickHeapI64,
    quickheap_i64_new,
    quickheap_i64_free,
    quickheap_i64_push,
    quickheap_i64_pop,
    quickheap_i64_len,
    quickheap_i64_is_empty,
    quickheap_i64_capacity
);

define_heap_c_api!(
    u64,
    QuickHeapU64,
    quickheap_u64_new,
    quickheap_u64_free,
    quickheap_u64_push,
    quickheap_u64_pop,
    quickheap_u64_len,
    quickheap_u64_is_empty,
    quickheap_u64_capacity
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_i32() {
        unsafe {
            let h = quickheap_i32_new();
            assert!(quickheap_i32_is_empty(h));
            quickheap_i32_push(h, 4);
            quickheap_i32_push(h, 1);
            quickheap_i32_push(h, 7);
            assert_eq!(quickheap_i32_len(h), 3);

            let mut out = 0i32;
            assert!(quickheap_i32_pop(h, &mut out));
            assert_eq!(out, 1);
            assert!(quickheap_i32_pop(h, &mut out));
            assert_eq!(out, 4);
            assert!(quickheap_i32_pop(h, &mut out));
            assert_eq!(out, 7);
            assert!(!quickheap_i32_pop(h, &mut out));
            assert!(quickheap_i32_is_empty(h));

            quickheap_i32_free(h);
        }
    }
}
