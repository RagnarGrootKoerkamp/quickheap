use std::ffi::c_void;

macro_rules! radix_heap_ffi {
    (
        $Heap:ident, $t:ty,
        $new:ident, $free:ident, $push:ident, $pop:ident, $top:ident, $size:ident, $empty:ident, $clear:ident
    ) => {
        #[repr(C)]
        pub struct $Heap(c_void);

        unsafe extern "C" {
            /// Allocates and constructs a new radix heap.
            /// Returns null on allocation failure.
            pub fn $new() -> *mut $Heap;
            /// Destroys and deallocates the radix heap.
            pub fn $free(heap: *mut $Heap);
            /// Inserts `item` into the heap.
            /// Note: The item must be >= the last popped item (monotone heap property).
            pub fn $push(heap: *mut $Heap, item: $t);
            /// Removes and returns the minimum element. Undefined behaviour if the heap is empty.
            pub fn $pop(heap: *mut $Heap) -> $t;
            /// Returns the minimum element without removing it. Undefined behaviour if the heap is empty.
            pub fn $top(heap: *mut $Heap) -> $t;
            /// Returns the number of elements in the heap.
            pub fn $size(heap: *const $Heap) -> usize;
            /// Returns `true` if the heap contains no elements.
            pub fn $empty(heap: *const $Heap) -> bool;
            /// Clears all elements from the heap.
            pub fn $clear(heap: *mut $Heap);
        }
    };
}

radix_heap_ffi!(
    RadixHeapI32,
    i32,
    radix_heap_i32_new,
    radix_heap_i32_free,
    radix_heap_i32_push,
    radix_heap_i32_pop,
    radix_heap_i32_top,
    radix_heap_i32_size,
    radix_heap_i32_empty,
    radix_heap_i32_clear
);

radix_heap_ffi!(
    RadixHeapI64,
    i64,
    radix_heap_i64_new,
    radix_heap_i64_free,
    radix_heap_i64_push,
    radix_heap_i64_pop,
    radix_heap_i64_top,
    radix_heap_i64_size,
    radix_heap_i64_empty,
    radix_heap_i64_clear
);

radix_heap_ffi!(
    RadixHeapU32,
    u32,
    radix_heap_u32_new,
    radix_heap_u32_free,
    radix_heap_u32_push,
    radix_heap_u32_pop,
    radix_heap_u32_top,
    radix_heap_u32_size,
    radix_heap_u32_empty,
    radix_heap_u32_clear
);

radix_heap_ffi!(
    RadixHeapU64,
    u64,
    radix_heap_u64_new,
    radix_heap_u64_free,
    radix_heap_u64_push,
    radix_heap_u64_pop,
    radix_heap_u64_top,
    radix_heap_u64_size,
    radix_heap_u64_empty,
    radix_heap_u64_clear
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i32_basic_push_pop() {
        unsafe {
            let heap = radix_heap_i32_new();
            assert!(!heap.is_null());
            assert!(radix_heap_i32_empty(heap));
            assert_eq!(radix_heap_i32_size(heap), 0);

            // Note: radix heap requires monotone property (items must be >= last popped)
            for n in [1i32, 3, 5, 8, 10] {
                radix_heap_i32_push(heap, n);
            }
            assert_eq!(radix_heap_i32_size(heap), 5);
            assert_eq!(radix_heap_i32_top(heap), 1);

            let mut out = Vec::new();
            while !radix_heap_i32_empty(heap) {
                out.push(radix_heap_i32_pop(heap));
            }
            assert_eq!(out, vec![1, 3, 5, 8, 10]);

            radix_heap_i32_free(heap);
        }
    }

    #[test]
    fn u32_basic_push_pop() {
        unsafe {
            let heap = radix_heap_u32_new();
            assert!(!heap.is_null());

            for n in [1u32, 3, 5, 8, 10] {
                radix_heap_u32_push(heap, n);
            }
            assert_eq!(radix_heap_u32_top(heap), 1);

            let mut out = Vec::new();
            while !radix_heap_u32_empty(heap) {
                out.push(radix_heap_u32_pop(heap));
            }
            assert_eq!(out, vec![1, 3, 5, 8, 10]);

            radix_heap_u32_free(heap);
        }
    }

    #[test]
    fn i64_basic_push_pop() {
        unsafe {
            let heap = radix_heap_i64_new();
            assert!(!heap.is_null());

            for n in [1i64, 3, 5, 8, 10] {
                radix_heap_i64_push(heap, n);
            }
            assert_eq!(radix_heap_i64_top(heap), 1);

            let mut out = Vec::new();
            while !radix_heap_i64_empty(heap) {
                out.push(radix_heap_i64_pop(heap));
            }
            assert_eq!(out, vec![1, 3, 5, 8, 10]);

            radix_heap_i64_free(heap);
        }
    }

    #[test]
    fn u64_basic_push_pop() {
        unsafe {
            let heap = radix_heap_u64_new();
            assert!(!heap.is_null());

            for n in [1u64, 3, 5, 8, 10] {
                radix_heap_u64_push(heap, n);
            }

            let mut out = Vec::new();
            while !radix_heap_u64_empty(heap) {
                out.push(radix_heap_u64_pop(heap));
            }
            assert_eq!(out, vec![1, 3, 5, 8, 10]);

            radix_heap_u64_free(heap);
        }
    }

    #[test]
    fn test_clear() {
        unsafe {
            let heap = radix_heap_u32_new();
            
            for n in [1u32, 3, 5, 8, 10] {
                radix_heap_u32_push(heap, n);
            }
            assert_eq!(radix_heap_u32_size(heap), 5);

            radix_heap_u32_clear(heap);
            assert_eq!(radix_heap_u32_size(heap), 0);
            assert!(radix_heap_u32_empty(heap));

            // Should be able to use heap again after clear
            radix_heap_u32_push(heap, 42);
            assert_eq!(radix_heap_u32_size(heap), 1);
            assert_eq!(radix_heap_u32_pop(heap), 42);

            radix_heap_u32_free(heap);
        }
    }
}
