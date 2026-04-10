use std::ffi::c_int;
use std::ffi::c_void;

macro_rules! seq_heap_ffi {
    (
        $Pq:ident, $t:ty,
        $new:ident, $free:ident, $push:ident, $pop:ident, $top:ident, $size:ident, $empty:ident
    ) => {
        #[repr(C)]
        pub struct $Pq(c_void);

        unsafe extern "C" {
            /// Allocates and constructs a new priority queue. Returns null on allocation failure.
            pub fn $new() -> *mut $Pq;
            /// Destroys and deallocates the priority queue.
            pub fn $free(pq: *mut $Pq);
            /// Pushes `key` onto the priority queue.
            /// # Safety
            /// Key must not equal the MIN or MAX of the type.
            pub fn $push(pq: *mut $Pq, key: $t);
            /// Removes and returns the minimum element. Undefined behaviour if the queue is empty.
            pub fn $pop(pq: *mut $Pq) -> $t;
            /// Returns the minimum element without removing it. Undefined behaviour if the queue is empty.
            pub fn $top(pq: *mut $Pq) -> $t;
            /// Returns the number of elements in the priority queue.
            pub fn $size(pq: *const $Pq) -> c_int;
            /// Returns `true` if the priority queue contains no elements.
            pub fn $empty(pq: *const $Pq) -> bool;
        }
    };
}

seq_heap_ffi!(
    SeqHeapI32,
    i32,
    seq_heap_i32_new,
    seq_heap_i32_free,
    seq_heap_i32_push,
    seq_heap_i32_pop,
    seq_heap_i32_top,
    seq_heap_i32_size,
    seq_heap_i32_empty
);

seq_heap_ffi!(
    SeqHeapI64,
    i64,
    seq_heap_i64_new,
    seq_heap_i64_free,
    seq_heap_i64_push,
    seq_heap_i64_pop,
    seq_heap_i64_top,
    seq_heap_i64_size,
    seq_heap_i64_empty
);

seq_heap_ffi!(
    SeqHeapU32,
    u32,
    seq_heap_u32_new,
    seq_heap_u32_free,
    seq_heap_u32_push,
    seq_heap_u32_pop,
    seq_heap_u32_top,
    seq_heap_u32_size,
    seq_heap_u32_empty
);

seq_heap_ffi!(
    SeqHeapU64,
    u64,
    seq_heap_u64_new,
    seq_heap_u64_free,
    seq_heap_u64_push,
    seq_heap_u64_pop,
    seq_heap_u64_top,
    seq_heap_u64_size,
    seq_heap_u64_empty
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i32_basic_push_pop() {
        unsafe {
            let pq = seq_heap_i32_new();
            assert!(!pq.is_null());
            assert!(seq_heap_i32_empty(pq));
            assert_eq!(seq_heap_i32_size(pq), 0);

            for n in [5i32, -3, 8, -1, 4] {
                seq_heap_i32_push(pq, n);
            }
            assert_eq!(seq_heap_i32_size(pq), 5);
            assert_eq!(seq_heap_i32_top(pq), -3);

            let mut out = Vec::new();
            while !seq_heap_i32_empty(pq) {
                out.push(seq_heap_i32_pop(pq));
            }
            assert_eq!(out, vec![-3, -1, 4, 5, 8]);

            seq_heap_i32_free(pq);
        }
    }

    #[test]
    fn i64_basic_push_pop() {
        unsafe {
            let pq = seq_heap_i64_new();
            assert!(!pq.is_null());
            assert!(seq_heap_i64_empty(pq));
            assert_eq!(seq_heap_i64_size(pq), 0);

            for n in [5i64, -3, 8, -1, 4] {
                seq_heap_i64_push(pq, n);
            }
            assert_eq!(seq_heap_i64_size(pq), 5);
            assert_eq!(seq_heap_i64_top(pq), -3);

            let mut out = Vec::new();
            while !seq_heap_i64_empty(pq) {
                out.push(seq_heap_i64_pop(pq));
            }
            assert_eq!(out, vec![-3, -1, 4, 5, 8]);

            seq_heap_i64_free(pq);
        }
    }
}
