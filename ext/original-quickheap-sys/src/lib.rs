use std::ffi::c_void;

macro_rules! rqh2_ffi {
    (
        $Pq:ident, $t:ty,
        $new:ident, $free:ident, $push:ident, $pop:ident, $top:ident, $size:ident, $empty:ident
    ) => {
        #[repr(C)]
        pub struct $Pq(c_void);

        unsafe extern "C" {
            /// Allocates and constructs a new priority queue with the given capacity.
            /// Returns null on allocation failure. Capacity is rounded up to the next power of 2.
            pub fn $new(capacity: i32) -> *mut $Pq;
            /// Destroys and deallocates the priority queue.
            pub fn $free(pq: *mut $Pq);
            /// Inserts `item`. Returns `false` if the heap is full (capacity exceeded).
            pub fn $push(pq: *mut $Pq, item: $t) -> bool;
            /// Removes and returns the minimum element. Undefined behaviour if the queue is empty.
            pub fn $pop(pq: *mut $Pq) -> $t;
            /// Returns the minimum element without removing it. Undefined behaviour if the queue is empty.
            /// Note: may restructure the heap internally (lazy partitioning).
            pub fn $top(pq: *mut $Pq) -> $t;
            /// Returns the number of elements in the priority queue.
            pub fn $size(pq: *const $Pq) -> i32;
            /// Returns `true` if the priority queue contains no elements.
            pub fn $empty(pq: *const $Pq) -> bool;
        }
    };
}

rqh2_ffi!(
    OriginalQuickHeapI32,
    i32,
    rqh2_i32_pq_new,
    rqh2_i32_pq_free,
    rqh2_i32_pq_push,
    rqh2_i32_pq_pop,
    rqh2_i32_pq_top,
    rqh2_i32_pq_size,
    rqh2_i32_pq_empty
);

rqh2_ffi!(
    OriginalQuickHeapI64,
    i64,
    rqh2_i64_pq_new,
    rqh2_i64_pq_free,
    rqh2_i64_pq_push,
    rqh2_i64_pq_pop,
    rqh2_i64_pq_top,
    rqh2_i64_pq_size,
    rqh2_i64_pq_empty
);

rqh2_ffi!(
    OriginalQuickHeapU32,
    u32,
    rqh2_u32_pq_new,
    rqh2_u32_pq_free,
    rqh2_u32_pq_push,
    rqh2_u32_pq_pop,
    rqh2_u32_pq_top,
    rqh2_u32_pq_size,
    rqh2_u32_pq_empty
);

rqh2_ffi!(
    OriginalQuickHeapU64,
    u64,
    rqh2_u64_pq_new,
    rqh2_u64_pq_free,
    rqh2_u64_pq_push,
    rqh2_u64_pq_pop,
    rqh2_u64_pq_top,
    rqh2_u64_pq_size,
    rqh2_u64_pq_empty
);

#[repr(C)]
pub struct OriginalQuickHeapI64Counting(std::ffi::c_void);

unsafe extern "C" {
    pub fn rqh2_i64_counting_pq_new(capacity: i32) -> *mut OriginalQuickHeapI64Counting;
    pub fn rqh2_i64_counting_pq_free(pq: *mut OriginalQuickHeapI64Counting);
    pub fn rqh2_i64_counting_pq_push(pq: *mut OriginalQuickHeapI64Counting, item: i64) -> bool;
    pub fn rqh2_i64_counting_pq_pop(pq: *mut OriginalQuickHeapI64Counting) -> i64;
    pub fn rqh2_i64_counting_pq_empty(pq: *const OriginalQuickHeapI64Counting) -> bool;
    pub fn rqh2_i64_counting_pq_reset_comparisons();
    pub fn rqh2_i64_counting_pq_push_comparisons() -> u64;
    pub fn rqh2_i64_counting_pq_pop_comparisons() -> u64;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i32_basic_push_pop() {
        unsafe {
            let pq = rqh2_i32_pq_new(1024);
            assert!(!pq.is_null());
            assert!(rqh2_i32_pq_empty(pq));
            assert_eq!(rqh2_i32_pq_size(pq), 0);

            for n in [5i32, -3, 8, -1, 4] {
                assert!(rqh2_i32_pq_push(pq, n));
            }
            assert_eq!(rqh2_i32_pq_size(pq), 5);
            assert_eq!(rqh2_i32_pq_top(pq), -3);

            let mut out = Vec::new();
            while !rqh2_i32_pq_empty(pq) {
                out.push(rqh2_i32_pq_pop(pq));
            }
            assert_eq!(out, vec![-3, -1, 4, 5, 8]);

            rqh2_i32_pq_free(pq);
        }
    }

    #[test]
    fn u32_basic_push_pop() {
        unsafe {
            let pq = rqh2_u32_pq_new(1024);
            assert!(!pq.is_null());

            for n in [5u32, 3, 8, 1, 4] {
                assert!(rqh2_u32_pq_push(pq, n));
            }
            assert_eq!(rqh2_u32_pq_top(pq), 1);

            let mut out = Vec::new();
            while !rqh2_u32_pq_empty(pq) {
                out.push(rqh2_u32_pq_pop(pq));
            }
            assert_eq!(out, vec![1, 3, 4, 5, 8]);

            rqh2_u32_pq_free(pq);
        }
    }

    #[test]
    fn i64_basic_push_pop() {
        unsafe {
            let pq = rqh2_i64_pq_new(1024);
            assert!(!pq.is_null());

            for n in [5i64, -3, 8, -1, 4] {
                assert!(rqh2_i64_pq_push(pq, n));
            }
            assert_eq!(rqh2_i64_pq_top(pq), -3);

            let mut out = Vec::new();
            while !rqh2_i64_pq_empty(pq) {
                out.push(rqh2_i64_pq_pop(pq));
            }
            assert_eq!(out, vec![-3, -1, 4, 5, 8]);

            rqh2_i64_pq_free(pq);
        }
    }

    #[test]
    fn u64_basic_push_pop() {
        unsafe {
            let pq = rqh2_u64_pq_new(1024);
            assert!(!pq.is_null());

            for n in [5u64, 3, 8, 1, 4] {
                assert!(rqh2_u64_pq_push(pq, n));
            }

            let mut out = Vec::new();
            while !rqh2_u64_pq_empty(pq) {
                out.push(rqh2_u64_pq_pop(pq));
            }
            assert_eq!(out, vec![1, 3, 4, 5, 8]);

            rqh2_u64_pq_free(pq);
        }
    }
}
