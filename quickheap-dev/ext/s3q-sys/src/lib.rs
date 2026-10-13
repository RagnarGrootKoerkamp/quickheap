use std::ffi::c_void;

macro_rules! s3q_ffi {
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
            /// Pushes `item` onto the priority queue.
            /// # Safety
            /// `item` must not equal the MIN or MAX of the type.
            pub fn $push(pq: *mut $Pq, item: $t);
            /// Removes and returns the minimum element. Undefined behaviour if the queue is empty.
            pub fn $pop(pq: *mut $Pq) -> $t;
            /// Returns the minimum element without removing it. Undefined behaviour if the queue is empty.
            pub fn $top(pq: *const $Pq) -> $t;
            /// Returns the number of elements in the priority queue.
            pub fn $size(pq: *const $Pq) -> usize;
            /// Returns `true` if the priority queue contains no elements.
            pub fn $empty(pq: *const $Pq) -> bool;
        }
    };
}

s3q_ffi!(
    S3qI32Pq,
    i32,
    s3q_i32_pq_new,
    s3q_i32_pq_free,
    s3q_i32_pq_push,
    s3q_i32_pq_pop,
    s3q_i32_pq_top,
    s3q_i32_pq_size,
    s3q_i32_pq_empty
);

s3q_ffi!(
    S3qI64Pq,
    i64,
    s3q_i64_pq_new,
    s3q_i64_pq_free,
    s3q_i64_pq_push,
    s3q_i64_pq_pop,
    s3q_i64_pq_top,
    s3q_i64_pq_size,
    s3q_i64_pq_empty
);

s3q_ffi!(
    S3qU32Pq,
    u32,
    s3q_u32_pq_new,
    s3q_u32_pq_free,
    s3q_u32_pq_push,
    s3q_u32_pq_pop,
    s3q_u32_pq_top,
    s3q_u32_pq_size,
    s3q_u32_pq_empty
);

s3q_ffi!(
    S3qU64Pq,
    u64,
    s3q_u64_pq_new,
    s3q_u64_pq_free,
    s3q_u64_pq_push,
    s3q_u64_pq_pop,
    s3q_u64_pq_top,
    s3q_u64_pq_size,
    s3q_u64_pq_empty
);

#[repr(C)]
pub struct S3qI64CountingPq(c_void);

unsafe extern "C" {
    pub fn s3q_i64_counting_pq_new() -> *mut S3qI64CountingPq;
    pub fn s3q_i64_counting_pq_free(pq: *mut S3qI64CountingPq);
    pub fn s3q_i64_counting_pq_push(pq: *mut S3qI64CountingPq, item: i64);
    pub fn s3q_i64_counting_pq_pop(pq: *mut S3qI64CountingPq) -> i64;
    pub fn s3q_i64_counting_pq_empty(pq: *const S3qI64CountingPq) -> bool;
    pub fn s3q_i64_counting_pq_reset_comparisons();
    pub fn s3q_i64_counting_pq_push_comparisons() -> u64;
    pub fn s3q_i64_counting_pq_pop_comparisons() -> u64;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i32_basic_push_pop() {
        unsafe {
            let pq = s3q_i32_pq_new();
            assert!(!pq.is_null());
            assert!(s3q_i32_pq_empty(pq));
            assert_eq!(s3q_i32_pq_size(pq), 0);

            for n in [5i32, -3, 8, -1, 4] {
                s3q_i32_pq_push(pq, n);
            }
            assert_eq!(s3q_i32_pq_size(pq), 5);
            assert_eq!(s3q_i32_pq_top(pq), -3);

            let mut out = Vec::new();
            while !s3q_i32_pq_empty(pq) {
                out.push(s3q_i32_pq_pop(pq));
            }
            assert_eq!(out, vec![-3, -1, 4, 5, 8]);

            s3q_i32_pq_free(pq);
        }
    }

    #[test]
    fn i64_basic_push_pop() {
        unsafe {
            let pq = s3q_i64_pq_new();
            assert!(!pq.is_null());
            assert!(s3q_i64_pq_empty(pq));
            assert_eq!(s3q_i64_pq_size(pq), 0);

            for n in [5i64, -3, 8, -1, 4] {
                s3q_i64_pq_push(pq, n);
            }
            assert_eq!(s3q_i64_pq_size(pq), 5);
            assert_eq!(s3q_i64_pq_top(pq), -3);

            let mut out = Vec::new();
            while !s3q_i64_pq_empty(pq) {
                out.push(s3q_i64_pq_pop(pq));
            }
            assert_eq!(out, vec![-3, -1, 4, 5, 8]);

            s3q_i64_pq_free(pq);
        }
    }
}

#[cfg(test)]
mod counting_tests {
    use super::*;

    #[test]
    fn counting_works() {
        unsafe {
            let pq = s3q_i64_counting_pq_new();
            assert!(!pq.is_null());
            s3q_i64_counting_pq_reset_comparisons();
            
            for i in 0..100i64 {
                s3q_i64_counting_pq_push(pq, (i * 7) % 100 + 1);
            }
            
            let push_cmp = s3q_i64_counting_pq_push_comparisons();
            eprintln!("push comparisons: {push_cmp}");
            
            for _ in 0..100 {
                s3q_i64_counting_pq_pop(pq);
            }
            
            let pop_cmp = s3q_i64_counting_pq_pop_comparisons();
            eprintln!("pop comparisons: {pop_cmp}");
            
            assert!(push_cmp > 0, "expected push_cmp > 0, got {push_cmp}");
            assert!(pop_cmp > 0, "expected pop_cmp > 0, got {pop_cmp}");
            
            s3q_i64_counting_pq_free(pq);
        }
    }
}
