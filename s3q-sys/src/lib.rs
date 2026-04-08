use std::ffi::c_void;

/// Opaque handle to a `s3q::PriorityQueue<I32Cfg>` instance.
#[repr(C)]
pub struct S3qI32Pq(c_void);

unsafe extern "C" {
    /// Allocates and constructs a new priority queue. Returns null on allocation failure.
    pub fn s3q_i32_pq_new() -> *mut S3qI32Pq;

    /// Destroys and deallocates the priority queue.
    pub fn s3q_i32_pq_free(pq: *mut S3qI32Pq);

    /// Pushes `item` onto the priority queue.
    pub fn s3q_i32_pq_push(pq: *mut S3qI32Pq, item: i32);

    /// Removes and returns the minimum element. Undefined behaviour if the queue is empty.
    pub fn s3q_i32_pq_pop(pq: *mut S3qI32Pq) -> i32;

    /// Returns the minimum element without removing it. Undefined behaviour if the queue is empty.
    pub fn s3q_i32_pq_top(pq: *const S3qI32Pq) -> i32;

    /// Returns the number of elements in the priority queue.
    pub fn s3q_i32_pq_size(pq: *const S3qI32Pq) -> usize;

    /// Returns `true` if the priority queue contains no elements.
    pub fn s3q_i32_pq_empty(pq: *const S3qI32Pq) -> bool;
}

/// Opaque handle to a `s3q::PriorityQueue<I64Cfg>` instance.
#[repr(C)]
pub struct S3qI64Pq(c_void);

unsafe extern "C" {
    /// Allocates and constructs a new priority queue. Returns null on allocation failure.
    pub fn s3q_i64_pq_new() -> *mut S3qI64Pq;

    /// Destroys and deallocates the priority queue.
    pub fn s3q_i64_pq_free(pq: *mut S3qI64Pq);

    /// Pushes `item` onto the priority queue.
    pub fn s3q_i64_pq_push(pq: *mut S3qI64Pq, item: i64);

    /// Removes and returns the minimum element. Undefined behaviour if the queue is empty.
    pub fn s3q_i64_pq_pop(pq: *mut S3qI64Pq) -> i64;

    /// Returns the minimum element without removing it. Undefined behaviour if the queue is empty.
    pub fn s3q_i64_pq_top(pq: *const S3qI64Pq) -> i64;

    /// Returns the number of elements in the priority queue.
    pub fn s3q_i64_pq_size(pq: *const S3qI64Pq) -> usize;

    /// Returns `true` if the priority queue contains no elements.
    pub fn s3q_i64_pq_empty(pq: *const S3qI64Pq) -> bool;
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
