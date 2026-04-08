use std::ffi::c_void;

/// Opaque handle to a `s3q::PriorityQueue<U32Cfg>` instance.
#[repr(C)]
pub struct S3qU32Pq(c_void);

unsafe extern "C" {
    /// Allocates and constructs a new priority queue. Returns null on allocation failure.
    pub fn s3q_u32_pq_new() -> *mut S3qU32Pq;

    /// Destroys and deallocates the priority queue.
    pub fn s3q_u32_pq_free(pq: *mut S3qU32Pq);

    /// Pushes `item` onto the priority queue.
    pub fn s3q_u32_pq_push(pq: *mut S3qU32Pq, item: u32);

    /// Removes and returns the minimum element. Undefined behaviour if the queue is empty.
    pub fn s3q_u32_pq_pop(pq: *mut S3qU32Pq) -> u32;

    /// Returns the minimum element without removing it. Undefined behaviour if the queue is empty.
    pub fn s3q_u32_pq_top(pq: *const S3qU32Pq) -> u32;

    /// Returns the number of elements in the priority queue.
    pub fn s3q_u32_pq_size(pq: *const S3qU32Pq) -> usize;

    /// Returns `true` if the priority queue contains no elements.
    pub fn s3q_u32_pq_empty(pq: *const S3qU32Pq) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_push_pop() {
        unsafe {
            let pq = s3q_u32_pq_new();
            assert!(!pq.is_null());
            assert!(s3q_u32_pq_empty(pq));
            assert_eq!(s3q_u32_pq_size(pq), 0);

            for n in [5u32, 3, 8, 1, 4] {
                s3q_u32_pq_push(pq, n);
            }
            assert_eq!(s3q_u32_pq_size(pq), 5);
            assert_eq!(s3q_u32_pq_top(pq), 1);

            let mut out = Vec::new();
            while !s3q_u32_pq_empty(pq) {
                out.push(s3q_u32_pq_pop(pq));
            }
            assert_eq!(out, vec![1, 3, 4, 5, 8]);

            s3q_u32_pq_free(pq);
        }
    }
}
