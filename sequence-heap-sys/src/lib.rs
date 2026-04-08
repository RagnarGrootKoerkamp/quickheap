use std::ffi::c_int;
use std::ffi::c_void;

/// Opaque handle to a `KNHeap<int32_t, int32_t>` instance.
#[repr(C)]
pub struct SeqHeapI32(c_void);

unsafe extern "C" {
    /// Allocates and constructs a new priority queue. Returns null on allocation failure.
    pub fn seq_heap_i32_new() -> *mut SeqHeapI32;

    /// Destroys and deallocates the priority queue.
    pub fn seq_heap_i32_free(pq: *mut SeqHeapI32);

    /// Pushes `key` onto the priority queue.
    /// # Safety
    /// `key` must not equal `i32::MIN` or `i32::MAX` (reserved as sentinels).
    pub fn seq_heap_i32_push(pq: *mut SeqHeapI32, key: i32);

    /// Removes and returns the minimum element. Undefined behaviour if the queue is empty.
    pub fn seq_heap_i32_pop(pq: *mut SeqHeapI32) -> i32;

    /// Returns the minimum element without removing it. Undefined behaviour if the queue is empty.
    pub fn seq_heap_i32_top(pq: *mut SeqHeapI32) -> i32;

    /// Returns the number of elements in the priority queue.
    pub fn seq_heap_i32_size(pq: *const SeqHeapI32) -> c_int;

    /// Returns `true` if the priority queue contains no elements.
    pub fn seq_heap_i32_empty(pq: *const SeqHeapI32) -> bool;
}

/// Opaque handle to a `KNHeap<int64_t, int64_t>` instance.
#[repr(C)]
pub struct SeqHeapI64(c_void);

unsafe extern "C" {
    /// Allocates and constructs a new priority queue. Returns null on allocation failure.
    pub fn seq_heap_i64_new() -> *mut SeqHeapI64;

    /// Destroys and deallocates the priority queue.
    pub fn seq_heap_i64_free(pq: *mut SeqHeapI64);

    /// Pushes `key` onto the priority queue.
    /// # Safety
    /// `key` must not equal `i64::MIN` or `i64::MAX` (reserved as sentinels).
    pub fn seq_heap_i64_push(pq: *mut SeqHeapI64, key: i64);

    /// Removes and returns the minimum element. Undefined behaviour if the queue is empty.
    pub fn seq_heap_i64_pop(pq: *mut SeqHeapI64) -> i64;

    /// Returns the minimum element without removing it. Undefined behaviour if the queue is empty.
    pub fn seq_heap_i64_top(pq: *mut SeqHeapI64) -> i64;

    /// Returns the number of elements in the priority queue.
    pub fn seq_heap_i64_size(pq: *const SeqHeapI64) -> c_int;

    /// Returns `true` if the priority queue contains no elements.
    pub fn seq_heap_i64_empty(pq: *const SeqHeapI64) -> bool;
}

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
