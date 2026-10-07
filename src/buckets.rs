use crate::buckets::block_arena::BlockArena;

pub mod block;
pub mod block_arena;
pub mod equal_buckets;
pub mod list_block_bucket;
pub mod partitioning;
pub mod vec_block_bucket;
pub mod vec_bucket;

pub trait FlatBucket<T: PartialEq> {
    unsafe fn set_len(&mut self, n: usize);
    unsafe fn get_unchecked(&self, from: usize, len: usize) -> &[T];
    fn write_ptr(&mut self) -> *mut T;
    fn reserve(&mut self, n: usize);
    // fn override_elem(&mut self, pos: usize, elem: T);
    // fn get_unchecked_single(&self, idx: usize) -> T;
}

pub trait BlockedBucket<T: PartialEq, const K: usize, const CAP: usize> {
    fn reset_iters(&mut self);
    fn active_write(&mut self) -> *mut T;
    fn write_next(&mut self);
    fn set_last_block_len(&mut self, len: usize);
    fn next_read_block(&mut self) -> *const T;
    fn get_next_unchecked(&mut self, n: usize) -> &[T];
}

pub trait Bucket<T: PartialEq, const K: usize, const CAP: usize> {
    fn concat(&mut self, other: Self);
    fn print(&self);
    fn default(free_arena: *mut BlockArena<T, K, CAP>) -> Self;
    fn push(&mut self, elem: T);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn remove(&mut self, i: usize) -> T;
    fn insert(&mut self, pos: usize, elem: T);
    // fn as_chunks<const S: usize>(&self) -> (Vec<[T; S]>, Vec<T>);
    fn get(&self, i: usize) -> T;
    fn pop(&mut self) -> Option<T>;
    fn sort_decreasing(&mut self);
    fn insert_index(&self, elem: T) -> usize;
    fn capacity(&self) -> usize;
    fn clear(&mut self);

    fn min(&mut self) -> (T, usize);
    fn max(&mut self) -> (T, usize);
    fn recompute_min_max(&mut self);
}
