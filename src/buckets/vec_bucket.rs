use crate::buckets::{Bucket, block_arena::BlockArena};
use std::fmt::Debug;

pub struct VecBucket<T, const K: usize, const CAP: usize> {
    data: Vec<T>,
    buff: Vec<T>,
}

impl<T: Copy + Default + Ord + Debug, const K: usize, const CAP: usize> Bucket<T, K, CAP>
    for VecBucket<T, K, CAP>
{
    fn default(_: *mut BlockArena<T, K, CAP>) -> Self {
        Self {
            data: Vec::with_capacity(128),
            buff: Vec::with_capacity(128),
        }
    }

    fn active_write(&mut self) -> *mut T {
        unreachable!();
    }

    fn get_next_unchecked(&mut self, n: usize) -> &[T] {
        unreachable!();
    }

    fn reset_iters(&mut self) {
        unreachable!();
    }

    fn write_next(&mut self) {
        unreachable!();
    }

    fn next_read_block(&mut self) -> *const T {
        unreachable!();
    }

    #[inline]
    fn push(&mut self, elem: T) {
        self.data.push(elem);
    }

    #[inline]
    fn pop(&mut self) -> Option<T> {
        debug_assert!(self.data.len() > 0);
        self.data.pop()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.data.len() == 0
    }

    #[inline]
    fn reserve(&mut self, n: usize) {
        self.data.reserve(n);
        self.buff.reserve(self.data.capacity());
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.data.capacity()
    }

    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn insert(&mut self, pos: usize, elem: T) {
        self.data.insert(pos, elem);
    }

    #[inline]
    fn as_chunks<const S: usize>(&self) -> (Vec<[T; S]>, Vec<T>) {
        let (chunks, remainder) = self.data.as_chunks::<S>();
        (chunks.to_vec(), remainder.to_vec())
    }

    #[inline]
    fn get(&self, i: usize) -> T {
        self.data[i]
    }

    #[inline]
    fn get_unchecked_single(&self, idx: usize) -> T {
        self.data[idx]
    }

    #[inline]
    fn remove(&mut self, i: usize) -> T {
        self.data.swap_remove(i)
    }

    #[inline]
    fn sort_decreasing(&mut self) {
        self.data.sort_unstable_by_key(|&x| std::cmp::Reverse(x));
    }

    #[inline]
    fn insert_index(&self, elem: T) -> usize {
        self.data.partition_point(|&x| x > elem)
    }

    #[inline]
    fn clear(&mut self) {
        self.data.clear();
        self.buff.clear();
    }

    #[inline]
    unsafe fn set_len(&mut self, n: usize) {
        unsafe {
            self.data.set_len(n);
        }
    }

    #[inline]
    unsafe fn get_unchecked(&self, from: usize, len: usize) -> &[T] {
        unsafe { self.data.get_unchecked(from..from + len) }
    }

    #[inline]
    fn write_buffer(&mut self) -> *mut T {
        self.buff.as_mut_ptr()
    }

    #[inline]
    fn flush(&mut self, idx: usize) {
        debug_assert!(idx < self.buff.capacity());

        unsafe {
            self.buff.set_len(idx);
        }

        std::mem::swap(&mut self.data, &mut self.buff);
        self.buff.clear();
    }

    fn set_last_block_len(&mut self, len: usize) {}

    #[inline]
    fn print(&self) {
        print!("{:?}\n", self.data);
    }

    #[inline]
    fn override_elem(&mut self, pos: usize, elem: T) {
        self.data[pos] = elem;
    }
}
