use crate::buckets::{Bucket, FlatBucket, block_arena::BlockArena};
use std::fmt::Debug;

pub struct VecBucket<T, const K: usize, const CAP: usize> {
    data: Vec<T>,
}

impl<T: Copy + Default + Ord + Debug, const K: usize, const CAP: usize> FlatBucket<T>
    for VecBucket<T, K, CAP>
{
    #[inline]
    fn reserve(&mut self, n: usize) {
        self.data.reserve(n);
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
    fn write_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
}

impl<T: Copy + Default + Ord + Debug, const K: usize, const CAP: usize> Bucket<T, K, CAP>
    for VecBucket<T, K, CAP>
{
    fn default(_: *mut BlockArena<T, K, CAP>) -> Self {
        Self {
            data: Vec::with_capacity(128),
        }
    }

    fn min(&mut self) -> (T, usize) {
        let min_pos = self
            .data
            .iter()
            .enumerate()
            .min_by_key(|&(_, x)| x)
            .map(|(i, _)| i)
            .unwrap();

        (self.data[min_pos], min_pos)
    }

    fn max(&mut self) -> (T, usize) {
        let max_pos = self
            .data
            .iter()
            .enumerate()
            .max_by_key(|&(_, x)| x)
            .map(|(i, _)| i)
            .unwrap();

        (self.data[max_pos], max_pos)
    }

    fn recompute_min_max(&mut self) {}

    #[inline]
    fn push(&mut self, elem: T) {
        self.data.push(elem);
    }

    #[inline]
    fn concat(&mut self, other: Self) {
        self.data.extend(other.data);
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
    fn get(&self, i: usize) -> T {
        self.data[i]
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
    }

    #[inline]
    fn print(&self) {
        print!("{:?}\n", self.data);
    }
}
