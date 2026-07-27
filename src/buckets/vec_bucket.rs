use crate::buckets::Bucket;
use std::fmt::Debug;

pub struct VecBucket<T> {
    data: Vec<T>,
    buff: Vec<T>,
}

impl<T: Copy + Default + Ord + Debug> Bucket<T> for VecBucket<T> {
    fn default() -> Self {
        Self {
            data: Vec::with_capacity(128),
            buff: Vec::with_capacity(128),
        }
    }

    fn push(&mut self, elem: T) {
        self.data.push(elem);
    }

    fn pop(&mut self) -> Option<T> {
        self.data.pop()
    }

    fn is_empty(&self) -> bool {
        self.data.len() == 0
    }

    fn reserve(&mut self, n: usize) {
        self.data.reserve(n);
        self.buff.reserve(self.data.capacity());
    }

    fn capacity(&self) -> usize {
        self.data.capacity()
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn insert(&mut self, pos: usize, elem: T) {
        self.data.insert(pos, elem);
    }

    fn as_chunks<const S: usize>(&self) -> (Vec<[T; S]>, Vec<T>) {
        let (chunks, remainder) = self.data.as_chunks::<S>();
        (chunks.to_vec(), remainder.to_vec())
    }

    fn get(&self, i: usize) -> T {
        self.data[i]
    }

    fn get_unchecked_single(&self, idx: usize) -> T {
        self.data[idx]
    }

    fn remove(&mut self, i: usize) -> T {
        self.data.swap_remove(i)
    }

    fn sort_decreasing(&mut self) {
        self.data.sort_unstable_by_key(|&x| std::cmp::Reverse(x));
    }

    fn insert_index(&self, elem: T) -> usize {
        self.data.partition_point(|&x| x > elem)
    }

    fn clear(&mut self) {
        self.data.clear();
        self.buff.clear();
    }

    unsafe fn set_len(&mut self, n: usize) {
        unsafe {
            self.data.set_len(n);
        }
    }

    unsafe fn get_unchecked(&self, from: usize, len: usize) -> Vec<T> {
        unsafe { self.data.get_unchecked(from..from + len).to_vec() }
    }

    fn write_buffer(&mut self) -> *mut T {
        self.buff.as_mut_ptr()
    }

    fn flush(&mut self, idx: usize) {
        debug_assert!(idx < self.buff.capacity());

        unsafe {
            self.buff.set_len(idx);
        }

        self.data.clear();
        self.data.append(&mut self.buff);
        self.buff.clear();
    }

    fn print(&self) {
        print!("{:?}\n", self.data);
    }

    fn override_elem(&mut self, pos: usize, elem: T) {
        self.data[pos] = elem;
    }
}
