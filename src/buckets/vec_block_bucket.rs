use crate::buckets::{Block, Bucket};
use std::fmt::Debug;

pub struct VecBlockBucket<T, const K: usize> {
    data: Vec<Block<T, K>>,
    total_size: usize,
    buff: Vec<T>,
}

impl<T: Copy + Default + Ord + Debug, const K: usize> Bucket<T> for VecBlockBucket<T, K> {
    fn default() -> Self {
        Self {
            data: Vec::with_capacity(128),
            total_size: 0,
            buff: Vec::with_capacity(128),
        }
    }

    fn clear(&mut self) {
        self.total_size = 0;
        self.buff.clear();
        self.data.clear();
    }

    fn get(&self, i: usize) -> T {
        let block = i / K;
        let in_block = i % K;
        self.data[block].get(in_block)
    }

    fn capacity(&self) -> usize {
        self.data.capacity() * K
    }

    fn push(&mut self, elem: T) {
        let mut data_len = self.data.len();
        if self.data[data_len - 1].full() {
            // Last block is already full
            self.data.push(Block::default());
            data_len += 1
        }

        self.total_size += 1;
        self.data[data_len - 1].push(elem);
    }

    fn len(&self) -> usize {
        self.total_size
    }

    fn pop(&mut self) -> Option<T> {
        let total_size = self.total_size;
        if total_size == 0 {
            return None;
        }

        let data_len = self.data.len();
        let r = self.data[data_len - 1].remove((total_size % K) - 1);
        if self.total_size % K == 0 {
            self.data.pop();
        }
        self.total_size -= 1;

        Some(r)
    }

    fn write_buffer(&mut self) -> *mut T {
        self.buff.as_mut_ptr()
    }

    fn print(&self) {
        for block in &self.data {
            block.print();
        }
    }

    fn is_empty(&self) -> bool {
        self.total_size == 0
    }

    fn reserve(&mut self, n: usize) {
        self.data.reserve((n + K - 1) / K);
        let max = self.data.capacity() * K;
        self.buff.reserve(max);
    }

    fn remove(&mut self, i: usize) -> T {
        let block = i / K;
        let in_block = i % K;

        let last_block = self.data.len() - 1;

        // TODO: Does not work for sorted buckets
        // (If K < smallest partition size)

        let r: T;

        if block != last_block {
            let last_idx = self.total_size % K;
            let inter = self.data[last_block].remove(last_idx);
            r = self.data[block].get(in_block);
            self.data[block].override_elem(in_block, inter);
            assert!(self.data[block].size() == K);
        } else {
            r = self.data[last_block].remove(i % K);
        }

        self.total_size -= 1;
        if self.total_size % K == 0 {
            assert!(self.data[last_block].size() == 0);
            self.data.pop();
        }

        r
    }

    fn flush(&mut self, idx: usize) {
        unsafe {
            self.buff.set_len(idx);
        }

        self.data.clear();
        self.total_size = 0;

        let mut elements_left: &[T] = self.buff.as_slice();
        while elements_left.len() > K {
            let (slice, rest) = elements_left.split_at(K);
            let block = Block::from_slice(slice);
            self.data.push(block);
            elements_left = rest;
            self.total_size += K;
        }

        if elements_left.len() > 0 {
            let last_block = Block::<T, K>::from_slice(elements_left);
            self.total_size += last_block.size();
            self.data.push(last_block);
        }
    }

    fn override_elem(&mut self, pos: usize, elem: T) {
        self.data[pos / K].override_elem(pos % K, elem);
    }

    fn as_chunks<const S: usize>(&self) -> (Vec<[T; S]>, Vec<T>) {
        let mut idx = 0;
        let mut result: Vec<[T; S]> = vec![];
        let mut curr_array = [T::default(); S];
        while idx < self.total_size {
            curr_array[idx % S] = self.get_unchecked_single(idx);
            idx += 1;
            if idx % S == S - 1 {
                result.push(curr_array);
            }
        }

        let remainder = curr_array[..self.total_size % K].to_vec();
        (result, remainder)
    }

    fn sort_decreasing(&mut self) {
        // Flatten to vec
        let mut data = vec![];
        for i in 0..self.total_size {
            data.push(self.get(i));
        }

        data.sort_unstable_by_key(|&x| std::cmp::Reverse(x));

        // TODO: Do something faster..
        for i in 0..self.total_size {
            self.override_elem(i, data[i]);
        }
    }

    unsafe fn set_len(&mut self, n: usize) {
        unsafe {
            self.data.set_len((n + K - 1) / K);
        }
    }

    unsafe fn get_unchecked(&self, from: usize, len: usize) -> Vec<T> {
        let mut result: Vec<T> = vec![];
        let mut idx = from;

        while idx < from + len {
            result.push(self.get_unchecked_single(idx));
            idx += 1;
        }

        result
    }

    fn get_unchecked_single(&self, idx: usize) -> T {
        let block = idx / K;
        let in_block = idx % K;

        self.data[block].get_unchecked(in_block)
    }

    fn insert(&mut self, pos: usize, elem: T) {
        let block = pos / K;
        let mut elem_to_insert = elem;
        let mut curr_block = block;
        let mut curr_pos = pos;

        if self.data.len() == 0 {
            let mut b = Block::<T, K>::default();
            b.push(elem);
            self.data.push(b);
            self.total_size += 1;
            return;
        }

        if self.total_size % K == (K - 1) {
            self.data.push(Block::<T, K>::default());
        }

        while curr_block < self.data.len() - 1 {
            elem_to_insert =
                self.data[curr_block].insert_with_overflow(elem_to_insert, curr_pos % K);
            curr_block += 1;
            curr_pos = (curr_block / K) * K;
        }

        self.total_size += 1;
        self.data[curr_block].insert(elem, curr_pos % K);
    }

    fn insert_index(&self, elem: T) -> usize {
        let mut idx = 0;

        for i in 0..self.total_size {
            let block = i / K;
            let in_block = i % K;

            if self.data[block].get(in_block) > elem {
                idx += 1;
            }
        }

        idx
    }
}
