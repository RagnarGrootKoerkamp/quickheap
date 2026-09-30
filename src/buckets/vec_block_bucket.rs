use crate::buckets::{Block, Bucket, block_arena::BlockArena};
use std::fmt::Debug;

pub struct VecBlockBucket<T, const K: usize, const CAP: usize> {
    data: Vec<*mut Block<T, K, CAP>>,
    total_size: usize,

    write_idx: usize,
    read_idx: usize,
    read_block_idx: usize,
    free_arena: *mut BlockArena<T, K, CAP>,
}

impl<T: Copy + Default + Ord + Debug, const K: usize, const CAP: usize> VecBlockBucket<T, K, CAP> {
    fn push_block(&mut self) {
        unsafe {
            self.data.push((*self.free_arena).alloc());
        }
    }
}

impl<T: Copy + Default + Ord + Debug, const K: usize, const CAP: usize> Bucket<T, K, CAP>
    for VecBlockBucket<T, K, CAP>
{
    fn default(free_arena: *mut BlockArena<T, K, CAP>) -> Self {
        Self {
            data: vec![Box::into_raw(Box::new(Block::default()))],
            total_size: 0,
            write_idx: 0,
            read_idx: 0,
            read_block_idx: 0,
            free_arena,
        }
    }

    const BLOCK_SIZE: usize = K;
    const BLOCKED: bool = true;

    fn concat(&mut self, other: Self) {
        debug_assert!(!self.data.is_empty());
        debug_assert!(!other.data.is_empty());

        let last_idx_self = self.data.len() - 1;
        let last_idx_other = other.data.len() - 1;

        let last_block_self = self.data[last_idx_self];
        let last_block_other = other.data[last_idx_other];

        unsafe {
            if (*last_block_self).full() {
                self.data.extend(other.data);
                self.total_size += other.total_size;
                return;
            }

            if (*last_block_other).full() {
                self.data.extend(other.data);
                let new_last_idx = self.data.len() - 1;
                self.data.swap(last_idx_self, new_last_idx);
                self.total_size += other.total_size;
                return;
            }

            // Invariant: Both blocks are not full cnt_self and cnt_other are not 0
            let cnt_self = self.total_size % K;
            let cnt_other = other.total_size % K;

            if cnt_self + cnt_other <= K {
                // Merge in one block (last of other)
                let src = (*last_block_self).as_ptr();
                let dst = (*last_block_other).as_mut_ptr().add(cnt_other);
                std::ptr::copy_nonoverlapping(src, dst, cnt_self);
                self.data.pop();
                self.data.extend(other.data);
                (*last_block_other).set_len(cnt_self + cnt_other);
                (*self.free_arena).free(last_block_self);
            } else {
                // Keep both blocks and only fill up first block
                let copy_elems = K - cnt_self;

                let src = (*last_block_other).as_ptr().add(cnt_other - copy_elems);
                let dst = (*last_block_self).as_mut_ptr().add(cnt_self);

                std::ptr::copy_nonoverlapping(src, dst, copy_elems);
                (*last_block_self).set_len(K);
                (*last_block_other).set_len(cnt_other - copy_elems);

                self.data.extend(other.data);
            }

            self.total_size += other.total_size;
        }
    }

    #[inline]
    fn reset_iters(&mut self) {
        self.total_size = 0;
        self.write_idx = 0;
        self.read_idx = 0;
        self.read_block_idx = 0;
    }

    #[inline]
    fn active_write(&mut self) -> *mut T {
        if self.data.len() == self.write_idx {
            self.push_block();
        }

        let block = self.data[self.write_idx];
        unsafe { (*block).as_mut_ptr() }
    }

    #[inline(always)]
    fn next_read_block(&mut self) -> *const T {
        let b = self.data[self.read_idx];
        self.read_idx += 1;
        unsafe { (*b).as_ptr() }
    }

    #[inline(always)]
    fn get_next_unchecked(&mut self, n: usize) -> &[T] {
        let block = self.data[self.read_idx];
        unsafe {
            debug_assert!(self.read_block_idx + n <= K);
            let c = (*block).as_slice(self.read_block_idx, n);
            self.read_block_idx += n;

            if self.read_block_idx == K {
                self.read_block_idx = 0;
                self.read_idx += 1;
            }
            c
        }
    }

    #[inline(always)]
    fn write_next(&mut self) {
        self.total_size += K;
        unsafe {
            (*self.data[self.write_idx]).set_len(K);
        }
        self.write_idx += 1;
    }

    #[inline]
    fn clear(&mut self) {
        self.total_size = 0;
        self.reset_iters();

        for b in self.data.drain(..self.data.len()) {
            unsafe {
                (*self.free_arena).free(b);
            }
        }

        debug_assert!(self.data.len() == 0);
    }

    #[inline]
    fn get(&self, i: usize) -> T {
        debug_assert!(i < self.total_size);
        let block = i / K;
        let in_block = i % K;
        unsafe { (*self.data[block]).get(in_block) }
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.data.capacity() * K
    }

    #[inline]
    fn push(&mut self, elem: T) {
        if self.data.is_empty() {
            self.push_block();
        }

        let mut data_len = self.data.len();
        unsafe {
            if (*self.data[self.data.len() - 1]).full() {
                // Last block is already full
                self.push_block();
                // self.data.push(Box::into_raw(Box::new(Block::default())));
                data_len += 1
            }
            self.total_size += 1;
            (*self.data[data_len - 1]).push(elem);
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.total_size
    }

    #[inline]
    fn pop(&mut self) -> Option<T> {
        let total_size = self.total_size;
        if total_size == 0 {
            return None;
        }

        let data_len = self.data.len();
        unsafe {
            let idx = (total_size - 1) % K;
            let r = (*self.data[data_len - 1]).remove(idx);
            self.total_size -= 1;

            if self.total_size % K == 0 {
                let b = self.data.pop().unwrap();
                (*self.free_arena).free(b);
            }

            Some(r)
        }
    }

    #[inline]
    fn write_buffer(&mut self) -> *mut T {
        unimplemented!();
        // self.buff.as_mut_ptr()
    }

    #[inline]
    fn print(&self) {
        println!("# blocks: {}", self.data.len());
        for block in &self.data {
            unsafe {
                (**block).print();
            }
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.total_size == 0
    }

    #[inline]
    fn reserve(&mut self, n: usize) {
        unreachable!();
    }

    #[inline]
    fn remove(&mut self, i: usize) -> T {
        unimplemented!();

        /*
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
         */
    }

    #[inline]
    fn flush(&mut self, _: usize) {
        unreachable!();
    }

    #[inline]
    fn override_elem(&mut self, pos: usize, elem: T) {
        unsafe {
            (*self.data[pos / K]).override_elem(pos % K, elem);
        }
    }

    #[inline]
    fn as_chunks<const S: usize>(&self) -> (Vec<[T; S]>, Vec<T>) {
        unimplemented!();

        /*
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
         */
    }

    #[inline]
    fn sort_decreasing(&mut self) {
        debug_assert!(self.total_size <= K);
        unsafe {
            (*self.data[0]).sort_decreasing();
        }
    }

    #[inline]
    unsafe fn set_len(&mut self, _: usize) {
        unreachable!();
    }

    #[inline]
    unsafe fn get_unchecked(&self, _: usize, _: usize) -> &[T] {
        unreachable!();
    }

    #[inline]
    fn set_last_block_len(&mut self, len: usize) {
        assert!(self.data.len() > 0);

        unsafe {
            for b in self.data.drain(self.write_idx + 1..) {
                (*self.free_arena).free(b);
            }
            (*self.data[self.write_idx]).set_len(len);
            self.total_size += len;
        }
    }

    #[inline]
    fn get_unchecked_single(&self, idx: usize) -> T {
        unimplemented!();

        /*
        let block = idx / K;
        let in_block = idx % K;

        self.data[block].get_unchecked(in_block)
         */
    }

    fn insert(&mut self, pos: usize, elem: T) {
        debug_assert!(self.total_size < K);

        if self.data.is_empty() {
            self.push_block();
            self.push(elem);
            return;
        }

        unsafe {
            (*self.data[0]).insert(elem, pos);
            self.total_size += 1;
        }
    }

    fn insert_index(&self, elem: T) -> usize {
        debug_assert!(self.total_size < K);

        if self.total_size == 0 {
            return 0;
        }

        // unsafe { S::insert_index((*(*self.head).block()).as_slice(0, self.total_size), elem) }

        unsafe {
            (*self.data[0])
                .as_slice(0, self.total_size)
                .partition_point(|&x| x > elem)
        }
    }
}
