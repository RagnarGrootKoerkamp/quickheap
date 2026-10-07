use crate::buckets::{Block, Bucket, block_arena::BlockArena};
use std::{fmt::Debug, ptr};

pub struct ListBlockBucket<T: Default + Copy + Debug + PartialOrd, const K: usize, const CAP: usize>
{
    head: *mut Block<T, K, CAP>,
    tail: *mut Block<T, K, CAP>,

    total_size: usize,
    current_write: *mut Block<T, K, CAP>,
    current_read: *mut Block<T, K, CAP>,

    current_read_idx: usize,

    free_arena: *mut BlockArena<T, K, CAP>,
}

impl<T: Copy + Default + Ord + Debug, const K: usize, const CAP: usize> ListBlockBucket<T, K, CAP> {
    fn push_new_block(&mut self) {
        unsafe {
            let block_ptr = (*self.free_arena).alloc();

            if self.total_size == 0 {
                self.tail = block_ptr;
                self.head = block_ptr;
                self.current_write = self.head;
                self.current_read = self.head;
            } else {
                (*block_ptr).set_prev(self.tail);
                (*self.tail).set_next(block_ptr);
                self.tail = (*self.tail).next();
            }
        }
    }
}

impl<T: Copy + Default + Ord + Debug, const K: usize, const CAP: usize> Bucket<T, K, CAP>
    for ListBlockBucket<T, K, CAP>
{
    const BLOCK_SIZE: usize = K;
    const BLOCKED: bool = true;

    fn default(free_arena: *mut BlockArena<T, K, CAP>) -> Self {
        Self {
            head: ptr::null_mut(),
            tail: ptr::null_mut(),
            total_size: 0,

            current_read: Default::default(),
            current_write: Default::default(),
            current_read_idx: 0,
            free_arena,
        }
    }

    fn min(&mut self) -> (T, usize) {
        debug_assert!(self.total_size > 0);

        let mut i = 1;
        self.current_read = self.head;
        let mut curr = self.next_read_block();

        unsafe {
            let mut min: T = *curr;
            let mut min_pos = 0;

            while i < self.total_size {
                if i % K == 0 {
                    curr = self.next_read_block();
                }

                if *(curr.add(i % K)) < min {
                    min = *(curr.add(i % K));
                    min_pos = i;
                }

                i += 1;
            }

            (min, min_pos)
        }
    }

    fn max(&mut self) -> (T, usize) {
        let mut i = 1;
        self.current_read = self.head;
        let mut curr = self.next_read_block();

        unsafe {
            let mut max: T = *curr;
            let mut max_pos = 0;

            while i < self.total_size {
                if i % K == 0 {
                    curr = self.next_read_block();
                }

                if *(curr.add(i % K)) > max {
                    max = *(curr.add(i % K));
                    max_pos = i;
                }

                i += 1;
            }

            (max, max_pos)
        }
    }

    fn concat(&mut self, other: Self) {
        debug_assert!(!other.head.is_null());
        // debug_assert!(!self.tail.is_null());
        if self.tail.is_null() {
            // Concat to empty bucket
            self.total_size = other.total_size;
            self.head = other.head;
            self.tail = other.tail;
            return;
        }

        let last_block_self = self.tail;
        let last_block_other = other.tail;

        unsafe {
            assert!(!(*last_block_self).empty());
            assert!(!(*last_block_other).empty());
        }

        unsafe {
            if (*last_block_self).full() {
                self.tail = other.tail;
                (*last_block_self).set_next(other.head);
                (*other.head).set_prev(last_block_self);
                self.total_size += other.total_size;
                return;
            }

            let prev_self = (*self.tail).prev();
            if (*last_block_other).full() {
                if !prev_self.is_null() {
                    (*prev_self).set_next(other.head);
                } else {
                    self.head = other.head;
                }

                (*last_block_self).set_prev(last_block_other);
                (*last_block_other).set_next(last_block_self);
                (*other.head).set_prev(prev_self);

                self.total_size += other.total_size;
                return;
            }

            // Invariant: Both blocks are not full cnt_self and cnt_other are not 0
            let cnt_self = self.total_size % K;
            let cnt_other = other.total_size % K;

            debug_assert!(cnt_self > 0);
            debug_assert!(cnt_other > 0);

            if cnt_self + cnt_other <= K {
                // Merge in one block (last of other)
                let src = (*last_block_self).as_ptr();
                let dst = (*last_block_other).as_mut_ptr().add(cnt_other);
                std::ptr::copy_nonoverlapping(src, dst, cnt_self);

                if !prev_self.is_null() {
                    (*prev_self).set_next(other.head);
                } else {
                    self.head = other.head;
                }

                (*other.head).set_prev(prev_self);
                self.tail = other.tail;
                debug_assert!(cnt_self + cnt_other > 0);
                (*self.tail).set_len(cnt_self + cnt_other);
                (*self.free_arena).free(last_block_self);
            } else {
                // Keep both blocks and only fill up first block
                let copy_elems = K - cnt_self;

                let src = (*last_block_other).as_ptr().add(cnt_other - copy_elems);
                let dst = (*last_block_self).as_mut_ptr().add(cnt_self);

                std::ptr::copy_nonoverlapping(src, dst, copy_elems);
                (*last_block_self).set_len(K);
                (*last_block_other).set_len(cnt_other - copy_elems);

                assert!(copy_elems > 0);
                assert!(cnt_other > copy_elems);
                assert!(cnt_other - copy_elems > 0);

                (*last_block_self).set_next(other.head);
                (*other.head).set_prev(self.tail);
                self.tail = other.tail;
            }

            self.total_size += other.total_size;
        }
    }

    #[inline]
    fn reset_iters(&mut self) {
        debug_assert!(self.total_size != 0);
        debug_assert!(!self.head.is_null());
        self.current_read = self.head;
        self.current_write = self.head;
        self.current_read_idx = 0;
        self.total_size = 0;
    }

    #[inline]
    fn active_write(&mut self) -> *mut T {
        if self.head.is_null() {
            // Empty list
            self.push_new_block();
        }

        unsafe { (*self.current_write).as_mut_ptr() }
    }

    #[inline]
    fn next_read_block(&mut self) -> *const T {
        unsafe {
            let node = self.current_read;
            assert!(!node.is_null());
            self.current_read = (*node).next();
            (*node).as_ptr()
        }
    }

    #[inline]
    fn write_next(&mut self) {
        self.total_size += K;
        unsafe {
            (*self.current_write).set_len(K);

            if ((*self.current_write).next()).is_null() {
                self.push_new_block();
            }

            self.current_write = (*self.current_write).next()
        }
    }

    #[inline]
    fn get_next_unchecked(&mut self, n: usize) -> &[T] {
        debug_assert!(!self.current_read.is_null());
        unsafe {
            let c = (*self.current_read).as_slice(self.current_read_idx, n);
            self.current_read_idx += n;

            if self.current_read_idx == K {
                self.current_read = (*self.current_read).next();
                self.current_read_idx = 0;
            }

            c
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.total_size
    }

    #[inline]
    fn write_buffer(&mut self) -> *mut T {
        unreachable!();
    }

    #[inline]
    fn clear(&mut self) {
        let mut curr = self.head;
        unsafe {
            while !curr.is_null() {
                let node = curr;
                curr = (*curr).next();
                (*self.free_arena).free(node);
            }
        }

        self.current_read = Default::default();
        self.current_write = Default::default();

        self.head = ptr::null_mut();
        self.tail = ptr::null_mut();
        self.total_size = 0;
        self.current_read_idx = 0;
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.total_size == 0
    }

    #[inline]
    fn capacity(&self) -> usize {
        let mut blocks = self.total_size as f64 / K as f64;
        blocks = blocks.ceil();
        blocks as usize * K
    }

    #[inline]
    fn push(&mut self, elem: T) {
        if self.total_size % K == 0 {
            self.push_new_block();
        }
        unsafe {
            (*self.tail).push(elem);
        }
        self.total_size += 1;
    }

    #[inline]
    fn pop(&mut self) -> Option<T> {
        if self.total_size == 0 {
            return None;
        }

        let last_idx = (self.total_size - 1) % K;
        unsafe {
            let r = (*self.tail).remove(last_idx);
            self.total_size -= 1;
            if (*self.tail).empty() {
                let free_node = self.tail;
                self.tail = (*free_node).prev();

                if !self.tail.is_null() {
                    (*self.tail).set_next(ptr::null_mut());
                }

                (*self.free_arena).free(free_node);
            }

            if self.total_size == 0 {
                self.head = ptr::null_mut();
            }

            Some(r)
        }
    }

    #[inline]
    fn get(&self, i: usize) -> T {
        debug_assert!(i < self.total_size);
        let mut curr_pos = i;
        let mut curr = self.head;

        while curr_pos >= K {
            unsafe {
                debug_assert!(!curr.is_null());
                curr = (*curr).next();
            }
            curr_pos -= K;
        }

        debug_assert!(i % K == curr_pos);

        unsafe {
            debug_assert!(!curr.is_null());
            (*curr).get(curr_pos)
        }
    }

    fn flush(&mut self, _: usize) {
        unreachable!();
    }

    fn print(&self) {
        print!("Size: {} ", self.total_size);

        let mut curr = self.head;
        while !curr.is_null() {
            unsafe {
                (*curr).print();
                curr = (*curr).next();
            }
        }
        println!(".. end");
    }

    #[inline]
    fn insert(&mut self, pos: usize, elem: T) {
        debug_assert!(self.total_size < K); // Only insert to the smallest block (at most N elements, N <= K)

        if self.total_size == 0 {
            debug_assert!(pos == 0);
            self.push(elem);
            return;
        }

        unsafe {
            (*self.head).insert(elem, pos);
            self.total_size += 1;
        }
    }

    fn as_chunks<const S: usize>(&self) -> (Vec<[T; S]>, Vec<T>) {
        unimplemented!();

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

    #[inline]
    fn get_unchecked_single(&self, pos: usize) -> T {
        unimplemented!();

        /*
        let block_idx = pos / K;
        assert!(false);
        unsafe { self.idx_map[block_idx].as_ref().get_unchecked(pos % K) }
         */
    }

    #[inline]
    fn override_elem(&mut self, pos: usize, elem: T) {
        unimplemented!();

        /*
        let block_idx = pos / K;
        unsafe {
            self.idx_map[block_idx]
                .as_mut()
                .override_elem(pos % K, elem);
        }
         */
    }

    #[inline]
    fn insert_index(&self, elem: T) -> usize {
        debug_assert!(self.total_size < K); // Only insert to the smallest block (at most N elements, N <= K)

        if self.total_size == 0 {
            return 0;
        }

        // unsafe { S::insert_index((*(*self.head).block()).as_slice(0, self.total_size), elem) }

        unsafe {
            (*self.head)
                .as_slice(0, self.total_size)
                .partition_point(|&x| x > elem)
        }
    }

    #[inline]
    fn remove(&mut self, i: usize) -> T {
        unimplemented!();

        /*
        assert!(i < self.total_size);
        let need_swap = (i / K) != (self.total_size / K);
        let last_elem = if need_swap {
            let last_block = self.data.back_mut().expect("total_size > 0 but no blocks");
            Some(last_block.remove(self.total_size % K))
        } else {
            None
        };

        let mut c = self.data.cursor_front_mut();
        let mut idx = i;

        // TODO: Does not work for sorted buckets
        // (If K < smallest partition size)

        let r;
        while let Some(block) = c.current() {
            if idx >= K {
                idx -= K;
                c.move_next();
                continue;
            }

            r = block.remove(idx);
            self.total_size -= 1;

            if block.empty() {
                self.idx_map.pop();
                self.data.pop_back();
                return r;
            }

            // Swap an element from the back
            if need_swap {
                block.override_elem(
                    i % K,
                    last_elem.expect("Needs swap is true but no element to swap."),
                );
            }
            break;
        }

        assert!(false);
        T::default()
         */
    }

    fn reserve(&mut self, n: usize) {
        unreachable!();
    }

    unsafe fn set_len(&mut self, _: usize) {
        unreachable!();
    }

    unsafe fn get_unchecked(&self, _: usize, _: usize) -> &[T] {
        unreachable!();
    }

    fn set_last_block_len(&mut self, len: usize) {
        assert!(!self.current_write.is_null());
        self.total_size += len;
        unsafe {
            (*self.current_write).set_len(len);
            self.tail = self.current_write;

            let mut after_write = (*self.current_write).next();
            (*self.current_write).set_next(ptr::null_mut());

            let mut curr;
            while !after_write.is_null() {
                curr = after_write;
                after_write = (*after_write).next();

                (*curr).reset();
                (*self.free_arena).free(curr);
            }
        }
    }

    #[inline]
    fn sort_decreasing(&mut self) {
        debug_assert!(self.total_size <= K);
        debug_assert!(self.total_size > 0);
        unsafe {
            (*self.head).sort_decreasing();
        }
    }
}
