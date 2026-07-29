use crate::buckets::{Block, Bucket};
use std::{collections::LinkedList, fmt::Debug, ptr::NonNull};

pub struct ListBlockBucket<T, const K: usize> {
    data: LinkedList<Block<T, K>>,
    should_cap: usize,
    total_size: usize,
    buff: Vec<T>,
    idx_map: Vec<NonNull<Block<T, K>>>,
}

impl<T: Copy + Default + Ord + Debug, const K: usize> Bucket<T> for ListBlockBucket<T, K> {
    fn default() -> Self {
        Self {
            data: LinkedList::from([]),
            should_cap: 0,
            total_size: 0,
            buff: vec![],
            idx_map: vec![],
        }
    }

    fn len(&self) -> usize {
        self.total_size
    }

    fn write_buffer(&mut self) -> *mut T {
        self.buff.as_mut_ptr()
    }

    fn clear(&mut self) {
        self.data.clear();
        self.total_size = 0;
        self.buff.clear();
    }

    fn is_empty(&self) -> bool {
        self.total_size == 0
    }

    fn capacity(&self) -> usize {
        self.data.len() * K
    }

    fn push(&mut self, elem: T) {
        if self.total_size % K == 0 {
            let mut b = Block::<T, K>::default();
            b.push(elem);
            self.data.push_back(b);
            let ptr = NonNull::from(self.data.back_mut().unwrap());
            self.idx_map.push(ptr);
            self.total_size += 1;
            return;
        }

        let block = self
            .data
            .back_mut()
            .expect("List has no blocks but size > 0");

        assert!(block.size() < K);
        block.push(elem);
        self.total_size += 1;
    }

    fn pop(&mut self) -> Option<T> {
        if self.total_size == 0 {
            return None;
        }

        let back = self
            .data
            .back_mut()
            .expect("List has no blocks but size > 0.");

        let last_idx = (self.total_size - 1) % K;
        let r = back.remove(last_idx);
        self.total_size -= 1;

        if back.empty() {
            self.idx_map.pop();
            self.data.pop_back();
        }

        Some(r)
    }

    fn get(&self, i: usize) -> T {
        assert!(i < self.total_size);

        let block_idx = i / K;
        assert!(block_idx < self.idx_map.len());
        let block_ptr = self.idx_map[block_idx];
        unsafe {
            let block = block_ptr.read();
            assert!(!block.empty());
            block.get(i % K)
        }
    }

    fn flush(&mut self, idx: usize) {
        unsafe {
            self.buff.set_len(idx);
        }

        self.idx_map.clear();
        self.data.clear();
        self.total_size = 0;

        let mut elements_left: &[T] = self.buff.as_slice();
        while elements_left.len() > K {
            let (slice, rest) = elements_left.split_at(K);
            let block = Block::from_slice(slice);
            self.data.push_back(block);
            let ptr = NonNull::from(self.data.back_mut().unwrap());
            self.idx_map.push(ptr);
            elements_left = rest;
            self.total_size += K;
        }

        if elements_left.len() > 0 {
            let last_block = Block::<T, K>::from_slice(elements_left);
            self.total_size += last_block.size();
            self.data.push_back(last_block);
            let last_block_list = self.data.back_mut().unwrap();
            let ptr = NonNull::from(last_block_list);
            self.idx_map.push(ptr);
        }
    }

    fn print(&self) {
        print!("Size: {} ", self.total_size);
        for x in self.data.iter() {
            x.print();
        }
    }

    fn insert(&mut self, pos: usize, elem: T) {
        assert!(pos <= self.total_size);

        if self.total_size == 0 {
            assert!(self.data.is_empty());
            let mut b = Block::<T, K>::default();
            b.push(elem);
            self.total_size += 1;
            self.data.push_back(b);
            let ptr = NonNull::from(self.data.back_mut().unwrap());
            self.idx_map.push(ptr);
            return;
        }

        self.total_size += 1;
        let mut overflow: Option<T>;
        let first_block_idx = pos / K;

        unsafe {
            let first_block = self.idx_map[first_block_idx].as_mut();
            if !first_block.full() {
                first_block.insert(elem, pos % K);
                return;
            } else {
                overflow = Some(first_block.insert_with_overflow(elem, pos % K));
            }
        }

        for i in (first_block_idx + 1)..self.data.len() {
            unsafe {
                let block = self.idx_map[i].as_mut();

                if let Some(t) = overflow {
                    if !block.full() {
                        block.insert(t, 0);
                        return;
                    } else {
                        overflow = Some(block.insert_with_overflow(t, 0));
                        continue;
                    }
                }
            }
        }
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

    fn get_unchecked_single(&self, pos: usize) -> T {
        let block_idx = pos / K;
        assert!(false);
        unsafe { self.idx_map[block_idx].read().get_unchecked(pos % K) }
    }

    fn override_elem(&mut self, pos: usize, elem: T) {
        let block_idx = pos / K;
        unsafe {
            self.idx_map[block_idx].read().override_elem(pos % K, elem);
        }
    }

    fn insert_index(&self, elem: T) -> usize {
        let mut idx = 0;
        for block in self.data.iter() {
            idx += block.insert_index(elem);
        }

        idx
    }

    fn remove(&mut self, i: usize) -> T {
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
    }

    fn reserve(&mut self, n: usize) {
        let free_cap = self.should_cap - self.total_size;
        let diff = free_cap as i64 - n as i64;

        if diff < 0 {
            let pos_diff = -1 * diff;
            self.should_cap += pos_diff as usize;
        }

        self.buff.reserve(self.should_cap);
    }

    unsafe fn set_len(&mut self, _: usize) {
        // no op
    }

    unsafe fn get_unchecked(&self, from: usize, len: usize) -> Vec<T> {
        let mut r = Vec::<T>::with_capacity(len);
        let mut dst = r.as_mut_ptr();
        let first_block_idx = from / K;
        let mut idx = from % K;
        let mut elems_left = len;
        for i in first_block_idx..self.data.len() {
            unsafe {
                let block = self.idx_map[i].read();
                let take = elems_left.min(K - idx);
                let src = block.as_ptr().add(idx);
                std::ptr::copy_nonoverlapping(src, dst, take);

                dst = dst.add(take);

                elems_left -= take;
                idx = 0;

                if elems_left == 0 {
                    break;
                }
            }
        }

        unsafe {
            r.set_len(len);
        }

        r
    }

    fn sort_decreasing(&mut self) {
        // Flatten the data to vector
        let mut data = vec![];

        for block in self.data.iter() {
            data.append(&mut block.to_vec());
        }

        data.sort_unstable_by_key(|&x| std::cmp::Reverse(x));

        let src = data.as_ptr();
        let mut pos = 0;

        for block in self.data.iter_mut() {
            let take = block.size().min(data.len() - pos);
            let dst = block.as_mut_ptr();
            unsafe {
                std::ptr::copy_nonoverlapping(src.add(pos), dst, take);
            }
            pos += take;
        }
    }
}
