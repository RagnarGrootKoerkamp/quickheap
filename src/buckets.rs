use std::fmt::Debug;

pub mod list_block_bucket;
pub mod vec_block_bucket;
pub mod vec_bucket;

pub mod equal_buckets;

#[derive(Clone, Copy)]
// #[repr(align(32))]
pub struct Block<T, const K: usize> {
    size: usize,
    data: [T; K],
}

impl<T: Default + Copy + Debug + PartialOrd, const K: usize> Block<T, K> {
    fn default() -> Self {
        Self {
            size: 0,
            data: [T::default(); K],
        }
    }

    fn empty(&self) -> bool {
        self.size == 0
    }

    fn from_slice(slice: &[T]) -> Self {
        let mut data = [T::default(); K];
        let size;

        if slice.len() == K {
            data = slice
                .try_into()
                .expect("Something went wrong converting the slice into an array.");
            size = K;
        } else {
            data[..slice.len()].copy_from_slice(slice);
            size = slice.len();
        }

        Self { data, size }
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    fn push(&mut self, elem: T) {
        assert!(self.size < K);
        self.data[self.size] = elem;
        self.size += 1;
    }

    fn insert(&mut self, elem: T, pos: usize) {
        assert!(self.size < K);

        let mut idx = pos;
        let mut old;
        let mut new = elem;

        self.size += 1;

        while idx < self.size {
            old = self.data[idx];
            self.data[idx] = new;
            new = old;
            idx += 1;
        }
    }

    fn insert_index(&self, elem: T) -> usize {
        let mut idx = 0;

        for i in 0..self.size {
            if elem < self.data[i] {
                idx += 1;
            }
        }

        idx
    }

    fn to_vec(&self) -> Vec<T> {
        self.data[0..self.size].to_vec()
    }

    fn insert_with_overflow(&mut self, elem: T, pos: usize) -> T {
        assert!(pos < K);
        assert!(self.size == K);

        let mut idx = pos;
        let mut old;
        let mut new = elem;

        let r = self.data[K - 1];

        while idx < self.size {
            old = self.data[idx];
            self.data[idx] = new;
            new = old;
            idx += 1;
        }

        r
    }

    fn remove(&mut self, i: usize) -> T {
        assert!(i < K);
        assert!(self.size > 0);

        let elem = self.data[i];
        self.data[i] = self.data[self.size - 1];
        self.size -= 1;

        elem
    }

    fn get(&self, i: usize) -> T {
        if i >= self.size {
            println!("PROBLEM {} {}", i, self.size);
        }
        assert!(self.size <= K);
        assert!(i < self.size);
        self.data[i]
    }

    fn get_unchecked(&self, i: usize) -> T {
        assert!(self.size <= K);
        self.data[i]
    }

    fn size(&self) -> usize {
        self.size
    }

    fn full(&self) -> bool {
        self.size >= K
    }

    fn print(&self) {
        let data = self.data.to_vec();
        println!("{:?}", &data[..self.size]);
    }

    fn override_elem(&mut self, pos: usize, val: T) {
        self.data[pos] = val;
    }
}

pub trait Bucket<T> {
    fn default() -> Self;
    fn push(&mut self, elem: T);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn reserve(&mut self, n: usize);
    fn remove(&mut self, i: usize) -> T;
    fn insert(&mut self, pos: usize, elem: T);
    fn as_chunks<const S: usize>(&self) -> (Vec<[T; S]>, Vec<T>);
    fn get(&self, i: usize) -> T;
    fn pop(&mut self) -> Option<T>;
    fn sort_decreasing(&mut self);
    fn insert_index(&self, elem: T) -> usize;
    fn capacity(&self) -> usize;
    fn clear(&mut self);
    fn override_elem(&mut self, pos: usize, elem: T);
    unsafe fn set_len(&mut self, n: usize);
    unsafe fn get_unchecked(&self, from: usize, len: usize) -> Vec<T>;
    fn write_buffer(&mut self) -> *mut T;
    fn flush(&mut self, idx: usize);
    fn print(&self);
    fn get_unchecked_single(&self, idx: usize) -> T;
}

#[cfg(test)]
mod tests {
    use crate::{
        Avx2, ConfigurableSimdQuickHeap, pivot_strategies::MedianOfM,
        rebalancing_strategies::NoRebalancing,
    };

    use super::*;

    #[test]
    fn init_block() {
        let mut block = Block::<i32, 16>::default();
        assert!(block.size() == 0);

        for i in 0..16 {
            block.push(i);
        }

        for i in 0..16 {
            assert!(block.get(i) == i as i32)
        }
    }

    #[test]
    fn delete_from_block() {
        let mut block = Block::<i32, 16>::default();
        assert!(block.size() == 0);

        for i in 0..16 {
            block.push(i);
        }

        // 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
        block.remove(2);
        // 0 1 15 3 4 5 6 7 8 9 10 11 12 13 14
        block.remove(14);
        // 0 1 15 3 4 5 6 7 8 9 10 11 12 13
        block.remove(6);
        // 0 1 15 3 4 5 13 7 8 9 10 11 12

        assert!(block.size() == 13);
        assert!(block.get(0) == 0);
        assert!(block.get(2) == 15);
        assert!(block.get(6) == 13);
        assert!(block.get(7) == 7);
        assert!(block.get(12) == 12);
    }

    #[test]
    fn insert_into_bucket() {
        let mut bucket1 = vec_bucket::VecBucket::<i32>::default();
        // let mut bucket2 = VecBlockBucket::<i32, 16>::default();
        // let mut bucket3 = ListBlockBucket::<i32, 16>::default();

        // let mut bucket2_copy = VecBlockBucket::<i32, 16>::default();

        bucket1.push(12);
        // bucket2.insert(12);
        // bucket3.insert(12);
        bucket1.push(2);
        // bucket2.insert(2);
        // bucket3.insert(2);

        // bucket2_copy.insert(2);
        // bucket2_copy.insert(2);

        assert!(bucket1.len() == 2);
        // assert!(bucket2.size() == 2);
        // assert!(bucket3.size() == 2);

        // bucket2.join(bucket2_copy);
        // assert!(bucket2.size() == 4);
    }

    #[test]
    fn test_partition_vec_bucket() {
        let mut h = ConfigurableSimdQuickHeap::<
            i32,
            vec_bucket::VecBucket<i32>,
            Avx2,
            MedianOfM<3>,
            NoRebalancing,
        >::default();

        for i in 0..10000 {
            h.push(i);
        }

        for i in 0..10000 {
            let r = h.pop().unwrap();
            assert!(r == i);
        }
    }

    #[test]
    fn test_partition_vec_block_bucket() {
        let mut h = ConfigurableSimdQuickHeap::<
            i32,
            vec_block_bucket::VecBlockBucket<i32, 128>,
            Avx2,
            MedianOfM<3>,
            NoRebalancing,
        >::default();

        for i in 0..10000 {
            h.push(i);
        }

        for i in 0..10000 {
            let r = h.pop().unwrap();
            assert!(r == i);
        }
    }

    #[test]
    fn test_partition_list_block_bucket() {
        let mut h = ConfigurableSimdQuickHeap::<
            i32,
            list_block_bucket::ListBlockBucket<i32, 128>,
            Avx2,
            MedianOfM<3>,
            NoRebalancing,
        >::default();

        for i in 0..10000 {
            h.push(i);
        }

        for i in 0..10000 {
            let r = h.pop().unwrap();
            assert!(r == i);
        }
    }

    #[test]
    fn pen_test_vec_block_bucket() {
        let mut h = ConfigurableSimdQuickHeap::<
            u64,
            vec_block_bucket::VecBlockBucket<u64, 128>,
            Avx2,
            MedianOfM<3>,
            NoRebalancing,
        >::default();

        let mut rng = fastrand::Rng::new();
        let n = 10000;
        let mut values = std::iter::repeat_with(|| rng.u64(0..1000000000))
            .take(10 * n as usize)
            .collect::<Vec<_>>()
            .into_iter();

        for _ in 0..n {
            h.push(values.next().unwrap());
            let _l = h.pop().unwrap();
        }
    }

    #[test]
    fn pen_test_list_block_bucket() {
        let mut h = ConfigurableSimdQuickHeap::<
            u64,
            list_block_bucket::ListBlockBucket<u64, 128>,
            Avx2,
            MedianOfM<3>,
            NoRebalancing,
        >::default();

        let mut rng = fastrand::Rng::new();
        let n = 10000;
        let mut values = std::iter::repeat_with(|| rng.u64(0..1000000000))
            .take(10 * n as usize)
            .collect::<Vec<_>>()
            .into_iter();

        for _ in 0..n {
            h.push(values.next().unwrap());
            let _l = h.pop().unwrap();
        }
    }

    #[test]
    fn simple_test_list_block_bucket() {
        let mut h = ConfigurableSimdQuickHeap::<
            u64,
            list_block_bucket::ListBlockBucket<u64, 128>,
            Avx2,
            MedianOfM<3>,
            NoRebalancing,
        >::default();

        h.push(1);
        h.push(2);
        h.push(3);
        h.push(4);

        let e = h.pop().unwrap();
        assert!(e == 1);
    }

    #[test]
    fn simple_test_list_block_bucket_2() {
        let mut h = ConfigurableSimdQuickHeap::<
            u64,
            list_block_bucket::ListBlockBucket<u64, 128>,
            Avx2,
            MedianOfM<3>,
            NoRebalancing,
            4,
            true,
        >::default();

        h.push(14);
        h.push(10);
        assert_eq!(h.pop().unwrap(), 10);
        h.push(4);
        h.push(0);
        assert_eq!(h.pop().unwrap(), 0);
        assert_eq!(h.pop().unwrap(), 4);
        h.push(9);
        h.push(15);
        assert_eq!(h.pop().unwrap(), 9);
        assert_eq!(h.pop().unwrap(), 14);
        h.push(2);
        h.push(6);
        assert_eq!(h.pop().unwrap(), 2);
        assert_eq!(h.pop().unwrap(), 6);
        assert_eq!(h.pop().unwrap(), 15);
        h.push(8);
        h.push(1);
        assert_eq!(h.pop().unwrap(), 1);
        h.push(11);
        assert_eq!(h.pop().unwrap(), 8);
        h.push(3);
        assert_eq!(h.pop().unwrap(), 3);
        assert_eq!(h.pop().unwrap(), 11);
        h.push(13);
        h.push(7);
        h.push(5);
        h.push(12);
        assert_eq!(h.pop().unwrap(), 5);
        assert_eq!(h.pop().unwrap(), 7);
        assert_eq!(h.pop().unwrap(), 12);
        assert_eq!(h.pop().unwrap(), 13);
    }
}
