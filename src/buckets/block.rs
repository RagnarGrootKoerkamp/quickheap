use std::{fmt::Debug, ptr};

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Block<T, const K: usize, const CAP: usize> {
    size: usize,
    next: *mut Block<T, K, CAP>,
    prev: *mut Block<T, K, CAP>,
    data: [T; CAP],
}

impl<T: Default + Ord + Copy + Debug + PartialOrd, const K: usize, const CAP: usize>
    Block<T, K, CAP>
{
    pub fn default() -> Self {
        Self {
            size: 0,
            data: [T::default(); CAP],
            next: ptr::null_mut(),
            prev: ptr::null_mut(),
        }
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.size = 0;
        self.next = ptr::null_mut();
        self.prev = ptr::null_mut();
    }

    #[inline]
    pub fn empty(&self) -> bool {
        self.size == 0
    }

    #[inline]
    pub fn from_slice(slice: &[T]) -> Self {
        let mut data = [T::default(); CAP];
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

        Self {
            size,
            next: ptr::null_mut(),
            prev: ptr::null_mut(),
            data,
        }
    }

    #[inline]
    pub fn as_slice(&self, from: usize, len: usize) -> &[T] {
        debug_assert!(from + len <= K);
        unsafe { &self.data.get_unchecked(from..from + len) }
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    #[inline]
    pub fn push(&mut self, elem: T) {
        assert!(self.size < K);
        self.data[self.size] = elem;
        self.size += 1;
    }

    #[inline]
    pub fn sort_decreasing(&mut self) {
        self.data[..self.size].sort_unstable_by_key(|&x| std::cmp::Reverse(x));
    }

    #[inline]
    pub fn insert(&mut self, elem: T, pos: usize) {
        debug_assert!(self.size < K);
        self.data[pos..=self.size].rotate_right(1);
        self.data[pos] = elem;
        self.size += 1;
    }

    #[inline]
    pub fn insert_index(&self, elem: T) -> usize {
        let mut idx = 0;

        for i in 0..self.size {
            if elem < self.data[i] {
                idx += 1;
            }
        }

        idx
    }

    #[inline]
    pub fn remove(&mut self, i: usize) -> T {
        assert!(i < K);
        assert!(self.size > 0);

        let elem = self.data[i];
        self.data[i] = self.data[self.size - 1];
        self.size -= 1;

        elem
    }

    #[inline]
    pub fn get(&self, i: usize) -> T {
        assert!(self.size <= K);
        assert!(i < self.size);
        self.data[i]
    }

    #[inline]
    pub fn get_unchecked(&self, i: usize) -> T {
        assert!(self.size <= K);
        self.data[i]
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn full(&self) -> bool {
        self.size >= K
    }

    pub fn print(&self) {
        let data = self.data.to_vec();
        println!("{:?}", &data[..self.size]);
    }

    #[inline]
    fn override_elem(&mut self, pos: usize, val: T) {
        self.data[pos] = val;
    }

    #[inline]
    pub fn set_len(&mut self, len: usize) {
        self.size = len;
    }

    #[inline]
    pub fn next(&self) -> *mut Block<T, K, CAP> {
        self.next
    }

    #[inline]
    pub fn prev(&self) -> *mut Block<T, K, CAP> {
        self.prev
    }

    #[inline]
    pub fn set_next(&mut self, next: *mut Block<T, K, CAP>) {
        self.next = next;
    }

    #[inline]
    pub fn set_prev(&mut self, prev: *mut Block<T, K, CAP>) {
        self.prev = prev;
    }
}
