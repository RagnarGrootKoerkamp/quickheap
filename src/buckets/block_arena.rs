use std::{
    alloc::{Layout, alloc_zeroed, dealloc},
    ptr,
};

use crate::buckets::Block;

const SLAB_BLOCKS: usize = 1024;

pub struct BlockArena<T, const K: usize, const CAP: usize> {
    free: Vec<*mut Block<T, K, CAP>>,
    slabs: Vec<*mut Block<T, K, CAP>>,
}

impl<T, const K: usize, const CAP: usize> BlockArena<T, K, CAP> {
    pub fn new() -> Self {
        Self {
            free: Vec::with_capacity(SLAB_BLOCKS),
            slabs: Vec::new(),
        }
    }

    #[inline]
    pub fn alloc(&mut self) -> *mut Block<T, K, CAP> {
        if let Some(b) = self.free.pop() {
            return b;
        }
        self.grow();
        self.free.pop().unwrap()
    }

    #[inline]
    pub fn free(&mut self, b: *mut Block<T, K, CAP>) {
        unsafe {
            (*b).size = 0;
            (*b).next = ptr::null_mut();
            (*b).prev = ptr::null_mut()
        };
        self.free.push(b);
    }

    #[cold]
    fn grow(&mut self) {
        let layout = Layout::array::<Block<T, K, CAP>>(SLAB_BLOCKS).unwrap();
        // zeroed: valid for the integer T (u32/i32/u64/i64) and size = 0
        let p = unsafe { alloc_zeroed(layout) as *mut Block<T, K, CAP> };
        assert!(!p.is_null());
        self.slabs.push(p);
        // reverse push => pop() hands out ascending addresses
        for i in (0..SLAB_BLOCKS).rev() {
            self.free.push(unsafe { p.add(i) });
        }
    }
}

impl<T, const K: usize, const CAP: usize> Drop for BlockArena<T, K, CAP> {
    fn drop(&mut self) {
        let layout = Layout::array::<Block<T, K, CAP>>(SLAB_BLOCKS).unwrap();
        for &p in &self.slabs {
            unsafe { dealloc(p as *mut u8, layout) };
        }
    }
}
