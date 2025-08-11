pub use fibonacci_heap::FibonacciHeap;
pub use orx_priority_queue::{DaryHeap, PriorityQueue};
pub use pheap::PairingHeap;
pub use radix_heap::RadixHeapMap;
pub use std::collections::{BTreeSet, BinaryHeap};

use super::*;

impl Heap for BinaryHeap<Reverse<T>> {
    fn default() -> Self {
        BinaryHeap::with_capacity(1 << 20)
    }
    #[inline(always)]
    fn push(&mut self, t: T) {
        self.push(Reverse(t));
    }
    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        Some(self.pop()?.0)
    }
}

impl Heap for BTreeSet<T> {
    fn default() -> Self {
        Default::default()
    }
    #[inline(always)]
    fn push(&mut self, t: T) {
        self.insert(t);
    }
    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        self.pop_first()
    }
}

impl Heap for BTreeSet<Reverse<T>> {
    fn default() -> Self {
        Default::default()
    }
    #[inline(always)]
    fn push(&mut self, t: T) {
        self.insert(Reverse(t));
    }
    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        Some(self.pop_last()?.0)
    }
}

impl<const N: usize> Heap for DaryHeap<(), T, N> {
    #[inline(always)]
    fn default() -> Self {
        Default::default()
    }

    #[inline(always)]
    fn push(&mut self, t: T) {
        PriorityQueue::push(self, (), t);
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        PriorityQueue::pop(self).map(|((), y)| y)
    }
}

impl Heap for FibonacciHeap {
    #[inline(always)]
    fn default() -> Self {
        Default::default()
    }

    #[inline(always)]
    fn push(&mut self, t: T) {
        self.insert(t as i32);
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        self.extract_min().map(|x| x as T)
    }
}

impl Heap for PairingHeap<(), T> {
    #[inline(always)]
    fn default() -> Self {
        Default::default()
    }

    #[inline(always)]
    fn push(&mut self, t: T) {
        self.insert((), t);
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        self.delete_min().map(|(_x, y)| y)
    }
}

impl Heap for RadixHeapMap<Reverse<T>, ()> {
    #[inline(always)]
    fn default() -> Self {
        Default::default()
    }

    #[inline(always)]
    fn push(&mut self, t: T) {
        self.push(Reverse(t), ());
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        self.pop().map(|(k, _v)| k.0)
    }
}

impl<const N: usize> Heap for dary_heap::DaryHeap<T, N> {
    #[inline(always)]
    fn default() -> Self {
        Default::default()
    }

    #[inline(always)]
    fn push(&mut self, t: T) {
        self.push(t)
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        self.pop()
    }
}

impl Heap for indexset::BTreeSet<T> {
    #[inline(always)]
    fn default() -> Self {
        Default::default()
    }

    #[inline(always)]
    fn push(&mut self, t: T) {
        self.insert(t);
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        self.pop_first()
    }
}

impl Heap for indexset::BTreeSet<Reverse<T>> {
    #[inline(always)]
    fn default() -> Self {
        Default::default()
    }

    #[inline(always)]
    fn push(&mut self, t: T) {
        self.insert(Reverse(t));
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        Some(self.pop_last()?.0)
    }
}
