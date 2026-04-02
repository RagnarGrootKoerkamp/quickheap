pub use fibonacci_heap::FibonacciHeap;
pub use orx_priority_queue::{DaryHeap, PriorityQueue};
pub use pheap::PairingHeap;
use radix_heap::Radix;
pub use radix_heap::RadixHeapMap;
use std::cmp::Reverse;
pub use std::collections::{BTreeSet, BinaryHeap};

use super::Heap;

impl<T: Ord> Heap<T> for BinaryHeap<Reverse<T>> {
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

impl<T: Ord> Heap<T> for BTreeSet<T> {
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

impl<T: Ord> Heap<T> for BTreeSet<Reverse<T>> {
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

impl<T: Ord + Clone, const N: usize> Heap<T> for DaryHeap<(), T, N> {
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

impl Heap<i32> for FibonacciHeap {
    #[inline(always)]
    fn default() -> Self {
        Default::default()
    }

    #[inline(always)]
    fn push(&mut self, t: i32) {
        self.insert(t);
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<i32> {
        self.extract_min()
    }
}

impl<T: Ord> Heap<T> for PairingHeap<(), T> {
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

impl<T: Ord + Copy + Radix> Heap<T> for RadixHeapMap<Reverse<T>, ()> {
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

impl<T: Ord, const N: usize> Heap<T> for dary_heap::DaryHeap<Reverse<T>, N> {
    #[inline(always)]
    fn default() -> Self {
        Default::default()
    }

    #[inline(always)]
    fn push(&mut self, t: T) {
        self.push(Reverse(t))
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        self.pop().map(|x| x.0)
    }
}

impl<T: Ord> Heap<T> for indexset::BTreeSet<T> {
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

impl<T: Ord> Heap<T> for indexset::BTreeSet<Reverse<T>> {
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

impl<T: Ord> Heap<T> for weakheap::WeakHeap<T> {
    #[inline(always)]
    fn default() -> Self {
        Default::default()
    }

    #[inline(always)]
    fn push(&mut self, t: T) {
        self.push(t);
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        self.pop()
    }
}
