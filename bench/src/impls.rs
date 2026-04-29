use crate::workloads::{CountComparisons, CountingHeap, CountingHeapT, Elem};

use super::Heap;
use std::cmp::Reverse;

pub type BinaryHeap<T> = std::collections::BinaryHeap<Reverse<T>>;
pub type DaryHeap<T, const N: usize> = dary_heap::DaryHeap<Reverse<T>, N>;
pub type OrxDaryHeap<T, const N: usize> = orx_priority_queue::DaryHeap<(), T, N>;
pub type PairingHeap<T> = pheap::PairingHeap<(), T>;
pub type RadixHeap<T> = radix_heap::RadixHeapMap<Reverse<T>, ()>;
pub type WeakHeap<T> = weakheap::WeakHeap<Reverse<T>>;
pub type FibonacciHeap<T> = fibonacci_heap::GenericFibonacciHeap<T>;

/// set-based, so do not support duplicate elements.
pub type BTreeSet<T> = std::collections::BTreeSet<T>;
pub type RevBTreeSet<T> = std::collections::BTreeSet<Reverse<T>>;
pub type IndexSetBTreeSet<T> = indexset::BTreeSet<T>;
pub type IndexSetRevBTreeSet<T> = indexset::BTreeSet<Reverse<T>>;

impl<T: Elem> Heap<T> for BinaryHeap<T> {
    type CountedHeap = CountingHeap<T, BinaryHeap<CountComparisons<T>>>;
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

impl<T: Elem> Heap<T> for BTreeSet<T> {
    type CountedHeap = CountingHeap<T, BTreeSet<CountComparisons<T>>>;
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

impl<T: Elem> Heap<T> for RevBTreeSet<T> {
    type CountedHeap = CountingHeap<T, RevBTreeSet<CountComparisons<T>>>;
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

impl<T: Elem, const N: usize> Heap<T> for OrxDaryHeap<T, N> {
    type CountedHeap = CountingHeap<T, OrxDaryHeap<CountComparisons<T>, N>>;
    #[inline(always)]
    fn default() -> Self {
        Default::default()
    }

    #[inline(always)]
    fn push(&mut self, t: T) {
        orx_priority_queue::PriorityQueue::push(self, (), t);
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        orx_priority_queue::PriorityQueue::pop(self).map(|((), y)| y)
    }
}

impl<T: Elem> Heap<T> for FibonacciHeap<T> {
    type CountedHeap = CountingHeap<T, FibonacciHeap<CountComparisons<T>>>;
    #[inline(always)]
    fn default() -> Self {
        Default::default()
    }

    #[inline(always)]
    fn push(&mut self, t: T) {
        let _ = self.insert(t);
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        self.extract_min()
    }
}

pub struct NoHeap;
impl<T: Elem> Heap<T> for NoHeap {
    type CountedHeap = NoHeap;
    fn default() -> Self {
        unimplemented!()
    }
    fn push(&mut self, _t: T) {
        unimplemented!()
    }
    fn pop(&mut self) -> Option<T> {
        unimplemented!()
    }
}
impl<T: Elem> CountingHeapT<T> for NoHeap {
    fn reset_comparisons() {
        unimplemented!()
    }
    fn get_comparisons() -> (u64, u64) {
        unimplemented!()
    }
}

impl<T: Elem> Heap<T> for PairingHeap<T> {
    type CountedHeap = CountingHeap<T, PairingHeap<CountComparisons<T>>>;
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

impl<T: Elem> Heap<T> for RadixHeap<T> {
    type CountedHeap = CountingHeap<T, RadixHeap<CountComparisons<T>>>;
    const MONOTONE: bool = true;

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

impl<T: Elem, const N: usize> Heap<T> for DaryHeap<T, N> {
    type CountedHeap = CountingHeap<T, DaryHeap<CountComparisons<T>, N>>;
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

impl<T: Elem> Heap<T> for IndexSetBTreeSet<T> {
    type CountedHeap = CountingHeap<T, IndexSetBTreeSet<CountComparisons<T>>>;
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

impl<T: Elem> Heap<T> for IndexSetRevBTreeSet<T> {
    type CountedHeap = CountingHeap<T, IndexSetRevBTreeSet<CountComparisons<T>>>;
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

impl<T: Elem> Heap<T> for WeakHeap<T> {
    type CountedHeap = CountingHeap<T, WeakHeap<CountComparisons<T>>>;
    #[inline(always)]
    fn default() -> Self {
        Default::default()
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
