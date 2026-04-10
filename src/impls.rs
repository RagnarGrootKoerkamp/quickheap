use crate::workloads::Elem;

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

impl<T: Ord> Heap<T> for BinaryHeap<T> {
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

    type Casted<T2: Elem> = BinaryHeap<T2>;
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

    type Casted<T2: Elem> = BTreeSet<T2>;
}

impl<T: Ord> Heap<T> for RevBTreeSet<T> {
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

    type Casted<T2: Elem> = RevBTreeSet<T2>;
}

impl<T: Ord + Clone, const N: usize> Heap<T> for OrxDaryHeap<T, N> {
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

    type Casted<T2: Elem> = OrxDaryHeap<T2, N>;
}

impl<T: Ord + Clone + std::fmt::Debug + 'static> Heap<T> for FibonacciHeap<T> {
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

    type Casted<T2: Elem> = FibonacciHeap<T2>;
}

pub struct NoHeap;
impl<T> Heap<T> for NoHeap {
    fn default() -> Self {
        unimplemented!()
    }
    fn push(&mut self, _t: T) {
        unimplemented!()
    }
    fn pop(&mut self) -> Option<T> {
        unimplemented!()
    }
    type Casted<T2: Elem> = Self;
}

impl<T: Ord> Heap<T> for PairingHeap<T> {
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

    type Casted<T2: Elem> = PairingHeap<T2>;
}

impl<T: Ord + Copy + radix_heap::Radix> Heap<T> for RadixHeap<T> {
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

    type Casted<T2: Elem> = RadixHeap<T2>;
}

impl<T: Ord, const N: usize> Heap<T> for DaryHeap<T, N> {
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

    type Casted<T2: Elem> = DaryHeap<T2, N>;
}

impl<T: Ord> Heap<T> for IndexSetBTreeSet<T> {
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

    type Casted<T2: Elem> = IndexSetBTreeSet<T2>;
}

impl<T: Ord> Heap<T> for IndexSetRevBTreeSet<T> {
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

    type Casted<T2: Elem> = IndexSetRevBTreeSet<T2>;
}

impl<T: Ord> Heap<T> for WeakHeap<T> {
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

    type Casted<T2: Elem> = WeakHeap<T2>;
}
