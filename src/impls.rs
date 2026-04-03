use super::Heap;
use std::cmp::Reverse;

pub type BinaryHeap<T> = std::collections::BinaryHeap<Reverse<T>>;
pub type DaryHeap<T, const N: usize> = dary_heap::DaryHeap<Reverse<T>, N>;
pub type OrxDaryHeap<T, const N: usize> = orx_priority_queue::DaryHeap<(), T, N>;
pub type PairingHeap<T> = pheap::PairingHeap<(), T>;
pub type RadixHeap<T> = radix_heap::RadixHeapMap<Reverse<T>, ()>;
pub type WeakHeap<T> = weakheap::WeakHeap<Reverse<T>>;
/// only for i32
pub use fibonacci_heap::FibonacciHeap;

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
}

impl<T: Ord + Copy + radix_heap::Radix> Heap<T> for RadixHeap<T> {
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
}
