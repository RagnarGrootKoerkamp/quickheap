use fibonacci_heap::FibonacciHeap;
use orx_priority_queue::PriorityQueue;

use super::*;

impl<const N: usize> Heap for orx_priority_queue::DaryHeap<(), T, N> {
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

impl Heap for fibonacci_heap::FibonacciHeap {
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

impl Heap for pheap::PairingHeap<(), T> {
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

impl Heap for radix_heap::RadixHeapMap<T, ()> {
    #[inline(always)]
    fn default() -> Self {
        Default::default()
    }

    #[inline(always)]
    fn push(&mut self, t: T) {
        self.push(t, ());
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        self.pop().map(|(k, _v)| k)
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
