use crate::{Heap, impls::NoHeap, workloads};

impl<
    T: quickheap::Elem
        + workloads::Elem
        + std::ops::Sub<Output = T>
        + Default
        + quickheap::EqualBucketConstraints,
    B: quickheap::buckets::Bucket<T, K, CAP>,
    S: quickheap::SimdElem<T>,
    P: quickheap::pivot_strategies::PivotStrategy,
    R: quickheap::rebalancing_strategies::RebalancingStrategy<T, B, K, CAP>,
    const N: usize,
    const K: usize,
    const CAP: usize,
    const SORT: bool,
    const EQUAL: bool,
> Heap<T> for quickheap::ConfigurableSimdQuickHeap<T, B, S, P, R, N, K, CAP, SORT, EQUAL>
{
    type CountedType = workloads::CountComparisons<T>;
    type CountedHeap = NoHeap;

    fn default() -> Self {
        Default::default()
    }

    fn push(&mut self, t: T) {
        self.push(t)
    }

    fn pop(&mut self) -> Option<T> {
        self.pop()
    }

    fn capacity(&self) -> usize {
        self.capacity()
    }
}
