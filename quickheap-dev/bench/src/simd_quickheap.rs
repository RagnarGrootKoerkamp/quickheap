use quickheap::partitioning::{MultiPartition, Partition};

use crate::{Heap, impls::NoHeap, workloads};

impl<
    T: quickheap::Elem
        + workloads::Elem
        + std::ops::Sub<Output = T>
        + Default
        + quickheap_dev::ElemConstraint,
    B: quickheap_dev::buckets::Bucket<T, K, CAP>,
    PART: Partition<T, B> + MultiPartition<T, B>,
    S: quickheap::simd::SimdElem<T>,
    PIVOT: quickheap_dev::pivot_strategies::PivotStrategy,
    R: quickheap_dev::rebalancing_strategies::RebalancingStrategy<T, B, K, CAP>,
    const N: usize,
    const K: usize,
    const CAP: usize,
    const SORT: bool,
    const EQUAL: bool,
    const P: usize,
> Heap<T>
    for quickheap_dev::ConfigurableSimdQuickHeap<T, B, PART, S, PIVOT, R, N, K, CAP, SORT, EQUAL, P>
{
    type CountedType = workloads::CountComparisons<T>;
    type CountedHeap = NoHeap;

    fn default() -> Self {
        Default::default()
    }

    fn push(&mut self, t: T) {
        self.push(t);
    }

    fn pop(&mut self) -> Option<T> {
        self.pop()
    }

    fn capacity(&self) -> usize {
        self.capacity()
    }
}

impl<T: quickheap::Elem + workloads::Elem + Default, S: quickheap::simd::SimdElem<T>> Heap<T>
    for quickheap::SimdQuickHeap<T, S>
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
