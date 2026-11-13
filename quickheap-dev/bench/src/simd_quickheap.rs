use quickheap::decrease_key::DecreaseKeyStrategy;
use quickheap::partitioning::{MultiPartition, Partition};
use quickheap::{ElemT, KvSimdQuickHeap};

use quickheap_dev::{ElemConstraint, EqualSimdQuickHeap, MultiSimdQuickHeap, RadixSimdQuickHeap};

use crate::{Heap, KvHeap, impls::NoHeap, workloads};

impl<
    T: ElemT + workloads::Elem + Default + std::ops::Sub<Output = T> + ElemConstraint,
    B: quickheap_dev::buckets::Bucket<T, K, CAP>,
    PART: Partition<T, B> + MultiPartition<T, B>,
    S: quickheap::simd::SimdElem<T>,
    PIVOT: quickheap_dev::pivot_strategies::PivotStrategy,
    MUTLI_PIVOT: quickheap_dev::pivot_strategies::MultiPivotStrategy<P>,
    R: quickheap_dev::rebalancing_strategies::RebalancingStrategy<T, B, K, CAP>,
    const N: usize,
    const K: usize,
    const CAP: usize,
    const SORT: bool,
    const EQUAL: bool,
    const P: usize,
    const MULTI_THRESH: usize,
> Heap<T>
    for quickheap_dev::ConfigurableSimdQuickHeap<
        T,
        B,
        PART,
        S,
        PIVOT,
        MUTLI_PIVOT,
        R,
        N,
        K,
        CAP,
        SORT,
        EQUAL,
        P,
        MULTI_THRESH,
    >
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

impl<T: ElemT + workloads::Elem + Default, S: quickheap::simd::SimdElem<T>> Heap<T>
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

/// Flat-vector development heap, with the bucket and partitioner selected automatically.
pub type DevSimdQuickHeap<
    T,
    S = quickheap_dev::Simd,
    PIVOT = quickheap_dev::pivot_strategies::MedianOfM<3>,
    R = quickheap_dev::rebalancing_strategies::NoRebalancing,
    const N: usize = 16,
    const SORT: bool = true,
    const EQUAL: bool = false,
    const P: usize = 2,
    const K: usize = 128,
    const CAP: usize = 128,
> = quickheap_dev::ConfigurableSimdQuickHeap<
    T,
    quickheap_dev::buckets::vec_bucket::VecBucket<T>,
    quickheap_dev::buckets::block_partitioning::FlatPartitioning<S>,
    S,
    PIVOT,
    quickheap_dev::pivot_strategies::GeoMultiMedianOfM<2>,
    R,
    N,
    K,
    CAP,
    SORT,
    EQUAL,
    P,
    512,
>;

impl<
    T: ElemT + workloads::Elem + Default,
    S: quickheap::simd::SimdElem<T>,
    const B: usize,
    const L: usize,
> Heap<T> for RadixSimdQuickHeap<T, S, B, L>
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

impl<
    T: ElemT + workloads::Elem + Default,
    S: quickheap::simd::SimdElem<T>,
    DK: DecreaseKeyStrategy<T, S> + Default,
> KvHeap<T> for KvSimdQuickHeap<T, S, DK>
{
    fn default() -> Self {
        Default::default()
    }

    fn push(&mut self, k: T, v: T) {
        self.push(k, v);
    }

    fn pop(&mut self) -> Option<(T, T)> {
        self.pop()
    }

    fn decrease_key(&mut self, old_key: T, new_key: T, val: T) {
        self.decrease_key(old_key, new_key, val);
    }

    fn capacity(&self) -> usize {
        let sum = 0;

        sum
    }
}

impl<
    T: ElemConstraint + ElemT + workloads::Elem + Default,
    S: quickheap::simd::SimdElem<T>,
    const P: usize,
    const L: usize,
> Heap<T> for MultiSimdQuickHeap<T, S, P, L>
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

impl<T: ElemConstraint + ElemT + workloads::Elem + Default, S: quickheap::simd::SimdElem<T>> Heap<T>
    for EqualSimdQuickHeap<T, S>
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
