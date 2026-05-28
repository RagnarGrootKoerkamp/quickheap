use crate::{Heap, impls::NoHeap, workloads};

impl<
    T: quickheap::Elem + workloads::Elem,
    S: quickheap::SimdElem<T>,
    P: quickheap::pivot_strategies::PivotStrategy,
    R: quickheap::rebalancing_strategies::RebalancingStrategy<T>,
    const N: usize,
    const SORT: bool,
> Heap<T> for quickheap::ConfigurableSimdQuickHeap<T, S, P, R, N, SORT>
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
}
