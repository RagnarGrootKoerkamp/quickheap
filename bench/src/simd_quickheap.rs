use crate::{Heap, impls::NoHeap, workloads};

impl<
    T: quickheap::Elem + workloads::Elem,
    S: quickheap::SimdElem<T>,
    P: quickheap::pivot_strategies::PivotStrategy,
    const N: usize,
    const SORT: bool,
> Heap<T> for quickheap::ConfigurableSimdQuickHeap<T, S, P, N, SORT>
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
