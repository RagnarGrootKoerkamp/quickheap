use super::*;

struct TestHeap<H0: Heap, H1: Heap>(H0, H1);
impl<H0: Heap, H1: Heap> Heap for TestHeap<H0, H1> {
    fn default() -> Self {
        TestHeap(H0::default(), H1::default())
    }

    fn push(&mut self, t: T) {
        (self.0.push(t), self.1.push(t));
    }

    fn pop(&mut self) -> Option<T> {
        let a0 = self.0.pop();
        let a1 = self.1.pop();
        assert_eq!(a0, a1);
        a0
    }
}

#[test]
fn quickheap() {
    use impls::*;
    // bench::bench::<TestHeap<QuickHeap<8, 3>, BinaryHeap<Reverse<T>>>>(false);
    // bench::bench::<TestHeap<QuickHeap<1, 1>, BinaryHeap<Reverse<T>>>>(false);
    // bench::bench::<TestHeap<dary_heap::DaryHeap<Reverse<T>, 2>, BinaryHeap<Reverse<T>>>>(false);
    // bench::bench::<TestHeap<dary_heap::DaryHeap<Reverse<T>, 4>, BinaryHeap<Reverse<T>>>>(false);
    // bench::bench::<TestHeap<dary_heap::DaryHeap<Reverse<T>, 8>, BinaryHeap<Reverse<T>>>>(false);
    // bench::bench::<TestHeap<DaryHeap<(), T, 2>, BinaryHeap<Reverse<T>>>>(false);
    // bench::bench::<TestHeap<DaryHeap<(), T, 4>, BinaryHeap<Reverse<T>>>>(false);
    // bench::bench::<TestHeap<DaryHeap<(), T, 8>, BinaryHeap<Reverse<T>>>>(false);
    bench::bench::<TestHeap<RadixHeapMap<Reverse<T>, ()>, BinaryHeap<Reverse<T>>>>(true);
}
