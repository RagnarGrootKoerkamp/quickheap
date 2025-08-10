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
fn bucketheap() {
    bench::bench::<TestHeap<BucketHeap<8, 3>, BinaryHeap<Reverse<T>>>>();
    bench::bench::<TestHeap<BucketHeap<1, 1>, BinaryHeap<Reverse<T>>>>();
}
