#![feature(portable_simd, vec_from_fn)]
pub mod impls;
pub mod scalar_quickheap;
mod simd;
pub mod simd_quickheap;
pub mod workloads;

pub trait Heap<T> {
    fn default() -> Self;
    fn push(&mut self, t: T);
    fn pop(&mut self) -> Option<T>;
}

#[cfg(test)]
mod test {
    use crate::workloads::Elem;
    use std::marker::PhantomData;

    use super::*;
    struct TestHeap<T, H0, H1>(H0, H1, PhantomData<T>);
    impl<T: Elem + Copy, H0: Heap<T>, H1: Heap<T>> Heap<T> for TestHeap<T, H0, H1> {
        fn default() -> Self {
            TestHeap(H0::default(), H1::default(), PhantomData)
        }

        fn push(&mut self, t: T) {
            eprintln!("Push {:?}", t);
            (self.0.push(t), self.1.push(t));
        }

        fn pop(&mut self) -> Option<T> {
            let a0 = self.0.pop();
            let a1 = self.1.pop();
            eprintln!("pop {:?} and {:?}", a0, a1);
            assert_eq!(a0, a1);
            a0
        }
    }

    impl<T: Elem + Copy, H0: Heap<T>, H1: Heap<T>> TestHeap<T, H0, H1> {
        fn run(n: u64) {
            eprintln!("Test: {:?}", std::any::type_name::<Self>());
            eprintln!("CONSTANT");
            workloads::constant_size::<T, Self>(n);
            eprintln!("LINEAR");
            workloads::push_linear::<T, Self>(n);
            eprintln!("LINEAR MIX");
            workloads::increasing_linear_mix::<T, Self, 2>(n);
            eprintln!("NATURAL");
            workloads::natural::<T, Self>(n);
        }
    }

    #[test]
    fn quickheap() {
        let n = 100000;

        type T = u64;
        type Base = impls::BinaryHeap<T>;

        TestHeap::<T, Base, impls::BinaryHeap<T>>::run(n);
        TestHeap::<T, Base, impls::DaryHeap<T, 2>>::run(n);
        TestHeap::<T, Base, impls::DaryHeap<T, 4>>::run(n);
        TestHeap::<T, Base, impls::DaryHeap<T, 8>>::run(n);
        TestHeap::<T, Base, impls::OrxDaryHeap<T, 2>>::run(n);
        TestHeap::<T, Base, impls::OrxDaryHeap<T, 4>>::run(n);
        TestHeap::<T, Base, impls::OrxDaryHeap<T, 8>>::run(n);
        TestHeap::<T, Base, impls::PairingHeap<T>>::run(n);
        TestHeap::<T, Base, impls::WeakHeap<T>>::run(n);
        // TestHeap::<i32, impls::BinaryHeap<i32>, impls::FibonacciHeap>::run(n); // broken

        {
            // Set-based implementations without support for duplicate elements.
            type Base = impls::BTreeSet<T>;
            TestHeap::<T, Base, impls::BTreeSet<T>>::run(n);
            TestHeap::<T, Base, impls::RevBTreeSet<T>>::run(n);
            TestHeap::<T, Base, impls::IndexSetBTreeSet<T>>::run(n);
            TestHeap::<T, Base, impls::IndexSetRevBTreeSet<T>>::run(n);
        }

        TestHeap::<T, Base, impls::RadixHeap<T>>::run(n);
    }

    #[test]
    fn fibonacci_heap() {
        let mut h = TestHeap::<i32, impls::BinaryHeap<i32>, impls::FibonacciHeap>::default();
        for n in 0..10 {
            eprintln!("Test {n} pushes and pops");
            for _ in 0..n {
                h.push(0);
            }
            for _ in 0..(n + 1) {
                h.pop();
            }
        }
    }
}
