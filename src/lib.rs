#![feature(portable_simd, vec_from_fn)]

use workloads::Elem;
pub mod impls;
pub mod s3q;
pub mod scalar_quickheap;
pub mod simd;
pub mod simd_quickheap;
pub mod workloads;

pub trait Heap<T> {
    const MONOTONE: bool = false;
    type Casted<T2: Elem>: Heap<T2>;
    fn default() -> Self;
    fn push(&mut self, t: T);
    fn pop(&mut self) -> Option<T>;
}

#[cfg(test)]
mod test {
    use crate::{
        scalar_quickheap::ScalarQuickHeap,
        simd_quickheap::SimdQuickHeap,
        workloads::{Elem, Workload},
    };
    use std::marker::PhantomData;

    use super::*;
    struct TestHeap<T, H0, H1>(H0, H1, PhantomData<T>);
    impl<T: Elem + Copy, H0: Heap<T>, H1: Heap<T>> Heap<T> for TestHeap<T, H0, H1> {
        const MONOTONE: bool = H0::MONOTONE || H1::MONOTONE;
        fn default() -> Self {
            TestHeap(H0::default(), H1::default(), PhantomData)
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
        type Casted<T2: Elem> = TestHeap<T2, H0::Casted<T2>, H1::Casted<T2>>;
    }

    impl<T: Elem + Copy, H0: Heap<T>, H1: Heap<T>> TestHeap<T, H0, H1> {
        fn run(n: u64) {
            eprintln!("Test: {:?}", std::any::type_name::<Self>());
            eprintln!("HEAP SORT");
            workloads::HeapSort::setup::<T, Self>(n)();
            eprintln!("CONSTANT");
            workloads::ConstantSize::setup::<T, Self>(n)();
            if !Self::MONOTONE {
                eprintln!("DECREASING");
                workloads::Decreasing::setup::<T, Self>(n)();
            }
        }
    }

    #[test]
    fn quickheap() {
        let n = 100000;

        type T = i64;
        type Base = impls::BinaryHeap<T>;

        TestHeap::<T, Base, impls::BinaryHeap<T>>::run(n);

        TestHeap::<T, Base, ScalarQuickHeap<T, 3>>::run(n);
        TestHeap::<T, Base, SimdQuickHeap<T, 8, 3>>::run(n);
        TestHeap::<T, Base, s3q::S3qHeapI64>::run(n);

        TestHeap::<T, Base, impls::DaryHeap<T, 2>>::run(n);
        TestHeap::<T, Base, impls::DaryHeap<T, 4>>::run(n);
        TestHeap::<T, Base, impls::DaryHeap<T, 8>>::run(n);
        TestHeap::<T, Base, impls::OrxDaryHeap<T, 2>>::run(n);
        TestHeap::<T, Base, impls::OrxDaryHeap<T, 4>>::run(n);
        TestHeap::<T, Base, impls::OrxDaryHeap<T, 8>>::run(n);
        TestHeap::<T, Base, impls::PairingHeap<T>>::run(n);
        TestHeap::<T, Base, impls::WeakHeap<T>>::run(n);
        // TestHeap::<i32, impls::BinaryHeap<i32>, impls::FibonacciHeap>::run(n); // broken

        if false {
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
    #[ignore = "Bug in FibonacciHeap crate"]
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
