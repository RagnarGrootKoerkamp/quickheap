#![feature(adt_const_params, associated_type_defaults)]

use workloads::{CountComparisons, CountingHeapT, Elem};
pub mod workloads;

pub mod impls;

#[cfg(feature = "boost")]
pub mod boost_heap;
#[cfg(feature = "ffi")]
pub mod radix_heap;
#[cfg(feature = "ffi")]
pub mod randomized_quickheap;
#[cfg(feature = "ffi")]
pub mod s3q;
#[cfg(feature = "ffi")]
pub mod sequence_heap;

pub mod scalar_quickheap;
pub mod simd_quickheap;

#[cfg(feature = "avx2")]
pub use quickheap::SimdQuickHeap;

pub mod binary_heap;
pub mod dary_heap;

pub mod dijkstra;
pub mod graph;
pub mod graph_util;
pub mod prim;
pub mod reimplemented_quickheap;

pub trait Heap<T: Elem>
where
    <Self as Heap<T>>::CountedType: Elem,
{
    const MONOTONE: bool = false;
    type CountedType = CountComparisons<T>;
    type CountedHeap: CountingHeapT<T>;
    fn default() -> Self;
    fn push(&mut self, t: T);
    fn pop(&mut self) -> Option<T>;
    fn capacity(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod test {

    use crate::impls::NoHeap;
    use crate::reimplemented_quickheap::ReimplementedQuickHeap;
    #[cfg(feature = "avx512")]
    use crate::simd::Avx512;
    #[cfg(feature = "avx2")]
    use quickheap::pivot_strategies::{MedianOfM, RandomPivot};

    use crate::scalar_quickheap::{ScalarQuickHeap, Search};
    use crate::workloads::{Elem, Workload};

    #[cfg(feature = "avx2")]
    use quickheap::{Avx2, ConfigurableSimdQuickHeap, rebalancing_strategies};

    use std::marker::PhantomData;

    use super::*;
    struct TestHeap<T, H0, H1>(H0, H1, PhantomData<T>);
    impl<T: Elem + Copy, H0: Heap<T>, H1: Heap<T>> Heap<T> for TestHeap<T, H0, H1> {
        const MONOTONE: bool = H0::MONOTONE || H1::MONOTONE;
        type CountedHeap = NoHeap;
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
    }

    impl<T: Elem + Copy, H0: Heap<T>, H1: Heap<T>> TestHeap<T, H0, H1> {
        fn run(n: u64) {
            eprintln!("Test: {:?}", std::any::type_name::<Self>());
            workloads::HeapSort::setup::<T, Self>(n)();
            workloads::MonotoneConstantSize::setup::<T, Self>(n)();
            workloads::MonotoneWiggle::setup::<T, Self>(n)();
            if !Self::MONOTONE {
                workloads::RandomWiggle::setup::<T, Self>(n)();
            }
        }
    }

    #[test]
    fn all_heaps() {
        let n = 100000;

        type T = u32;
        type Base = impls::BinaryHeap<T>;

        TestHeap::<T, Base, ReimplementedQuickHeap<T>>::run(n);

        TestHeap::<T, Base, binary_heap::CustomBinaryHeap<T>>::run(n);
        TestHeap::<T, Base, dary_heap::CustomDaryHeap<T, 2>>::run(n);
        TestHeap::<T, Base, dary_heap::CustomDaryHeap<T, 3>>::run(n);
        TestHeap::<T, Base, dary_heap::CustomDaryHeap<T, 4>>::run(n);
        TestHeap::<T, Base, dary_heap::CustomDaryHeap<T, 8>>::run(n);
        TestHeap::<T, Base, dary_heap::CustomDaryHeap<T, 16>>::run(n);

        TestHeap::<T, Base, impls::BinaryHeap<T>>::run(n);

        TestHeap::<T, Base, ScalarQuickHeap<T, 1, false>>::run(n);
        TestHeap::<T, Base, ScalarQuickHeap<T, 3, false>>::run(n);
        TestHeap::<T, Base, ScalarQuickHeap<T, 1, false, { Search::LinearScan }>>::run(n);

        #[cfg(feature = "avx2")]
        TestHeap::<
            T,
            Base,
            ConfigurableSimdQuickHeap<
                T,
                Avx2,
                MedianOfM<3>,
                rebalancing_strategies::NoRebalancing,
                8,
                true,
            >,
        >::run(n);
        #[cfg(feature = "avx2")]
        TestHeap::<
            T,
            Base,
            ConfigurableSimdQuickHeap<
                T,
                Avx2,
                RandomPivot,
                rebalancing_strategies::NoRebalancing,
                16,
                true,
            >,
        >::run(n);
        #[cfg(feature = "avx512")]
        TestHeap::<T, Base, ConfigurableSimdQuickHeap<T, Avx512<false>, 8, 3, true>>::run(n);
        #[cfg(feature = "avx512")]
        TestHeap::<T, Base, ConfigurableSimdQuickHeap<T, Avx512<false>, 16, 1, true>>::run(n);
        #[cfg(feature = "avx512")]
        TestHeap::<T, Base, ConfigurableSimdQuickHeap<T, Avx512<true>, 8, 3, true>>::run(n);
        #[cfg(feature = "avx512")]
        TestHeap::<T, Base, ConfigurableSimdQuickHeap<T, Avx512<true>, 16, 1, true>>::run(n);

        #[cfg(feature = "ffi")]
        TestHeap::<T, Base, sequence_heap::SequenceHeapU32>::run(n);
        #[cfg(feature = "ffi")]
        TestHeap::<T, Base, s3q::S3qHeapU32>::run(n);
        #[cfg(feature = "ffi")]
        TestHeap::<T, Base, randomized_quickheap::RandQH2HeapU32>::run(n);
        #[cfg(feature = "ffi")]
        TestHeap::<T, Base, radix_heap::RadixHeapU32>::run(n);

        #[cfg(feature = "boost")]
        TestHeap::<T, Base, boost_heap::BoostDary4HeapU32>::run(n);
        #[cfg(feature = "boost")]
        TestHeap::<T, Base, boost_heap::BoostFibHeapU32>::run(n);
        #[cfg(feature = "boost")]
        TestHeap::<T, Base, boost_heap::BoostPairingHeapU32>::run(n);
        #[cfg(feature = "boost")]
        TestHeap::<T, Base, boost_heap::BoostBinomialHeapU32>::run(n);
        #[cfg(feature = "boost")]
        TestHeap::<T, Base, boost_heap::BoostSkewHeapU32>::run(n);

        TestHeap::<T, Base, impls::DaryHeap<T, 2>>::run(n);
        TestHeap::<T, Base, impls::DaryHeap<T, 4>>::run(n);
        TestHeap::<T, Base, impls::DaryHeap<T, 8>>::run(n);
        TestHeap::<T, Base, impls::OrxDaryHeap<T, 2>>::run(n);
        TestHeap::<T, Base, impls::OrxDaryHeap<T, 4>>::run(n);
        TestHeap::<T, Base, impls::OrxDaryHeap<T, 8>>::run(n);
        TestHeap::<T, Base, impls::PairingHeap<T>>::run(n);
        TestHeap::<T, Base, impls::WeakHeap<T>>::run(n);
        TestHeap::<T, Base, impls::FibonacciHeap<T>>::run(n); // broken

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
}
