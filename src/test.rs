use std::cmp::Reverse;

use crate::{ConfigurableSimdQuickHeap, SimdElem};

fn heapsort<S: SimdElem<u64>>() {
    for n in [10, 100, 1000, 10000, 100000] {
        let mut data = vec![0u64; n];
        rand::fill(&mut data);
        let mut q = <ConfigurableSimdQuickHeap<_, S>>::default();
        for x in data {
            eprintln!("push {x}");
            q.push(x);
        }
        let mut last = 0;
        for _ in 0..n {
            let x = q.pop().unwrap();
            eprintln!("pop {x:?}");
            assert!(x >= last);
            last = x;
        }
    }
}

#[test]
fn heapsort_avx2() {
    heapsort::<crate::Avx2>();
}

#[cfg(target_feature = "avx512f")]
#[test]
fn heapsort_avx512() {
    heapsort::<crate::Avx512>();
}

fn wiggle<S: SimdElem<u64>>() {
    for n in [10, 100, 1000, 10000, 100000] {
        let mut q1 = <ConfigurableSimdQuickHeap<_, S>>::default();
        let mut q2 = std::collections::binary_heap::BinaryHeap::default();

        // (push pop push) xn
        for _ in 0..n {
            let x = rand::random::<u64>();
            q1.push(x);
            q2.push(Reverse(x));

            assert_eq!(q1.pop(), q2.pop().map(|x| x.0));

            let x = rand::random();
            q1.push(x);
            q2.push(Reverse(x));
        }

        // (pop push pop) xn
        for _ in 0..n {
            assert_eq!(q1.pop(), q2.pop().map(|x| x.0));

            let x = rand::random();
            q1.push(x);
            q2.push(Reverse(x));

            assert_eq!(q1.pop(), q2.pop().map(|x| x.0));
        }
    }
}

#[test]
fn wiggle_avx2() {
    wiggle::<crate::Avx2>();
}

#[cfg(target_feature = "avx512f")]
#[test]
fn wiggle_avx512() {
    wiggle::<crate::Avx512>();
}
