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

/// When all elements equal T::MAX and the layer
/// exceeds N (16), partitioning uses `wrapping_add_one(T::MAX)` which wraps
/// to 0 (for unsigned) or T::MIN (for signed). This causes all elements to
/// be classified as "large", leaving the new bottom layer empty. The
/// subsequent `pop` then panics on `unwrap()` of an empty layer.
fn max_value_partition<S: SimdElem<u64>>() {
    let mut q = <ConfigurableSimdQuickHeap<_, S>>::default();
    // Push more than N=16 copies of u64::MAX to force partitioning on pop.
    for _ in 0..20 {
        q.push(u64::MAX);
    }
    for _ in 0..20 {
        assert_eq!(q.pop(), Some(u64::MAX));
    }
    assert_eq!(q.pop(), None);
}

#[test]
fn max_value_partition_avx2() {
    max_value_partition::<crate::Avx2>();
}

#[cfg(target_feature = "avx512f")]
#[test]
fn max_value_partition_avx512() {
    max_value_partition::<crate::Avx512>();
}

/// 1000 copies of u64::MAX plus one smaller value.
/// The single smaller value must be popped first.
fn u64_max_with_one_smaller<S: SimdElem<u64>>() {
    let mut q = <ConfigurableSimdQuickHeap<_, S>>::default();
    let small_val: u64 = 42;
    for _ in 0..1000 {
        q.push(u64::MAX);
    }
    q.push(small_val);
    assert_eq!(q.pop(), Some(small_val));
    for _ in 0..1000 {
        assert_eq!(q.pop(), Some(u64::MAX));
    }
    assert_eq!(q.pop(), None);
}

#[test]
fn u64_max_with_one_smaller_avx2() {
    u64_max_with_one_smaller::<crate::Avx2>();
}

#[cfg(target_feature = "avx512f")]
#[test]
fn u64_max_with_one_smaller_avx512() {
    u64_max_with_one_smaller::<crate::Avx512>();
}

/// 1000 copies of u64::MIN (0) plus one larger value.
/// All zeros must be popped before the larger value.
fn u64_min_with_one_larger<S: SimdElem<u64>>() {
    let mut q = <ConfigurableSimdQuickHeap<_, S>>::default();
    let large_val: u64 = 999;
    for _ in 0..1000 {
        q.push(u64::MIN);
    }
    q.push(large_val);
    for _ in 0..1000 {
        assert_eq!(q.pop(), Some(u64::MIN));
    }
    assert_eq!(q.pop(), Some(large_val));
    assert_eq!(q.pop(), None);
}

#[test]
fn u64_min_with_one_larger_avx2() {
    u64_min_with_one_larger::<crate::Avx2>();
}

#[cfg(target_feature = "avx512f")]
#[test]
fn u64_min_with_one_larger_avx512() {
    u64_min_with_one_larger::<crate::Avx512>();
}

/// 1000 copies of i64::MAX plus one smaller value.
fn i64_max_with_one_smaller<S: SimdElem<i64>>() {
    let mut q = <ConfigurableSimdQuickHeap<_, S>>::default();
    let small_val: i64 = 7;
    for _ in 0..1000 {
        q.push(i64::MAX);
    }
    q.push(small_val);
    assert_eq!(q.pop(), Some(small_val));
    for _ in 0..1000 {
        assert_eq!(q.pop(), Some(i64::MAX));
    }
    assert_eq!(q.pop(), None);
}

#[test]
fn i64_max_with_one_smaller_avx2() {
    i64_max_with_one_smaller::<crate::Avx2>();
}

#[cfg(target_feature = "avx512f")]
#[test]
fn i64_max_with_one_smaller_avx512() {
    i64_max_with_one_smaller::<crate::Avx512>();
}

/// 1000 copies of i64::MIN plus one larger value.
fn i64_min_with_one_larger<S: SimdElem<i64>>() {
    let mut q = <ConfigurableSimdQuickHeap<_, S>>::default();
    let large_val: i64 = -5;
    for _ in 0..1000 {
        q.push(i64::MIN);
    }
    q.push(large_val);
    for _ in 0..1000 {
        assert_eq!(q.pop(), Some(i64::MIN));
    }
    assert_eq!(q.pop(), Some(large_val));
    assert_eq!(q.pop(), None);
}

#[test]
fn i64_min_with_one_larger_avx2() {
    i64_min_with_one_larger::<crate::Avx2>();
}

#[cfg(target_feature = "avx512f")]
#[test]
fn i64_min_with_one_larger_avx512() {
    i64_min_with_one_larger::<crate::Avx512>();
}

/// Pure i64::MAX — all elements equal, same overflow risk as u64::MAX.
fn i64_max_all_equal<S: SimdElem<i64>>() {
    let mut q = <ConfigurableSimdQuickHeap<_, S>>::default();
    for _ in 0..1000 {
        q.push(i64::MAX);
    }
    for _ in 0..1000 {
        assert_eq!(q.pop(), Some(i64::MAX));
    }
    assert_eq!(q.pop(), None);
}

#[test]
fn i64_max_all_equal_avx2() {
    i64_max_all_equal::<crate::Avx2>();
}

#[cfg(target_feature = "avx512f")]
#[test]
fn i64_max_all_equal_avx512() {
    i64_max_all_equal::<crate::Avx512>();
}

/// Pure i64::MIN — all equal at the minimum boundary.
fn i64_min_all_equal<S: SimdElem<i64>>() {
    let mut q = <ConfigurableSimdQuickHeap<_, S>>::default();
    for _ in 0..1000 {
        q.push(i64::MIN);
    }
    for _ in 0..1000 {
        assert_eq!(q.pop(), Some(i64::MIN));
    }
    assert_eq!(q.pop(), None);
}

#[test]
fn i64_min_all_equal_avx2() {
    i64_min_all_equal::<crate::Avx2>();
}

#[cfg(target_feature = "avx512f")]
#[test]
fn i64_min_all_equal_avx512() {
    i64_min_all_equal::<crate::Avx512>();
}