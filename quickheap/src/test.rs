use std::{cmp::Reverse, collections::BTreeSet};

use crate::simd::SimdElem;
use crate::{KvSimdQuickHeap, SimdQuickHeap};

/// Element-type capabilities needed by the generators.
trait GenElem: Copy + Ord + std::fmt::Debug {
    fn gen_random() -> Self;
    fn gen_min() -> Self;
    fn gen_max() -> Self;
    #[allow(unused)]
    fn gen_zero() -> Self;
    fn wrapping_inc(self) -> Self;
    fn wrapping_dec(self) -> Self;
}

#[rustfmt::skip]
impl GenElem for u64 {
    fn gen_random() -> Self { rand::random() }
    fn gen_min() -> Self { u64::MIN }
    fn gen_max() -> Self { u64::MAX }
    fn gen_zero() -> Self { 0 }
    fn wrapping_inc(self) -> Self { self.wrapping_add(1) }
    fn wrapping_dec(self) -> Self { self.wrapping_sub(1) }
}

#[rustfmt::skip]
impl GenElem for i64 {
    fn gen_random() -> Self { rand::random() }
    fn gen_min() -> Self { i64::MIN }
    fn gen_max() -> Self { i64::MAX }
    fn gen_zero() -> Self { 0 }
    fn wrapping_inc(self) -> Self { self.wrapping_add(1) }
    fn wrapping_dec(self) -> Self { self.wrapping_sub(1) }
}

#[rustfmt::skip]
impl GenElem for u32 {
    fn gen_random() -> Self { rand::random() }
    fn gen_min() -> Self { u32::MIN }
    fn gen_max() -> Self { u32::MAX }
    fn gen_zero() -> Self { 0 }
    fn wrapping_inc(self) -> Self { self.wrapping_add(1) }
    fn wrapping_dec(self) -> Self { self.wrapping_sub(1) }
}

#[rustfmt::skip]
impl GenElem for i32 {
    fn gen_random() -> Self { rand::random() }
    fn gen_min() -> Self { i32::MIN }
    fn gen_max() -> Self { i32::MAX }
    fn gen_zero() -> Self { 0 }
    fn wrapping_inc(self) -> Self { self.wrapping_add(1) }
    fn wrapping_dec(self) -> Self { self.wrapping_sub(1) }
}

trait Generator<T> {
    fn new() -> Self;
    fn popped(&mut self, x: T);
    fn get(&mut self) -> T;
}

/// Uniformly random values.
struct RandomGen;
impl<T: GenElem> Generator<T> for RandomGen {
    fn new() -> Self {
        Self
    }
    fn popped(&mut self, _x: T) {}
    fn get(&mut self) -> T {
        T::gen_random()
    }
}

/// MIN, MIN+1, MIN+2, … (wrapping).
struct IncreasingGen<T>(T);
impl<T: GenElem> Generator<T> for IncreasingGen<T> {
    fn new() -> Self {
        Self(T::gen_min())
    }
    fn popped(&mut self, _x: T) {}
    fn get(&mut self) -> T {
        let v = self.0;
        self.0 = v.wrapping_inc();
        v
    }
}

/// MAX, MAX-1, MAX-2, … (wrapping).
struct DecreasingGen<T>(T);
impl<T: GenElem> Generator<T> for DecreasingGen<T> {
    fn new() -> Self {
        Self(T::gen_max())
    }
    fn popped(&mut self, _x: T) {}
    fn get(&mut self) -> T {
        let v = self.0;
        self.0 = v.wrapping_dec();
        v
    }
}

/// approximately 99.85 % T::MAX, approximately 0.0015 % uniformly random.
struct MostlyMaxGen;
impl<T: GenElem> Generator<T> for MostlyMaxGen {
    fn new() -> Self {
        Self
    }
    fn popped(&mut self, _x: T) {}
    fn get(&mut self) -> T {
        if rand::random::<u16>() < 100 {
            T::gen_random()
        } else {
            T::gen_max()
        }
    }
}

/// approximately 99.85 % T::MIN, approximately 0.0015 % uniformly random.
struct MostlyMinGen;
impl<T: GenElem> Generator<T> for MostlyMinGen {
    fn new() -> Self {
        Self
    }
    fn popped(&mut self, _x: T) {}
    fn get(&mut self) -> T {
        if rand::random::<u16>() < 100 {
            T::gen_random()
        } else {
            T::gen_min()
        }
    }
}

/// Common interface so every workload runs against both heap variants.
trait TestHeap<T>: Sized {
    fn new() -> Self;
    fn push(&mut self, key: T);
    fn pop(&mut self) -> Option<T>;
    fn len(&self) -> usize;
}

impl<T: GenElem + Default, S: SimdElem<T>> TestHeap<T> for SimdQuickHeap<T, S> {
    fn new() -> Self {
        Self::default()
    }
    fn push(&mut self, key: T) {
        self.push(key);
    }
    fn pop(&mut self) -> Option<T> {
        self.pop()
    }
    fn len(&self) -> usize {
        self.len()
    }
}

/// Give each insertion a distinct value and verify every returned key/value pair.
/// Equal keys may be returned in any order.
struct CheckedKv<T: GenElem, S: SimdElem<T>> {
    heap: KvSimdQuickHeap<T, S>,
    entries: BTreeSet<(T, T)>,
    next_value: T,
}

impl<T: GenElem + Default, S: SimdElem<T>> TestHeap<T> for CheckedKv<T, S> {
    fn new() -> Self {
        Self {
            heap: KvSimdQuickHeap::new(),
            entries: BTreeSet::new(),
            next_value: T::gen_min(),
        }
    }

    fn push(&mut self, key: T) {
        let value = self.next_value;
        self.next_value = value.wrapping_inc();
        assert!(self.entries.insert((key, value)));
        self.heap.push(key, value);
    }

    fn pop(&mut self) -> Option<T> {
        match self.heap.pop() {
            Some((key, value)) => {
                assert!(
                    self.entries.remove(&(key, value)),
                    "unexpected pair: ({key:?}, {value:?})"
                );
                Some(key)
            }
            None => {
                assert!(self.entries.is_empty());
                None
            }
        }
    }

    fn len(&self) -> usize {
        assert_eq!(self.heap.len(), self.entries.len());
        self.heap.len()
    }
}

fn heapsort_with_gen<T, Q, G, const L: usize>()
where
    T: GenElem + Default,
    Q: TestHeap<T>,
    G: Generator<T>,
{
    let g = &mut G::new();
    for n in [
        0,
        1,
        L - 1,
        L,
        L + 1,
        2 * L - 1,
        2 * L,
        2 * L + 1,
        10,
        100,
        1000,
        10000,
        100000,
    ] {
        let mut q = Q::new();

        for _ in 0..n {
            let t = g.get();
            q.push(t);
        }
        assert_eq!(q.len(), n);
        let mut last: Option<T> = None;
        for _ in 0..n {
            let x = q.pop().unwrap();
            if let Some(prev) = last {
                assert!(x >= prev, "out of order: {x:?} < {prev:?}");
            }
            g.popped(x);
            last = Some(x);
        }
        assert_eq!(q.len(), 0);
        assert_eq!(q.pop(), None);
    }
}

fn wiggle_with_gen<T, Q, G, const L: usize>()
where
    T: GenElem + Default,
    Q: TestHeap<T>,
    G: Generator<T>,
{
    let g = &mut G::new();
    for n in [
        0,
        1,
        L - 1,
        L,
        L + 1,
        2 * L - 1,
        2 * L,
        2 * L + 1,
        10,
        100,
        1000,
        10000,
        100000,
    ] {
        let mut q1 = Q::new();
        let mut q2 = std::collections::binary_heap::BinaryHeap::default();

        // (push pop push) xn
        for _ in 0..n {
            let x = g.get();
            q1.push(x);
            q2.push(Reverse(x));

            let p = q1.pop();

            assert_eq!(p, q2.pop().map(|v| v.0));

            if let Some(v) = p {
                g.popped(v);
            }

            let x = g.get();
            q1.push(x);
            q2.push(Reverse(x));
        }

        assert_eq!(q1.len(), n);

        // (pop push pop) xn
        for _ in 0..n {
            let p = q1.pop();
            assert_eq!(p, q2.pop().map(|v| v.0));

            if let Some(v) = p {
                g.popped(v);
            }

            let x = g.get();
            q1.push(x);
            q2.push(Reverse(x));

            let p = q1.pop();

            assert_eq!(p, q2.pop().map(|v| v.0));

            if let Some(v) = p {
                g.popped(v);
            }
        }
        assert_eq!(q1.len(), 0);
        assert_eq!(q1.pop(), None);
    }
}

#[rustfmt::skip]
macro_rules! heap_tests {
    ($elem:ty, $heap:ty, $l:expr) => {
        #[test] fn heapsort_random()      { heapsort_with_gen::<$elem, $heap, RandomGen, $l>(); }
        #[test] fn heapsort_increasing()  { heapsort_with_gen::<$elem, $heap, IncreasingGen<$elem>, $l>(); }
        #[test] fn heapsort_decreasing()  { heapsort_with_gen::<$elem, $heap, DecreasingGen<$elem>, $l>(); }
        #[test] fn heapsort_mostly_max()  { heapsort_with_gen::<$elem, $heap, MostlyMaxGen, $l>(); }
        #[test] fn heapsort_mostly_min()  { heapsort_with_gen::<$elem, $heap, MostlyMinGen, $l>(); }

        #[test] fn wiggle_random()        { wiggle_with_gen::<$elem, $heap, RandomGen, $l>(); }
        #[test] fn wiggle_increasing()    { wiggle_with_gen::<$elem, $heap, IncreasingGen<$elem>, $l>(); }
        #[test] fn wiggle_decreasing()    { wiggle_with_gen::<$elem, $heap, DecreasingGen<$elem>, $l>(); }
        #[test] fn wiggle_mostly_max()    { wiggle_with_gen::<$elem, $heap, MostlyMaxGen, $l>(); }
        #[test] fn wiggle_mostly_min()    { wiggle_with_gen::<$elem, $heap, MostlyMinGen, $l>(); }
    };
}

macro_rules! all_tests {
    ($elem:ty, $simd:ty, $l:expr) => {
        mod key_only {
            use super::*;
            heap_tests!($elem, SimdQuickHeap<$elem, $simd>, $l);
        }
        mod kv {
            use super::*;
            heap_tests!($elem, CheckedKv<$elem, $simd>, $l);
        }
    };
}

#[rustfmt::skip]
mod u64 {
    #[cfg(target_feature = "avx2")]
    mod avx2   { use super::super::*; all_tests!(u64, crate::simd::Avx2, 4); }

    #[cfg(target_feature = "avx512f")]
    mod avx512 { use super::super::*; all_tests!(u64, crate::simd::Avx512, 8); }

    #[cfg(target_arch = "aarch64")]
    mod neon   { use super::super::*; all_tests!(u64, crate::simd::Neon, 2); }
}

#[rustfmt::skip]
mod i64 {
    #[cfg(target_feature = "avx2")]
    mod avx2   { use super::super::*; all_tests!(i64, crate::simd::Avx2, 4); }

    #[cfg(target_feature = "avx512f")]
    mod avx512 { use super::super::*; all_tests!(i64, crate::simd::Avx512, 8); }

    #[cfg(target_arch = "aarch64")]
    mod neon   { use super::super::*; all_tests!(i64, crate::simd::Neon, 2); }
}

#[rustfmt::skip]
mod i32 {
    #[cfg(target_feature = "avx2")]
    mod avx2   { use super::super::*; all_tests!(i32, crate::simd::Avx2, 8); }

    #[cfg(target_feature = "avx512f")]
    mod avx512 { use super::super::*; all_tests!(i32, crate::simd::Avx512, 16); }

    #[cfg(target_arch = "aarch64")]
    mod neon   { use super::super::*; all_tests!(i32, crate::simd::Neon, 4); }
}

#[rustfmt::skip]
mod u32 {
    #[cfg(target_feature = "avx2")]
    mod avx2   { use super::super::*; all_tests!(u32, crate::simd::Avx2, 8); }

    #[cfg(target_feature = "avx512f")]
    mod avx512 { use super::super::*; all_tests!(u32, crate::simd::Avx512, 16); }

    #[cfg(target_arch = "aarch64")]
    mod neon   { use super::super::*; all_tests!(u32, crate::simd::Neon, 4); }
}
