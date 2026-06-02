use std::cmp::Reverse;

use crate::{
    ConfigurableSimdQuickHeap, SimdElem,
    pivot_strategies::MedianOfM,
    rebalancing_strategies::{NaiveLogRebalancing, NoRebalancing, PivotForgetting},
};

/// Element-type capabilities needed by the generators.
trait GenElem: Copy + Ord + std::fmt::Debug {
    fn gen_random() -> Self;
    fn gen_min() -> Self;
    fn gen_max() -> Self;
    fn wrapping_inc(self) -> Self;
    fn wrapping_dec(self) -> Self;
}

#[rustfmt::skip]
impl GenElem for u64 {
    fn gen_random() -> Self { rand::random() }
    fn gen_min() -> Self { u64::MIN }
    fn gen_max() -> Self { u64::MAX }
    fn wrapping_inc(self) -> Self { self.wrapping_add(1) }
    fn wrapping_dec(self) -> Self { self.wrapping_sub(1) }
}

#[rustfmt::skip]
impl GenElem for i64 {
    fn gen_random() -> Self { rand::random() }
    fn gen_min() -> Self { i64::MIN }
    fn gen_max() -> Self { i64::MAX }
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

/// 99.9 % T::MAX, 0.1 % uniformly random.
struct MostlyMaxGen;
impl<T: GenElem> Generator<T> for MostlyMaxGen {
    fn new() -> Self {
        Self
    }
    fn popped(&mut self, _x: T) {}
    fn get(&mut self) -> T {
        if rand::random::<u16>() == 0 {
            T::gen_random()
        } else {
            T::gen_max()
        }
    }
}

/// 99.9 % T::MIN, 0.1 % uniformly random.
struct MostlyMinGen;
impl<T: GenElem> Generator<T> for MostlyMinGen {
    fn new() -> Self {
        Self
    }
    fn popped(&mut self, _x: T) {}
    fn get(&mut self) -> T {
        if rand::random::<u16>() == 0 {
            T::gen_random()
        } else {
            T::gen_min()
        }
    }
}

fn heapsort_with_gen<T, S, G>()
where
    T: GenElem,
    S: SimdElem<T>,
    G: Generator<T>,
{
    let g = &mut G::new();
    for n in [10, 100, 1000, 10000, 100000] {
        let mut q =
            <ConfigurableSimdQuickHeap<T, S, MedianOfM<3>, PivotForgetting<2, 128>>>::default();
        for _ in 0..n {
            q.push(g.get());
        }
        let mut last: Option<T> = None;
        for _ in 0..n {
            let x = q.pop().unwrap();
            if let Some(prev) = last {
                assert!(x >= prev, "out of order: {x:?} < {prev:?}");
            }
            g.popped(x);
            last = Some(x);
        }
    }
}

fn wiggle_with_gen<T, S, G>()
where
    T: GenElem,
    S: SimdElem<T>,
    G: Generator<T>,
{
    let g = &mut G::new();
    for n in [10, 100, 1000, 10000, 100000] {
        let mut q1 =
            <ConfigurableSimdQuickHeap<T, S, MedianOfM<3>, PivotForgetting<2, 128>>>::default();
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
    }
}

#[rustfmt::skip]
macro_rules! all_tests {
    ($elem:ty, $simd:ty) => {
        #[test] fn heapsort_random()      { heapsort_with_gen::<$elem, $simd, RandomGen>(); }
        #[test] fn heapsort_increasing()  { heapsort_with_gen::<$elem, $simd, IncreasingGen<$elem>>(); }
        #[test] fn heapsort_decreasing()  { heapsort_with_gen::<$elem, $simd, DecreasingGen<$elem>>(); }
        #[test] fn heapsort_mostly_max()  { heapsort_with_gen::<$elem, $simd, MostlyMaxGen>(); }
        #[test] fn heapsort_mostly_min()  { heapsort_with_gen::<$elem, $simd, MostlyMinGen>(); }

        #[test] fn wiggle_random()        { wiggle_with_gen::<$elem, $simd, RandomGen>(); }
        #[test] fn wiggle_increasing()    { wiggle_with_gen::<$elem, $simd, IncreasingGen<$elem>>(); }
        #[test] fn wiggle_decreasing()    { wiggle_with_gen::<$elem, $simd, DecreasingGen<$elem>>(); }
        #[test] fn wiggle_mostly_max()    { wiggle_with_gen::<$elem, $simd, MostlyMaxGen>(); }
        #[test] fn wiggle_mostly_min()    { wiggle_with_gen::<$elem, $simd, MostlyMinGen>(); }
    };
}

#[rustfmt::skip]
mod u64 {
    mod avx2   { use super::super::*; all_tests!(u64, crate::Avx2); }
    #[cfg(target_feature = "avx512f")]
    mod avx512 { use super::super::*; all_tests!(u64, crate::Avx512); }
}

#[rustfmt::skip]
mod i64 {
    mod avx2   { use super::super::*; all_tests!(i64, crate::Avx2); }
    #[cfg(target_feature = "avx512f")]
    mod avx512 { use super::super::*; all_tests!(i64, crate::Avx512); }
}
