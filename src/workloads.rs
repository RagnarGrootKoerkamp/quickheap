use radix_heap::Radix;
use rand_distr::{Distribution, Geometric};

use crate::impls::NoHeap;

use super::Heap;
use std::hint::black_box;
use std::marker::PhantomData;

/// Small wrapper type for elements to support random numbers in the workloads.
pub trait Elem: Ord + std::fmt::Debug + Clone + Copy + Radix + 'static {
    fn get(&self) -> u64;
    fn from(x: u64) -> Self;
    fn try_from(x: u64) -> Self;
    fn stride() -> u64;
}

macro_rules! impl_elem {
    ($t:ty) => {
        impl Elem for $t {
            #[inline(always)]
            fn get(&self) -> u64 {
                *self as u64
            }

            #[inline(always)]
            fn from(x: u64) -> Self {
                x as $t
            }
            #[inline(always)]
            fn try_from(x: u64) -> Self {
                TryFrom::try_from(x).unwrap()
            }
            #[inline(always)]
            fn stride() -> u64 {
                if <$t>::BITS == 32 {
                    1u64 << 24
                } else {
                    1u64 << 48
                }
            }
        }
    };
}

impl_elem!(u8);
impl_elem!(u16);
impl_elem!(u32);
impl_elem!(u64);
impl_elem!(i32);
impl_elem!(i64);

thread_local! {
    pub(crate) static COMPARISONS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[derive(Clone, Copy, Debug, Ord, Eq)]
pub struct CountComparisons<T: Elem>(T);

impl<T: Elem> PartialEq for CountComparisons<T> {
    #[inline(always)]
    fn eq(&self, _other: &Self) -> bool {
        panic!()
    }
}
impl<T: Elem> PartialOrd for CountComparisons<T> {
    #[inline(always)]
    fn partial_cmp(&self, _other: &Self) -> Option<std::cmp::Ordering> {
        COMPARISONS.with(|c| c.set(c.get() + 1));
        self.0.partial_cmp(&_other.0)
    }
    fn lt(&self, other: &Self) -> bool {
        COMPARISONS.with(|c| c.set(c.get() + 1));
        self.0 < other.0
    }
    fn le(&self, other: &Self) -> bool {
        COMPARISONS.with(|c| c.set(c.get() + 1));
        self.0 <= other.0
    }
    fn gt(&self, other: &Self) -> bool {
        COMPARISONS.with(|c| c.set(c.get() + 1));
        self.0 > other.0
    }
    fn ge(&self, other: &Self) -> bool {
        COMPARISONS.with(|c| c.set(c.get() + 1));
        self.0 >= other.0
    }
}
impl<T: Elem> Radix for CountComparisons<T> {
    fn radix_similarity(&self, other: &Self) -> u32 {
        COMPARISONS.with(|c| c.set(c.get() + 1));
        self.0.radix_similarity(&other.0)
    }

    const RADIX_BITS: u32 = T::RADIX_BITS;
}
impl<T: Elem> Elem for CountComparisons<T> {
    #[inline(always)]
    fn get(&self) -> u64 {
        self.0.get()
    }

    #[inline(always)]
    fn from(x: u64) -> Self {
        Self(T::from(x))
    }

    #[inline(always)]
    fn try_from(x: u64) -> Self {
        Self(T::try_from(x))
    }

    #[inline(always)]
    fn stride() -> u64 {
        T::stride()
    }
}

pub trait CountingHeapT<T: Elem>: Heap<T> {
    fn reset_comparisons();
    fn get_comparisons() -> (u64, u64);
}

/// A heap wrapper that counts comparisons during push and pop separately.
///
/// Wraps `H: Heap<CountComparisons<T>>` and implements `Heap<T>`.
/// `push_comparisons` and `pop_comparisons` accumulate the per-operation
/// counts on this instance.
pub struct CountingHeap<T: Elem, H: Heap<CountComparisons<T>>> {
    inner: H,
    _phantom: PhantomData<T>,
}

thread_local! {
    pub(crate) static PUSH_COMPARISONS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    pub(crate) static POP_COMPARISONS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

impl<T: Elem, H: Heap<CountComparisons<T>>> Heap<T> for CountingHeap<T, H> {
    const MONOTONE: bool = H::MONOTONE;
    type CountedHeap = NoHeap;

    fn default() -> Self {
        Self {
            inner: H::default(),
            _phantom: PhantomData,
        }
    }

    fn push(&mut self, t: T) {
        COMPARISONS.with(|c| c.set(0));
        self.inner.push(CountComparisons(t));
        let delta = COMPARISONS.with(|c| c.get());
        PUSH_COMPARISONS.with(|c| c.set(c.get() + delta));
    }

    fn pop(&mut self) -> Option<T> {
        COMPARISONS.with(|c| c.set(0));
        let result = self.inner.pop().map(|c| c.0);
        let delta = COMPARISONS.with(|c| c.get());
        POP_COMPARISONS.with(|c| c.set(c.get() + delta));
        result
    }
}

impl<T: Elem, H: Heap<CountComparisons<T>>> CountingHeapT<T> for CountingHeap<T, H> {
    fn reset_comparisons() {
        PUSH_COMPARISONS.with(|c| c.set(0));
        POP_COMPARISONS.with(|c| c.set(0));
    }

    fn get_comparisons() -> (u64, u64) {
        (
            PUSH_COMPARISONS.with(|c| c.get()),
            POP_COMPARISONS.with(|c| c.get()),
        )
    }
}

pub trait Workload {
    /// The default is one push-pop pair per `n`.
    const NORMALIZATION: u64 = 1;
    /// n is the maximum size of the data structure.
    fn setup<T: Elem, H: Heap<T>>(n: u64) -> impl FnOnce();
}

/// bench: push^n pop^n
/// values: random
pub struct HeapSort;

impl Workload for HeapSort {
    fn setup<T: Elem, H: Heap<T>>(n: u64) -> impl FnOnce() {
        let mut h = H::default();
        let mut rng = fastrand::Rng::new();
        let values = std::iter::repeat_with(|| T::from(rng.u64(..)))
            .take(n as usize)
            .collect::<Vec<_>>()
            .into_iter();
        move || {
            for value in values {
                h.push(value);
            }
            for _ in 0..n {
                h.pop().unwrap().get();
            }
            black_box(h);
        }
    }
}

/// init: push^n
/// bench: (pop push)^n
/// values: last + (0..C)
/// C = 1000 for u32, C=2^32 for u64.
pub struct ConstantSize;

impl Workload for ConstantSize {
    fn setup<T: Elem, H: Heap<T>>(n: u64) -> impl FnOnce() {
        let mut h = H::default();
        let stride = T::stride();
        let mut rng = fastrand::Rng::new();
        for _ in 0..n {
            h.push(T::from(rng.u64(0..stride)));
        }
        let values = std::iter::repeat_with(|| rng.u64(0..stride))
            .take(n as usize)
            .collect::<Vec<_>>()
            .into_iter();
        move || {
            for value in values {
                let l = h.pop().unwrap().get();
                h.push(T::try_from(l + value));
            }
            black_box(h);
        }
    }
}

/// bench: (push pop push)^n (pop push pop)^n
/// values: last + (0..C)
pub struct MonotoneWiggle;

impl Workload for MonotoneWiggle {
    const NORMALIZATION: u64 = 3;
    fn setup<T: Elem, H: Heap<T>>(n: u64) -> impl FnOnce() {
        let mut h = H::default();
        let stride = T::stride();
        let mut rng = fastrand::Rng::new();
        let mut values = std::iter::repeat_with(|| rng.u64(0..stride))
            .take(3 * n as usize)
            .collect::<Vec<_>>()
            .into_iter();
        move || {
            let mut l = 0;
            for _ in 0..n {
                h.push(T::try_from(l + values.next().unwrap()));
                l = h.pop().unwrap().get();
                h.push(T::try_from(l + values.next().unwrap()));
            }
            for _ in 0..n {
                l = h.pop().unwrap().get();
                h.push(T::try_from(l + values.next().unwrap()));
                h.pop().unwrap().get();
            }
            black_box(h);
        }
    }
}

/// bench: (push pop push)^n (pop push pop)^n
/// values: last + geometric(C)
pub struct GeometricMonotoneWiggle;

impl Workload for GeometricMonotoneWiggle {
    const NORMALIZATION: u64 = 3;
    fn setup<T: Elem, H: Heap<T>>(n: u64) -> impl FnOnce() {
        let mut h = H::default();
        let stride = T::stride();
        let geometric = Geometric::new(1.0 / stride as f64).unwrap();
        let values = {
            let mut rng = rand::rng();
            geometric
                .sample_iter(&mut rng)
                .take(3 * n as usize)
                .collect::<Vec<u64>>()
        };
        move || {
            let mut values = values.into_iter();
            let mut l = 0;
            for _ in 0..n {
                h.push(T::try_from(l + values.next().unwrap()));
                l = h.pop().unwrap().get();
                h.push(T::try_from(l + values.next().unwrap()));
            }
            for _ in 0..n {
                l = h.pop().unwrap().get();
                h.push(T::try_from(l + values.next().unwrap()));
                h.pop().unwrap().get();
            }
            black_box(h);
        }
    }
}

/// bench: (push pop push)^n (pop push pop)^n
/// values: random
pub struct Wiggle;

impl Workload for Wiggle {
    const NORMALIZATION: u64 = 3;
    fn setup<T: Elem, H: Heap<T>>(n: u64) -> impl FnOnce() {
        let mut h = H::default();
        let stride = T::stride();
        let mut rng = fastrand::Rng::new();
        let mut values = std::iter::repeat_with(|| T::from(rng.u64(0..stride)))
            .take(3 * n as usize)
            .collect::<Vec<T>>()
            .into_iter();
        move || {
            for _ in 0..n {
                h.push(values.next().unwrap());
                h.pop().unwrap().get();
                h.push(values.next().unwrap());
            }
            for _ in 0..n {
                h.pop().unwrap().get();
                h.push(values.next().unwrap());
                h.pop().unwrap().get();
            }
            black_box(h);
        }
    }
}
