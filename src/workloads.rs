use radix_heap::Radix;

use super::Heap;
use std::hint::black_box;

/// Small wrapper type for elements to support random numbers in the workloads.
pub trait Elem: Ord + std::fmt::Debug + Clone + Copy + Radix + 'static {
    fn get(&self) -> u64;
    fn from(x: u64) -> Self;
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
            fn stride() -> u64 {
                if <$t>::BITS == 32 {
                    1000u64
                } else {
                    1u64 << 32
                }
            }
        }
    };
}
impl_elem!(u32);
impl_elem!(u64);
impl_elem!(i32);
impl_elem!(i64);

thread_local! {
    pub(crate) static COMPARISONS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    pub(crate) static PUSH_COMPARISONS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[derive(Clone, Copy, Debug, Ord, Eq)]
pub struct CountComparisons<T: Elem>(T);

impl<T: Elem> CountComparisons<T> {
    pub fn reset_count() {
        COMPARISONS.with(|c| c.set(0));
        PUSH_COMPARISONS.with(|c| c.set(0));
    }
    pub fn get_counts() -> (u64, u64) {
        (
            COMPARISONS.with(|c| c.get()),
            PUSH_COMPARISONS.with(|c| c.get()),
        )
    }
}

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
    fn stride() -> u64 {
        T::stride()
    }
}

/// Random element from u64.
fn get<T: Elem>(rng: &mut fastrand::Rng) -> T {
    T::from(rng.u64(..))
}

pub trait Workload {
    fn setup<T: Elem, H: Heap<T>>(n: u64) -> impl FnOnce();
}

/// bench: push^n pop^n
/// values: random
pub struct HeapSort;

impl Workload for HeapSort {
    fn setup<T: Elem, H: Heap<T>>(n: u64) -> impl FnOnce() {
        let mut h = H::default();
        let mut rng = fastrand::Rng::new();
        move || {
            for _ in 0..n {
                h.push(get(&mut rng));
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
        let mut rng = fastrand::Rng::new();
        let stride = T::stride();
        for _ in 0..n {
            h.push(T::from(rng.u64(0..stride)));
        }
        move || {
            for _ in 0..n {
                let l = h.pop().unwrap().get();
                h.push(T::from(rng.u64(l..l + stride)));
            }
            black_box(h);
        }
    }
}

/// bench: (push pop push)^(n/3) (pop push pop)^(n/3)
/// values: n..0
pub struct Decreasing;

impl Workload for Decreasing {
    fn setup<T: Elem, H: Heap<T>>(n: u64) -> impl FnOnce() {
        let mut h = H::default();
        let mut i = n;
        move || {
            for _ in 0..n / 3 {
                h.push(T::from(i));
                i -= 1;
                h.pop().unwrap().get();
                h.push(T::from(i));
            }
            for _ in 0..n / 3 {
                h.pop().unwrap().get();
                h.push(T::from(i));
                i -= 1;
                h.pop().unwrap().get();
            }
            black_box(h);
        }
    }
}
