use crate::simd_quickheap::T_U32;

use super::Heap;
use std::hint::black_box;

pub trait Elem: Ord {
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
                (if <$t>::BITS == 32 { 1000 } else { 1 << 32 }) as u64
            }
        }
    };
}
impl_elem!(u32);
impl_elem!(u64);
impl_elem!(i32);
impl_elem!(i64);

thread_local! {
    static COMPARISON: std::cell::Cell<u64> = std::cell::Cell::new(0);
}

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
        panic!()
    }
    fn lt(&self, other: &Self) -> bool {
        COMPARISON.with(|c| c.set(c.get() + 1));
        self.0 < other.0
    }
    fn le(&self, other: &Self) -> bool {
        COMPARISON.with(|c| c.set(c.get() + 1));
        self.0 <= other.0
    }
    fn gt(&self, other: &Self) -> bool {
        COMPARISON.with(|c| c.set(c.get() + 1));
        self.0 > other.0
    }
    fn ge(&self, other: &Self) -> bool {
        COMPARISON.with(|c| c.set(c.get() + 1));
        self.0 >= other.0
    }
}

/// Random element from u64.
fn get<T: Elem>(rng: &mut fastrand::Rng) -> T {
    T::from(rng.u64(..))
}

/// Push 0 to n
pub fn push_linear<T: Elem, H: Heap<T>>(n: u64) {
    let mut h = H::default();
    for i in 0..n {
        h.push(T::from(i));
    }
    black_box(h);
}

/// Push n to 0
pub fn push_linear_rev<T: Elem, H: Heap<T>>(n: u64) {
    let mut h = H::default();
    for i in (0..n).rev() {
        h.push(T::from(i));
    }
    black_box(h);
}

/// Push random.
pub fn push_random<T: Elem, H: Heap<T>>(n: u64) {
    let mut h = H::default();
    let mut rng = fastrand::Rng::new();
    for _i in 0..n {
        h.push(get(&mut rng));
    }
    black_box(h);
}

/// Push 0 to n, then pop all.
pub fn linear<T: Elem, H: Heap<T>>(n: u64) {
    let mut h = H::default();
    for i in 0..n {
        h.push(T::from(i));
    }
    for _i in 0..n {
        h.pop();
    }
    black_box(h);
}

/// (push (pop push)^K)^n (pop (push pop)^K)^n
/// values are linear increasing
pub fn linear_mix<T: Elem, H: Heap<T>, const K: usize>(n: u64) {
    let mut h = H::default();
    let mut x = 0;
    for _ in 0..n {
        h.push(T::from(x));
        x += 1;
        for _ in 0..K {
            h.pop();
            h.push(T::from(x));
            x += 1;
        }
    }
    for _ in 0..n {
        h.pop();
        for _ in 0..K {
            h.push(T::from(x));
            x += 1;
            h.pop();
        }
    }
    black_box(h);
}

/// (push push pop)^n
/// values are linear decreasing
pub fn linear_rev_mix<T: Elem, H: Heap<T>>(n: u64) {
    let mut h = H::default();
    for i in (0..n).rev() {
        h.push(T::from(2 * i + 1));
        h.push(T::from(2 * i));
        h.pop();
    }
    black_box(h);
}

/// push^n pop^n
/// values are linear decreasing
pub fn linear_rev<T: Elem, H: Heap<T>>(n: u64) {
    let mut h = H::default();
    for i in (0..n).rev() {
        h.push(T::from(i));
    }
    for _i in 0..n {
        h.pop();
    }
    black_box(h);
}

/// push^n pop^n
/// random values, heapsort
pub fn heapsort<T: Elem, H: Heap<T>>(n: u64) {
    let mut h = H::default();
    let mut rng = fastrand::Rng::new();
    for _i in 0..n {
        h.push(get(&mut rng));
    }
    for _i in 0..n {
        h.pop();
    }
    black_box(h);
}

/// push^n (push pop)^n
/// random values
pub fn random_alternate<T: Elem, H: Heap<T>>(n: u64) {
    let mut h = H::default();
    let mut rng = fastrand::Rng::new();
    for _i in 0..n {
        h.push(get(&mut rng));
    }
    for _i in 0..n {
        h.push(get(&mut rng));
        h.pop();
    }
    black_box(h);
}

/// (push (pop push)^K)^n (pop (push pop)^K)^n
/// random values
pub fn random_mix<T: Elem, H: Heap<T>, const K: usize>(n: u64) {
    let mut h = H::default();
    let mut rng = fastrand::Rng::new();
    for _ in 0..n {
        h.push(get(&mut rng));
        for _ in 0..K {
            h.pop();
            h.push(get(&mut rng));
        }
    }
    for _ in 0..n {
        h.pop();
        for _ in 0..K {
            h.push(get(&mut rng));
            h.pop();
        }
    }
    black_box(h);
}

/// (push (pop push)^K)^n (pop (push pop)^K)^n
/// values are last-popped + random(O, C)
/// (C=1000 for u32, C=2^32 for u64)
pub fn increasing_random_mix<T: Elem, H: Heap<T>, const K: usize>(n: u64) {
    let mut h = H::default();
    let mut rng = fastrand::Rng::new();
    let mut l = 0;
    const C: u64 = if !T_U32 { 1 << 32 } else { 1000 };
    for _ in 0..n {
        h.push(T::from(rng.u64(l..l + C)));
        for _ in 0..K {
            l = h.pop().unwrap().get();
            h.push(T::from(rng.u64(l..l + C)));
        }
    }
    for _ in 0..n {
        l = h.pop().unwrap().get();
        for _ in 0..K {
            h.push(T::from(rng.u64(l..l + C)));
            l = h.pop().unwrap().get();
        }
    }
    black_box(h);
}

/// (push (pop push)^K)^n (pop (push pop)^K)^n
/// linear increasing values
pub fn increasing_linear_mix<T: Elem, H: Heap<T>, const K: usize>(n: u64) {
    let mut h = H::default();
    let mut l = 0;
    for _ in 0..n {
        h.push(T::from(l));
        l += 1;
        for _ in 0..K {
            h.pop().unwrap().get();
            h.push(T::from(l));
            l += 1;
        }
    }
    for _ in 0..n {
        h.pop().unwrap().get();
        for _ in 0..K {
            h.push(T::from(l));
            l += 1;
            h.pop().unwrap().get();
        }
    }
    black_box(h);
}

/// (push^(s-i) pop^i)^(s+1)
/// start with mostly pushes, then slowly transition to mostly pops
/// random values.
pub fn natural<T: Elem, H: Heap<T>>(n: u64) {
    let mut h = H::default();
    let mut rng = fastrand::Rng::new();

    let s = n.isqrt();
    for i in 0..=s {
        for j in 0..s {
            if j < i {
                h.pop();
            } else {
                h.push(get(&mut rng));
            }
        }
    }
    black_box(h);
    // assert_eq!(h.pop(), None);
}

/// push^n (push pop)^10n
/// values are last-popped + random(O, stride)
/// (stride=1000 for u32, stride=2^32 for u64)
pub fn constant_size<T: Elem, H: Heap<T>>(n: u64) {
    let mut h = H::default();
    let mut rng = fastrand::Rng::new();
    let stride = T::stride();
    for _ in 0..n {
        h.push(T::from(rng.u64(0..stride)));
    }
    for _ in 0..n {
        let l = h.pop().unwrap().get();
        h.push(T::from(rng.u64(l..l + stride)));
    }
    black_box(h);
}
