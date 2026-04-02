use std::{any::type_name, cmp::Reverse, collections::BinaryHeap, hint::black_box};

use orx_priority_queue::DaryHeap;
use radix_heap::RadixHeapMap;

use quickheap::*;

trait Elem: PartialOrd {
    fn get(&self) -> u64;
    fn from(x: u64) -> Self;
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
        }
    };
}
impl_elem!(u32);
impl_elem!(u64);
impl_elem!(i32);
impl_elem!(i64);

fn push_linear<T: Elem, H: Heap<T>>(n: u64) {
    let mut h = H::default();
    for i in 0..n {
        h.push(T::from(i));
    }
    black_box(h);
}

fn push_linear_rev<T: Elem, H: Heap<T>>(n: u64) {
    let mut h = H::default();
    for i in (0..n).rev() {
        h.push(T::from(i));
    }
    black_box(h);
}

fn get<T: Elem>(rng: &mut fastrand::Rng) -> T {
    T::from(rng.u64(..))
}

fn push_random<T: Elem, H: Heap<T>>(n: u64) {
    let mut h = H::default();
    let mut rng = fastrand::Rng::new();
    for _i in 0..n {
        h.push(get(&mut rng));
    }
    black_box(h);
}

fn linear<T: Elem, H: Heap<T>>(n: u64) {
    let mut h = H::default();
    for i in 0..n {
        h.push(T::from(i));
    }
    for _i in 0..n {
        h.pop();
    }
    black_box(h);
}

fn linear_mix<T: Elem, H: Heap<T>, const K: usize>(n: u64) {
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

fn linear_rev_mix<T: Elem, H: Heap<T>>(n: u64) {
    let mut h = H::default();
    for i in (0..n).rev() {
        h.push(T::from(2 * i + 1));
        h.push(T::from(2 * i));
        h.pop();
    }
    black_box(h);
}

fn linear_rev<T: Elem, H: Heap<T>>(n: u64) {
    let mut h = H::default();
    for i in (0..n).rev() {
        h.push(T::from(i));
    }
    for _i in 0..n {
        h.pop();
    }
    black_box(h);
}

fn random<T: Elem, H: Heap<T>>(n: u64) {
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

fn random_alternate<T: Elem, H: Heap<T>>(n: u64) {
    let mut h = H::default();
    let mut rng = fastrand::Rng::new();
    for _i in 0..n {
        h.push(get(&mut rng));
    }
    for _i in n..2 * n {
        h.push(get(&mut rng));
        h.pop();
    }
    black_box(h);
}

fn random_mix<T: Elem, H: Heap<T>, const K: usize>(n: u64) {
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

fn increasing_random_mix<T: Elem, H: Heap<T>, const K: usize>(n: u64) {
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

fn increasing_linear_mix<T: Elem, H: Heap<T>, const K: usize>(n: u64) {
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

fn natural<T: Elem, H: Heap<T>>(n: u64) {
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

pub fn time(n: u64, f: impl Fn(u64)) -> f64 {
    const REPEATS: usize = 2;
    let mut ts = vec![];
    for _ in 0..REPEATS {
        let start = std::time::Instant::now();
        f(n);
        ts.push(start.elapsed());
    }
    let t = ts.iter().min().unwrap();
    (t.as_secs_f64() / n as f64) * 1e9f64
}

pub fn bench<T: Elem, H: Heap<T>>(increasing: bool) {
    let minpow = 10;
    let maxpow = 25;
    let ns: Vec<_> = (minpow..=maxpow).map(|i| (2u64).pow(i)).collect();

    let mut ts = vec![
        (3, increasing_linear_mix::<T, H, 1> as fn(_)),
        (3, increasing_random_mix::<T, H, 1> as fn(_)),
        (1, random_mix::<T, H, 0> as fn(_)),
    ];
    if !increasing {
        ts.extend_from_slice(&[(1, random_mix::<T, H, 0> as fn(_))]);
        ts.extend_from_slice(&[(3, random_mix::<T, H, 1> as fn(_))]);
    }

    let mut ok = vec![true; ts.len()];

    for n in ns {
        eprint!("{:<70} {} {n:>10}", type_name::<H>(), type_name::<T>());
        print!("{}\t{}\t{n}", type_name::<H>(), type_name::<T>());
        for (&(cnt, t), ok) in std::iter::zip(&ts, &mut ok) {
            if !*ok {
                eprint!("{:>9}", "");
                print!("\t");
                continue;
            }
            let t = time(n, t) / (2 * cnt) as f64 / (n as f64).log2();
            eprint!("{t:>9.2}");
            print!("\t{t:>.2}");
            if t > 100.0 {
                *ok = false;
            }
        }
        eprintln!();
        println!();
    }
}

fn main() {
    type T = i32;

    eprintln!("QUICKHEAP");
    // bench::<QuickHeap<4, 1>>(false);
    // bench::<QuickHeap<8, 1>>(false);
    // bench::<QuickHeap<8, 3>>(false);
    bench::<T, QuickHeap<16, 1>>(false);
    // bench::<QuickHeap<16, 3>>(false);
    // bench::<QuickHeap<32, 1>>(false);
    // bench::<QuickHeap<32, 3>>(false);
    // bench::<QuickHeap<64, 3>>(false);

    // bench::<QuickHeap<32, 3>>(false);
    // bench::<QuickHeap<64, 3>>(false);

    // bench::<QuickHeap<16, 1>>(false); // actually slightly faster usually ??
    // bench::<QuickHeap<16, 5>>(false);

    eprintln!("BASELINE");
    bench::<T, BinaryHeap<Reverse<T>>>(false);

    eprintln!("DARY");
    // bench::<dary_heap::DaryHeap<Reverse<T>, 2>>(false);
    // bench::<dary_heap::DaryHeap<Reverse<T>, 4>>(false);
    bench::<T, dary_heap::DaryHeap<Reverse<T>, 8>>(false);
    // bench::<DaryHeap<(), T, 2>>(false);
    bench::<T, DaryHeap<(), T, 4>>(false);
    // bench::<DaryHeap<(), T, 8>>(false);

    eprintln!("RADIX");
    bench::<T, RadixHeapMap<Reverse<T>, ()>>(true);
    //
    // eprintln!("BTREES");
    // bench::<BTreeSet<T>>(false);
    // bench::<BTreeSet<Reverse<T>>>(false);
    // bench::<indexset::BTreeSet<T>>(false);
    // bench::<indexset::BTreeSet<Reverse<T>>>(false);

    // eprintln!("FANCY");
    // bench::<PairingHeap<(), T>>(false);
    // bench::<FibonacciHeap>(false); // too slow
    // bench::<WeakHeap>(false);
}

#[cfg(test)]
mod test {
    use super::*;
    struct TestHeap<H0: Heap, H1: Heap>(H0, H1);
    impl<H0: Heap, H1: Heap> Heap for TestHeap<H0, H1> {
        fn default() -> Self {
            TestHeap(H0::default(), H1::default())
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

    #[test]
    fn quickheap() {
        // bench::<TestHeap<QuickHeap<8, 3>, BinaryHeap<Reverse<T>>>>(false);
        // bench::<TestHeap<QuickHeap<1, 1>, BinaryHeap<Reverse<T>>>>(false);
        // bench::<TestHeap<dary_heap::DaryHeap<Reverse<T>, 2>, BinaryHeap<Reverse<T>>>>(false);
        // bench::<TestHeap<dary_heap::DaryHeap<Reverse<T>, 4>, BinaryHeap<Reverse<T>>>>(false);
        // bench::<TestHeap<dary_heap::DaryHeap<Reverse<T>, 8>, BinaryHeap<Reverse<T>>>>(false);
        // bench::<TestHeap<DaryHeap<(), T, 2>, BinaryHeap<Reverse<T>>>>(false);
        // bench::<TestHeap<DaryHeap<(), T, 4>, BinaryHeap<Reverse<T>>>>(false);
        // bench::<TestHeap<DaryHeap<(), T, 8>, BinaryHeap<Reverse<T>>>>(false);
        // bench::<TestHeap<WeakHeap<T>, BinaryHeap<Reverse<T>>>>(false);
        bench::<TestHeap<RadixHeapMap<Reverse<T>, ()>, BinaryHeap<Reverse<T>>>>(true);
    }
}
