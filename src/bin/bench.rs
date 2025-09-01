use std::{any::type_name, cmp::Reverse, collections::BinaryHeap, hint::black_box};

use orx_priority_queue::DaryHeap;
use radix_heap::RadixHeapMap;
use rand::{rng, seq::SliceRandom};

use quickheap::*;

fn push_random_shuffle<H: Heap>(n: T) -> impl Fn(T) {
    let mut v: Vec<T> = (0..n).collect();
    v.shuffle(&mut rng());
    move |n| {
        let mut h = H::default();
        for &x in &v {
            h.push(x);
        }
        black_box(h);
    }
}

fn push_linear<H: Heap>(n: T) {
    let mut h = H::default();
    for i in 0..n {
        h.push(i);
    }
    black_box(h);
}

fn push_linear_rev<H: Heap>(n: T) {
    let mut h = H::default();
    for i in (0..n).rev() {
        h.push(i);
    }
    black_box(h);
}

fn get(rng: &mut fastrand::Rng) -> T {
    rng.u64(..) as T
}

fn push_random<H: Heap>(n: T) {
    let mut h = H::default();
    let mut rng = fastrand::Rng::new();
    for i in 0..n {
        h.push(get(&mut rng));
    }
    black_box(h);
}

fn linear<H: Heap>(n: T) {
    let mut h = H::default();
    for i in 0..n {
        h.push(i);
    }
    for _i in 0..n {
        h.pop();
    }
    black_box(h);
}

fn linear_mix<H: Heap, const K: usize>(n: T) {
    let mut h = H::default();
    let mut x = 0;
    for _ in 0..n {
        h.push(x);
        x += 1;
        for _ in 0..K {
            h.pop();
            h.push(x);
            x += 1;
        }
    }
    for _ in 0..n {
        h.pop();
        for _ in 0..K {
            h.push(x);
            x += 1;
            h.pop();
        }
    }
    black_box(h);
}

fn linear_rev_mix<H: Heap>(n: T) {
    let mut h = H::default();
    for i in (0..n).rev() {
        h.push(2 * i + 1);
        h.push(2 * i);
        h.pop();
    }
    black_box(h);
}

fn linear_rev<H: Heap>(n: T) {
    let mut h = H::default();
    for i in (0..n).rev() {
        h.push(i);
    }
    for _i in 0..n {
        h.pop();
    }
    black_box(h);
}

fn random<H: Heap>(n: T) {
    let mut h = H::default();
    let mut rng = fastrand::Rng::new();
    for i in 0..n {
        h.push(get(&mut rng));
    }
    for _i in 0..n {
        h.pop();
    }
    black_box(h);
}

fn random_alternate<H: Heap>(n: T) {
    let mut h = H::default();
    let mut rng = fastrand::Rng::new();
    for i in 0..n {
        h.push(get(&mut rng));
    }
    for i in n..2 * n {
        h.push(get(&mut rng));
        h.pop();
    }
    black_box(h);
}

fn random_mix<H: Heap, const K: usize>(n: T) {
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

fn increasing_random_mix<H: Heap, const K: usize>(n: T) {
    let mut h = H::default();
    let mut rng = fastrand::Rng::new();
    let mut l = 0;
    const C: u64 = if !T_U32 { 1 << 32 } else { 1000 };
    for _ in 0..n {
        h.push(rng.u64(l..l + C) as T);
        for _ in 0..K {
            l = h.pop().unwrap() as u64;
            h.push(rng.u64(l..l + C) as T);
        }
    }
    for _ in 0..n {
        l = h.pop().unwrap() as u64;
        for _ in 0..K {
            h.push(rng.u64(l..l + C) as T);
            l = h.pop().unwrap() as u64;
        }
    }
    black_box(h);
}

fn increasing_linear_mix<H: Heap, const K: usize>(n: T) {
    let mut h = H::default();
    let mut l = 0;
    for _ in 0..n {
        h.push(l);
        l += 1;
        for _ in 0..K {
            h.pop().unwrap() as u64;
            h.push(l);
            l += 1;
        }
    }
    for _ in 0..n {
        h.pop().unwrap() as u64;
        for _ in 0..K {
            h.push(l);
            l += 1;
            h.pop().unwrap() as u64;
        }
    }
    black_box(h);
}

fn natural<H: Heap>(n: T) {
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

pub fn time(n: T, f: impl Fn(T)) -> f64 {
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

pub fn bench<H: Heap>(increasing: bool) {
    let minpow = 10;
    let maxpow = 25;
    let ns: Vec<_> = (minpow..=maxpow).map(|i| (2 as T).pow(i)).collect();

    let mut ts = vec![
        (9, increasing_linear_mix::<H, 1> as fn(_)),
        (9, increasing_random_mix::<H, 1> as fn(_)),
        (1, random_mix::<H, 0> as fn(_)),
    ];
    if !increasing {
        ts.extend_from_slice(&[(1, random_mix::<H, 0> as fn(_))]);
        ts.extend_from_slice(&[(3, random_mix::<H, 1> as fn(_))]);
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
    eprintln!("QUICKHEAP");
    // bench::<QuickHeap<4, 1>>(false);
    // bench::<QuickHeap<8, 1>>(false);
    // bench::<QuickHeap<8, 3>>(false);
    bench::<QuickHeap<16, 1>>(false);
    // bench::<QuickHeap<16, 3>>(false);
    // bench::<QuickHeap<32, 1>>(false);
    // bench::<QuickHeap<32, 3>>(false);
    // bench::<QuickHeap<64, 3>>(false);

    // bench::<QuickHeap<32, 3>>(false);
    // bench::<QuickHeap<64, 3>>(false);

    // bench::<QuickHeap<16, 1>>(false); // actually slightly faster usually ??
    // bench::<QuickHeap<16, 5>>(false);

    eprintln!("BASELINE");
    bench::<BinaryHeap<Reverse<T>>>(false);

    eprintln!("DARY");
    // bench::<dary_heap::DaryHeap<Reverse<T>, 2>>(false);
    // bench::<dary_heap::DaryHeap<Reverse<T>, 4>>(false);
    bench::<dary_heap::DaryHeap<Reverse<T>, 8>>(false);
    // bench::<DaryHeap<(), T, 2>>(false);
    bench::<DaryHeap<(), T, 4>>(false);
    // bench::<DaryHeap<(), T, 8>>(false);

    eprintln!("RADIX");
    bench::<RadixHeapMap<Reverse<T>, ()>>(true);

    //
    // eprintln!("BTREES");
    // bench::<BTreeSet<T>>(false);
    // bench::<BTreeSet<Reverse<T>>>(false);
    // bench::<indexset::BTreeSet<T>>(false);
    // bench::<indexset::BTreeSet<Reverse<T>>>(false);

    // eprintln!("FANCY");
    // bench::<PairingHeap<(), T>>(false);
    // bench::<FibonacciHeap>(false); // too slow
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
        bench::<TestHeap<RadixHeapMap<Reverse<T>, ()>, BinaryHeap<Reverse<T>>>>(true);
    }
}
