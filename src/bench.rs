use std::{any::type_name, hint::black_box};

use rand::{rng, seq::SliceRandom};

use super::*;

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

fn push_random<H: Heap>(n: T) {
    let mut h = H::default();
    let mut rng = fastrand::Rng::new();
    for i in 0..n {
        h.push(rng.u32(..));
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
        h.push(rng.u32(..));
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
        h.push(rng.u32(..));
    }
    for i in n..2 * n {
        h.push(rng.u32(..));
        h.pop();
    }
    black_box(h);
}

fn random_mix<H: Heap, const K: usize>(n: T) {
    let mut h = H::default();
    let mut rng = fastrand::Rng::new();
    for _ in 0..n {
        h.push(rng.u32(..));
        for _ in 0..K {
            h.pop();
            h.push(rng.u32(..));
        }
    }
    for _ in 0..n {
        h.pop();
        for _ in 0..K {
            h.push(rng.u32(..));
            h.pop();
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
                h.push(rng.u32(..));
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
    // eprintln!("ts: {ts:?}");
    let t = ts.iter().min().unwrap();
    (t.as_secs_f64() / n as f64) * 1e9f64
}

pub fn bench<H: Heap>() {
    let ns: Vec<_> = (10..=22)
        // .rev()
        .step_by(2)
        .map(|i| (2 as T).pow(i))
        .collect();

    // let ns = [1 << 20];

    for n in ns {
        let ts = [
            push_linear::<H> as fn(_),
            linear::<H> as fn(_),
            push_linear_rev::<H> as fn(_),
            linear_rev::<H> as fn(_),
            push_random::<H> as fn(_),
            // push_real_random::<H>(n) as fn(_),
            natural::<H> as fn(_),
            random::<H> as fn(_),
            random_alternate::<H> as fn(_),
            natural::<H> as fn(_),
            linear_mix::<H, 0> as fn(_),
            linear_mix::<H, 1> as fn(_),
            linear_mix::<H, 4> as fn(_),
            random_mix::<H, 0> as fn(_),
            random_mix::<H, 1> as fn(_),
            random_mix::<H, 4> as fn(_),
            // linear_rev_mix::<H> as fn(_),
        ];
        eprint!("{:<70}  {n:>10}", type_name::<H>());
        for t in ts {
            let t = time(n, t);
            eprint!("{t:>9.2}");
        }
        eprintln!();
    }
    eprintln!();
}
