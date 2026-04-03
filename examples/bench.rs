use quickheap::workloads::*;
use quickheap::*;
use std::any::type_name;
use std::any::TypeId;

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
        (1, heapsort::<T, H> as fn(_)),
        (9, constant_size::<T, H> as fn(_)),
        (3, increasing_random_mix::<T, H, 1> as fn(_)),
        // (3, increasing_linear_mix::<T, H, 1> as fn(_)),
        // (1, random_mix::<T, H, 0> as fn(_)),
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

fn test<T: Elem + 'static>() {
    eprintln!("QUICKHEAP");
    // bench::<T, scalar_quickheap::ScalarQuickHeap<T, 1>>(false);
    // bench::<T, scalar_quickheap::ScalarQuickHeap<T, 3>>(false);
    // if TypeId::of::<T>() == TypeId::of::<i32>() {
    //     bench::<i32, simd_quickheap::SimdQuickHeap<8, 3>>(false);
    //     bench::<i32, simd_quickheap::SimdQuickHeap<16, 1>>(false);
    // }

    eprintln!("BASELINE");
    bench::<T, impls::BinaryHeap<T>>(false);

    eprintln!("DARY");
    bench::<T, impls::DaryHeap<T, 2>>(false);
    bench::<T, impls::DaryHeap<T, 4>>(false);
    bench::<T, impls::DaryHeap<T, 8>>(false);
    bench::<T, impls::DaryHeap<T, 16>>(false);
    bench::<T, impls::OrxDaryHeap<T, 2>>(false);
    bench::<T, impls::OrxDaryHeap<T, 4>>(false);
    bench::<T, impls::OrxDaryHeap<T, 8>>(false);
    bench::<T, impls::OrxDaryHeap<T, 16>>(false);

    eprintln!("Amortized");
    bench::<T, impls::PairingHeap<T>>(false);

    if TypeId::of::<T>() == TypeId::of::<i32>() {
        bench::<i32, impls::FibonacciHeap>(false);
    }
    bench::<T, impls::WeakHeap<T>>(false);

    eprintln!("Monotone");
    bench::<T, impls::RadixHeap<T>>(true);

    eprintln!("Set");
    bench::<T, impls::BTreeSet<T>>(true);
    bench::<T, impls::RevBTreeSet<T>>(true);
    bench::<T, impls::IndexSetBTreeSet<T>>(true);
    bench::<T, impls::IndexSetRevBTreeSet<T>>(true);
}

fn main() {
    test::<i32>();
    test::<i64>();
}
