//! Paper's MonotoneConstantSize workload, with pre-generated increments.
use quickheap::{KvSimdQuickHeap, SimdQuickHeap};
use std::{hint::black_box, time::Instant};

const N: usize = 1 << 20;
const OPERATIONS: usize = 10 * N;
const REPEATS: usize = 3;

fn benchmark<H>(
    name: &str,
    stride: u64,
    new: impl Fn() -> H,
    push: impl Fn(&mut H, u64),
    pop: impl Fn(&mut H) -> u64,
) {
    let mut times = [0.0; REPEATS];
    for repeat in 0..REPEATS {
        let mut heap = new();
        let mut rng = fastrand::Rng::new();
        let mut last = 0;
        for _ in 0..N + 50 {
            push(&mut heap, last + rng.u64(0..stride));
            last = pop(&mut heap);
            push(&mut heap, last + rng.u64(0..stride));
        }
        let values: Vec<_> = (0..OPERATIONS).map(|_| rng.u64(0..stride)).collect();
        let start = Instant::now();
        for value in values {
            let last = pop(&mut heap);
            push(&mut heap, last + value);
        }
        drop(black_box(heap));
        times[repeat] = start.elapsed().as_secs_f64() * 1e9 / (OPERATIONS as f64);
    }
    times.sort_by(f64::total_cmp);
    println!(
        "constant_size_increasing {name} n={N}: {:.2} ns/pop+push",
        times[REPEATS / 2],
    );
}

fn main() {
    benchmark(
        "u32",
        1 << 24,
        SimdQuickHeap::<u32>::default,
        |heap, key| heap.push(key.try_into().unwrap()),
        |heap| heap.pop().unwrap().into(),
    );
    benchmark(
        "u64",
        1 << 48,
        SimdQuickHeap::<u64>::default,
        |heap, key| heap.push(key),
        |heap| heap.pop().unwrap(),
    );
    benchmark(
        "(u32,u32)",
        1 << 24,
        KvSimdQuickHeap::<u32>::default,
        |heap, key| {
            let key = u32::try_from(key).unwrap();
            heap.push(key, !key);
        },
        |heap| heap.pop().unwrap().0.into(),
    );
    benchmark(
        "(u64,u64)",
        1 << 48,
        KvSimdQuickHeap::<u64>::default,
        |heap, key| heap.push(key, !key),
        |heap| heap.pop().unwrap().0,
    );
}
