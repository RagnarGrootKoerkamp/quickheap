use quickheap::SimdQuickHeap;
use std::{hint::black_box, time::Instant};

fn main() {
    const N: usize = 1000000;
    const C: u64 = 1 << 20;
    let mut heap = SimdQuickHeap::<u64>::default();
    let mut last = 0;
    for _ in 0..N {
        heap.push(last + rand::random_range(0..C));
        heap.push(last + rand::random_range(0..C));
        last = heap.pop().unwrap();
    }

    let start = Instant::now();
    for _ in 0..10 * N {
        last = heap.pop().unwrap();
        heap.push(last + rand::random_range(0..C));
    }
    let elapsed = start.elapsed();
    black_box(&heap);
    assert_eq!(heap.len(), N);
    println!(
        "constant_size_increasing u64 n={N}: {:.3} ms, {:.2} ns/pop+push",
        elapsed.as_secs_f64() * 1e3,
        elapsed.as_secs_f64() * 1e9 / (10 * N) as f64,
    );
}
