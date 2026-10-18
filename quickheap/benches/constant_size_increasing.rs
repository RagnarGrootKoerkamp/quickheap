use quickheap::{KVSimdQuickHeap, SimdQuickHeap};
use std::{hint::black_box, time::Instant};

fn quickheap_u32() {
    const N: usize = 1000000;
    const C: u32 = 1 << 20;
    let mut heap = SimdQuickHeap::<u32>::default();
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
        "constant_size_increasing u32 n={N}: {:.3} ms, {:.2} ns/pop+push",
        elapsed.as_secs_f64() * 1e3,
        elapsed.as_secs_f64() * 1e9 / (10 * N) as f64,
    );
}

fn quickheap_u64() {
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

fn kv_quickheap_u32() {
    const N: usize = 1000000;
    const C: u32 = 1 << 9;
    let mut heap = KVSimdQuickHeap::<u32>::new();
    let mut last = 0;
    for _ in 0..N {
        let key = last + rand::random_range(0..C);
        heap.push(key, key);
        let key = last + rand::random_range(0..C);
        heap.push(key, key);
        last = black_box(heap.pop().unwrap()).0;
    }

    let start = Instant::now();
    for _ in 0..N {
        last = black_box(heap.pop().unwrap()).0;
        let key = last + rand::random_range(0..C);
        heap.push(key, key);
    }
    let elapsed = start.elapsed();
    black_box(&heap);
    assert_eq!(heap.len(), N);
    println!(
        "constant_size_increasing_kv u32 n={N}: {:.3} ms, {:.2} ns/pop+push",
        elapsed.as_secs_f64() * 1e3,
        elapsed.as_secs_f64() * 1e9 / (10 * N) as f64,
    );
}

fn kv_quickheap_u64() {
    const N: usize = 1000000;
    const C: u64 = 1 << 20;
    let mut heap = KVSimdQuickHeap::<u64>::new();
    let mut last = 0;
    for _ in 0..N {
        let key = last + rand::random_range(0..C);
        heap.push(key, key);
        let key = last + rand::random_range(0..C);
        heap.push(key, key);
        last = black_box(heap.pop().unwrap()).0;
    }

    let start = Instant::now();
    for _ in 0..10 * N {
        last = black_box(heap.pop().unwrap()).0;
        let key = last + rand::random_range(0..C);
        heap.push(key, key);
    }
    let elapsed = start.elapsed();
    black_box(&heap);
    assert_eq!(heap.len(), N);
    println!(
        "constant_size_increasing_kv u64 n={N}: {:.3} ms, {:.2} ns/pop+push",
        elapsed.as_secs_f64() * 1e3,
        elapsed.as_secs_f64() * 1e9 / (10 * N) as f64,
    );
}

fn main() {
    quickheap_u32();
    quickheap_u64();
    kv_quickheap_u64();
    kv_quickheap_u32();
}
