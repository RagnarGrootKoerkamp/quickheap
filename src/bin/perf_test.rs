use std::io::{Write, stdout};

use quickheap::{
    Avx2, ConfigurableSimdQuickHeap,
    buckets::{
        list_block_bucket::ListBlockBucket, vec_block_bucket::VecBlockBucket, vec_bucket::VecBucket,
    },
    pivot_strategies::MedianOfM,
    rebalancing_strategies::NoRebalancing,
};
use rand::RngExt;

fn main() {
    print!("Run perf test...");
    let _ = stdout().flush();

    let mut rng = rand::rng();

    let mut q = <ConfigurableSimdQuickHeap<
        i32,
        // ListBlockBucket<i64, 128>,
        // VecBlockBucket<i64, 128>,
        VecBucket<i32>,
        Avx2,
        MedianOfM<3>,
        NoRebalancing,
    >>::default();

    for _ in 0..1000000 {
        let n: i32 = rng.random();
        q.push(n)
    }

    for _ in 0..1000000 {
        q.pop().unwrap();
    }

    println!("done.")
}
