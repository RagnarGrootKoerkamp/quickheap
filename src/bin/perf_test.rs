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
    println!("Run perf test");

    let mut rng = rand::rng();

    let mut q = <ConfigurableSimdQuickHeap<
        i64,
        // ListBlockBucket<i64, 128>,
        VecBlockBucket<i64, 128>,
        // VecBucket<i64>,
        Avx2,
        MedianOfM<3>,
        NoRebalancing<128>,
    >>::default();

    for _ in 0..1000000 {
        let n: i64 = rng.random();
        q.push(n)
    }

    for _ in 0..1000000 {
        q.pop().unwrap();
    }
}
