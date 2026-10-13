use crate::buckets;

use quickheap::simd::SimdElem;

#[inline(never)]
pub fn position_min_bucket<
    T: Copy + Ord,
    S: SimdElem<T>,
    B: buckets::Bucket<T, K, CAP>,
    const K: usize,
    const CAP: usize,
>(
    v: &mut B,
) -> usize {
    // TODO:
    v.min().1
}
