use quickheap::partitioning::{MultiPartition, Partition};
use quickheap::simd;

use crate::buckets::BlockedBucket;
use crate::buckets::Bucket;
use crate::buckets::FlatBucket;

use std::fmt::Debug;
use std::marker::PhantomData;
use std::ptr;

pub struct FlatPartitioning<S, const K: usize, const CAP: usize>(PhantomData<S>);
pub struct BlockedPartitioning<S, const K: usize, const CAP: usize>(PhantomData<S>);

#[inline(always)]
fn prefetch_read<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        _mm_prefetch(p as *const i8, _MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{0}]",
            in(reg) p,
            options(nostack, preserves_flags, readonly)
        );
    }
}

#[cold]
#[inline(never)]
unsafe fn spill<
    T: Copy + PartialEq,
    B: BlockedBucket<T, K, CAP>,
    const K: usize,
    const CAP: usize,
>(
    bucket: &mut B,
    ptr: &mut *mut T,
    len: usize,
) -> usize {
    let over = len - K; // < L, lives in the slack region
    let old = *ptr;
    bucket.write_next();
    *ptr = bucket.active_write();
    unsafe {
        ptr::copy_nonoverlapping(old.add(K), *ptr, over);
    }

    over
}

impl<
    T: PartialEq + Copy,
    B: Bucket<T, K, CAP> + FlatBucket<T>,
    S: simd::SimdElem<T>,
    const K: usize,
    const CAP: usize,
> Partition<T, B> for FlatPartitioning<S, K, CAP>
{
    #[inline(always)]
    fn partition<const MODE: usize>(
        cur_layer: &mut B,
        next_layer: &mut B,
        pivot: T,
        pivot_pos: usize,
    ) {
        let n = cur_layer.len();

        // Reserve space in the next layer,
        // and make sure the current layer can hold a spare SIMD register.
        cur_layer.reserve(S::L);
        next_layer.reserve(n + S::L);

        unsafe { cur_layer.set_len(n + S::L) };
        unsafe { next_layer.set_len(n + S::L) };

        let n2 = n.next_multiple_of(S::L).saturating_sub(S::L);

        // Partition a list into two using SIMD.
        let mut cur_len = 0;
        let mut next_len = 0;
        let threshold = S::splat(pivot);

        let cur_layer_ptr = cur_layer.write_ptr();
        let next_layer_ptr = next_layer.write_ptr();

        for i in (0..n2).step_by(S::L) {
            unsafe {
                S::partition_fast::<MODE>(
                    S::simd_from_slice(cur_layer.get_unchecked(i, S::L)),
                    threshold,
                    pivot_pos,
                    cur_layer_ptr,
                    &mut cur_len,
                    next_layer_ptr,
                    &mut next_len,
                );
            }
        }

        if n2 < n {
            unsafe {
                S::partition_slow::<MODE>(
                    S::simd_from_slice(cur_layer.get_unchecked(n2, n - n2)),
                    S::splat(S::from_usize(n - n2)),
                    threshold,
                    pivot_pos,
                    cur_layer_ptr,
                    &mut cur_len,
                    next_layer_ptr,
                    &mut next_len,
                );
            }
        }

        unsafe {
            cur_layer.set_len(cur_len);
            next_layer.set_len(next_len);
        }
    }
}

impl<
    T: PartialEq + Copy + Debug,
    B: Bucket<T, K, CAP> + FlatBucket<T>,
    S: simd::SimdElem<T>,
    const K: usize,
    const CAP: usize,
> MultiPartition<T, B> for FlatPartitioning<S, K, CAP>
{
    #[inline(always)]
    fn multi_partition<const P: usize>(layers: &mut [B], pivots: &[T], pivot_pos: [usize; P]) {
        let n = layers[0].len();
        debug_assert!(layers.len() == P);
        debug_assert!(P > 1);

        // Reserve space in the next layers,
        // and make sure the current layer can hold a spare SIMD register.

        layers[0].reserve(S::L);
        unsafe {
            layers[0].set_len(n + S::L);
        }

        for i in 0..P {
            layers[i].reserve(n + S::L);
            unsafe { layers[i].set_len(n + S::L) };
        }

        let n2 = n.next_multiple_of(S::L).saturating_sub(S::L);

        // Partition a list into P using SIMD.

        let mut lens = [0; P];
        let thresholds: [S::Simd; P] = core::array::from_fn(|i| S::splat(pivots[i.min(P - 2)]));

        let mut ptrs: [*mut T; P] = [ptr::null_mut(); P];
        for i in 0..P {
            ptrs[i] = layers[i].write_ptr();
        }

        for i in (0..n2).step_by(S::L) {
            unsafe {
                S::partition_multi_fast::<P>(
                    S::simd_from_slice(layers[0].get_unchecked(i, S::L)),
                    &thresholds[0..P - 1],
                    pivot_pos[0],
                    &mut ptrs,
                    &mut lens,
                );
            }
        }

        if n2 < n {
            unsafe {
                S::partition_multi_slow::<P>(
                    S::simd_from_slice(layers[0].get_unchecked(n2, n - n2)),
                    S::splat(S::from_usize(n - n2)),
                    &thresholds[0..P - 1],
                    pivot_pos[0],
                    &mut ptrs,
                    &mut lens,
                );
            }
        }

        for i in 0..P {
            unsafe {
                layers[i].set_len(lens[i]);
            }
        }
    }
}

impl<
    T: PartialEq + Copy,
    B: Bucket<T, K, CAP> + BlockedBucket<T, K, CAP>,
    S: simd::SimdElem<T>,
    const K: usize,
    const CAP: usize,
> Partition<T, B> for BlockedPartitioning<S, K, CAP>
{
    #[inline]
    fn partition<const MODE: usize>(
        cur_layer: &mut B,
        next_layer: &mut B,
        pivot: T,
        pivot_pos: usize,
    ) {
        let n = cur_layer.len();

        cur_layer.reset_iters();

        let n2 = n.next_multiple_of(S::L).saturating_sub(S::L);

        // Partition a list into two using SIMD.
        let threshold = S::splat(pivot);

        let mut cur_write_ptr = cur_layer.active_write();
        let mut next_write_ptr = next_layer.active_write();
        let mut cur_len = 0;
        let mut next_len = 0;

        let full_blocks = n / K;

        for _ in 0..full_blocks {
            let src = cur_layer.next_read_block();

            /*
            let next_src = cur_layer.peek_read_block();
            if !next_src.is_null() {
                // Prefetch first 2 cache lines; the HW prefetcher continues the stream
                prefetch_read(next_src);
                prefetch_read(unsafe { (next_src as *const u8).add(64) });
            } */

            let mut i = 0;
            while i < K {
                unsafe {
                    let v = S::simd_from_ptr(src.add(i));
                    S::partition_fast::<MODE>(
                        v,
                        threshold,
                        pivot_pos,
                        cur_write_ptr,
                        &mut cur_len,
                        next_write_ptr,
                        &mut next_len,
                    );
                    if cur_len > K {
                        cur_len = spill(cur_layer, &mut cur_write_ptr, cur_len);
                    }
                    if next_len > K {
                        next_len = spill(next_layer, &mut next_write_ptr, next_len);
                    }
                }
                i += S::L;
            }
        }

        let n_rem = n - full_blocks * K;

        if n_rem > 0 {
            let n2r = n_rem.next_multiple_of(S::L).saturating_sub(S::L);
            for _ in (0..n2r).step_by(S::L) {
                unsafe {
                    S::partition_fast::<MODE>(
                        S::simd_from_slice(cur_layer.get_next_unchecked(S::L)),
                        threshold,
                        pivot_pos,
                        cur_write_ptr,
                        &mut cur_len,
                        next_write_ptr,
                        &mut next_len,
                    );

                    if cur_len > K {
                        cur_len = spill(cur_layer, &mut cur_write_ptr, cur_len);
                    }

                    if next_len > K {
                        next_len = spill(next_layer, &mut next_write_ptr, next_len);
                    }
                }
            }

            if n2r < n_rem {
                unsafe {
                    S::partition_slow::<MODE>(
                        S::simd_from_slice(cur_layer.get_next_unchecked(n_rem - n2r)),
                        S::splat(S::from_usize(n - n2)),
                        threshold,
                        pivot_pos,
                        cur_write_ptr,
                        &mut cur_len,
                        next_write_ptr,
                        &mut next_len,
                    );

                    if cur_len > K {
                        cur_len = spill(cur_layer, &mut cur_write_ptr, cur_len);
                    }

                    if next_len > K {
                        next_len = spill(next_layer, &mut next_write_ptr, next_len);
                    }
                }
            }
        }

        cur_layer.set_last_block_len(cur_len);
        next_layer.set_last_block_len(next_len);
    }
}
