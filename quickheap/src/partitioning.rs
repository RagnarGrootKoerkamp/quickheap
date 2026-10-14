use crate::simd;
use std::marker::PhantomData;

pub struct VecPartitioning<T, S>(PhantomData<T>, PhantomData<S>);
pub trait Partition<T, B> {
    fn partition<const MODE: usize>(
        cur_layer: &mut B,
        next_layer: &mut B,
        pivot: T,
        pivot_pos: usize,
    );
}

use simd::SPLIT_PIVOTS_CUR_AND_NEXT;

impl<T: PartialEq + Copy, S: simd::SimdElem<T>> VecPartitioning<T, S> {
    #[inline]
    pub fn partition(cur_layer: &mut Vec<T>, next_layer: &mut Vec<T>, pivot: T, pivot_pos: usize) {
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

        let cur_layer_ptr = cur_layer.as_mut_ptr();
        let next_layer_ptr = next_layer.as_mut_ptr();

        for i in (0..n2).step_by(S::L) {
            unsafe {
                S::partition_fast::<SPLIT_PIVOTS_CUR_AND_NEXT>(
                    S::simd_from_slice(cur_layer.get_unchecked(i..i + S::L)),
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
                S::partition_slow::<SPLIT_PIVOTS_CUR_AND_NEXT>(
                    S::simd_from_slice(cur_layer.get_unchecked(n2..n)),
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

    #[inline]
    pub fn partition_kv(
        cur_key_layer: &mut Vec<T>,
        cur_val_layer: &mut Vec<T>,
        next_key_layer: &mut Vec<T>,
        next_val_layer: &mut Vec<T>,
        pivot: T,
        pivot_pos: usize,
    ) {
        let n = cur_key_layer.len();

        // Reserve space in the next layer,
        // and make sure the current layer can hold a spare SIMD register.
        cur_key_layer.reserve(S::L);
        cur_val_layer.reserve(S::L);
        next_key_layer.reserve(n + S::L);
        next_val_layer.reserve(n + S::L);

        unsafe {
            cur_key_layer.set_len(n + S::L);
            cur_val_layer.set_len(n + S::L);
            next_key_layer.set_len(n + S::L);
            next_val_layer.set_len(n + S::L)
        };

        let n2 = n.next_multiple_of(S::L).saturating_sub(S::L);

        // Partition a list into two using SIMD.
        let mut cur_len = 0;
        let mut next_len = 0;
        let threshold = S::splat(pivot);

        let cur_key_layer_ptr = cur_key_layer.as_mut_ptr();
        let next_key_layer_ptr = next_key_layer.as_mut_ptr();
        let cur_val_layer_ptr = cur_val_layer.as_mut_ptr();
        let next_val_layer_ptr = next_val_layer.as_mut_ptr();

        for i in (0..n2).step_by(S::L) {
            unsafe {
                S::partition_fast_kv(
                    S::simd_from_slice(cur_key_layer.get_unchecked(i..i + S::L)),
                    S::simd_from_slice(cur_val_layer.get_unchecked(i..i + S::L)),
                    threshold,
                    pivot_pos,
                    cur_key_layer_ptr,
                    cur_val_layer_ptr,
                    &mut cur_len,
                    next_key_layer_ptr,
                    next_val_layer_ptr,
                    &mut next_len,
                );
            }
        }

        if n2 < n {
            unsafe {
                S::partition_slow_kv(
                    S::simd_from_slice(cur_key_layer.get_unchecked(n2..n)),
                    S::simd_from_slice(cur_val_layer.get_unchecked(n2..n)),
                    S::splat(S::from_usize(n - n2)),
                    threshold,
                    pivot_pos,
                    cur_key_layer_ptr,
                    cur_val_layer_ptr,
                    &mut cur_len,
                    next_key_layer_ptr,
                    next_val_layer_ptr,
                    &mut next_len,
                );
            }
        }

        unsafe {
            cur_key_layer.set_len(cur_len);
            cur_val_layer.set_len(cur_len);
            next_key_layer.set_len(next_len);
            next_val_layer.set_len(next_len);
        }
    }
}
