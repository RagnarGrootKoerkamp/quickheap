use crate::simd;
use std::marker::PhantomData;

pub trait Partition<T, B> {
    fn partition<const MODE: usize>(
        cur_layer: &mut B,
        next_layer: &mut B,
        pivot: T,
        pivot_pos: usize,
    );
}

pub trait MultiPartition<T, B> {
    fn multi_partition<const P: usize>(layers: &mut [B], pivot: &[T], pivot_pos: [usize; P]);
}

use simd::PIVOTS_LE;
use simd::PIVOTS_LT;
use simd::SPLIT_PIVOTS_CUR_AND_NEXT;

/// Implementation for a plain vector
pub struct VecPartitioning<T, S>(PhantomData<T>, PhantomData<S>);
impl<T: PartialEq + Copy, S: simd::SimdElem<T>> VecPartitioning<T, S> {
    #[inline]
    /// Partition `cur_layer` into the `cur_layer` and `next_layer`.
    /// Elements equal to `pivot` can go either way.
    pub fn partition(cur_layer: &mut Vec<T>, next_layer: &mut Vec<T>, pivot: T, pivot_pos: usize) {
        assert!(next_layer.is_empty(), "destination layer must be empty");
        let n = cur_layer.len();

        // Reserve space in the next layer,
        // and make sure the current layer can hold a spare SIMD register.
        cur_layer.reserve(S::L);
        next_layer.reserve(n + S::L);

        unsafe { cur_layer.set_len(n + S::L) };
        unsafe { next_layer.set_len(n + S::L) };

        let n2 = n.next_multiple_of(S::L).saturating_sub(S::L);
        // let half = pivot_pos.next_multiple_of(S::L).saturating_sub(S::L);

        // Partition a list into two using SIMD.
        let mut cur_len = 0;
        let mut next_len = 0;
        let threshold = S::splat(pivot);

        let cur_layer_ptr = cur_layer.as_mut_ptr();
        let next_layer_ptr = next_layer.as_mut_ptr();

        for i in (0..n2).step_by(S::L) {
            unsafe {
                S::partition_fast::<SPLIT_PIVOTS_CUR_AND_NEXT>(
                    S::simd_from_ptr(cur_layer_ptr.add(i)),
                    threshold,
                    pivot_pos,
                    cur_layer_ptr,
                    &mut cur_len,
                    next_layer_ptr,
                    &mut next_len,
                );
            }
        }

        /*
        for i in (0..half).step_by(S::L) {
            unsafe {
                S::partition_fast::<SPLIT>(
                    S::simd_from_ptr(cur_layer_ptr.add(i)),
                    threshold,
                    pivot_pos,
                    cur_layer_ptr,
                    &mut cur_len,
                    next_layer_ptr,
                    &mut next_len,
                );
            }
        }

        for i in (half..n2).step_by(S::L) {
            unsafe {
                S::partition_fast::<PIVOTS_LT>(
                    S::simd_from_ptr(cur_layer_ptr.add(i)),
                    threshold,
                    pivot_pos,
                    cur_layer_ptr,
                    &mut cur_len,
                    next_layer_ptr,
                    &mut next_len,
                );
            }
        } */

        if n2 < n {
            unsafe {
                S::partition_slow::<SPLIT_PIVOTS_CUR_AND_NEXT>(
                    S::simd_from_ptr(cur_layer_ptr.add(n2)),
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
    /// Partition matching key/value layers.
    ///
    /// `cur_key_layer` and `cur_val_layer` must have equal length.
    pub fn partition_kv(
        cur_key_layer: &mut Vec<T>,
        cur_val_layer: &mut Vec<T>,
        next_key_layer: &mut Vec<T>,
        next_val_layer: &mut Vec<T>,
        pivot: T,
        pivot_pos: usize,
    ) {
        assert_eq!(
            cur_key_layer.len(),
            cur_val_layer.len(),
            "key/value lengths must match"
        );
        assert!(
            next_key_layer.is_empty() && next_val_layer.is_empty(),
            "destination layers must be empty"
        );
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
                    S::simd_from_ptr(cur_key_layer_ptr.add(i)),
                    S::simd_from_ptr(cur_val_layer_ptr.add(i)),
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
                    S::simd_from_ptr(cur_key_layer_ptr.add(n2)),
                    S::simd_from_ptr(cur_val_layer_ptr.add(n2)),
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
