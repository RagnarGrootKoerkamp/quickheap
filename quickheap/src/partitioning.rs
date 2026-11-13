use crate::ElemT;
use crate::simd;
use std::marker::PhantomData;
use std::ptr;

const MAX_BUCKETS: usize = 16;

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
pub struct VecPartitioning<T: ElemT, S>(PhantomData<T>, PhantomData<S>);
impl<T: ElemT, S: simd::SimdElem<T>> VecPartitioning<T, S> {
    #[inline]
    /// Partition `cur_layer` into the `cur_layer` and `next_layer`.
    /// Elements equal to `pivot` can go either way.
    pub fn partition<const MODE: usize>(
        cur_layer: &mut Vec<T>,
        next_layer: &mut Vec<T>,
        pivot: T,
        pivot_pos: usize,
    ) {
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
                S::partition_fast::<MODE>(
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

        if n2 < n {
            unsafe {
                S::partition_slow::<MODE>(
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
        pivot_val: T,
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
        let threshold_val = S::splat(pivot_val);

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
                    threshold_val,
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
                    threshold_val,
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

    fn radix_mask<const B: usize>(layer: &Vec<T>) -> usize {
        let mut all_and = layer[0];
        let mut all_or = layer[0];

        for i in 1..layer.len() {
            all_and &= layer[i];
            all_or |= layer[i];
        }

        let diff = all_and ^ all_or;

        if diff == T::zero() {
            return 0;
        }

        let bits = T::zero().count_zeros() as usize;
        let idx = bits - 1 - diff.leading_zeros() as usize;

        let shift = (idx + 1).saturating_sub(B);

        shift
    }

    pub fn radix_partition<const B: usize>(layers: &mut [Vec<T>]) -> Vec<T> {
        let n = layers[0].len();

        let num_buckets: usize = 2usize.pow(B as u32);
        assert!(num_buckets <= MAX_BUCKETS);

        let shift = Self::radix_mask::<B>(&layers[0]); // TODO: Use simd for this
        if shift == 0 {
            return vec![];
        }

        let v0 = layers[0][0];
        let flip = T::min_value();
        let bits = T::zero().count_zeros() as usize;

        let simd_flip = S::splat(flip);
        let simd_shift = S::splat(T::from(shift).unwrap());
        let simd_nb_mask = S::splat(T::from(num_buckets - 1).unwrap());

        debug_assert!(layers.len() == num_buckets);
        debug_assert!(num_buckets >= 2);

        // Reserve space in the next layers,
        // and make sure the current layer can hold a spare SIMD register.
        layers[0].reserve(S::L);
        unsafe {
            layers[0].set_len(n + S::L);
        }

        for i in 0..num_buckets {
            layers[i].reserve(n + S::L);
            unsafe { layers[i].set_len(n + S::L) };
        }

        let n2 = n.next_multiple_of(S::L).saturating_sub(S::L);

        // Partition a list into `num_buckets` using SIMD and a radix style partitioning.
        let mut lens = [0; MAX_BUCKETS];
        let mut ptrs: [*mut T; MAX_BUCKETS] = [ptr::null_mut(); MAX_BUCKETS];

        for k in 0..num_buckets {
            ptrs[k] = layers[num_buckets - 1 - k].as_mut_ptr();
        }

        for i in (0..n2).step_by(S::L) {
            unsafe {
                S::partition_radix_fast::<B>(
                    simd_flip,
                    simd_shift,
                    simd_nb_mask,
                    S::simd_from_slice(layers[0].get_unchecked(i..i + S::L)),
                    &mut ptrs[0..MAX_BUCKETS],
                    &mut lens[0..MAX_BUCKETS],
                );
            }
        }

        if n2 < n {
            unsafe {
                S::partition_radix_slow::<B>(
                    simd_flip,
                    simd_shift,
                    simd_nb_mask,
                    S::simd_from_slice(layers[0].get_unchecked(n2..n)),
                    S::splat(S::from_usize(n - n2)),
                    &mut ptrs[0..MAX_BUCKETS],
                    &mut lens[0..MAX_BUCKETS],
                );
            }
        }

        for k in 0..num_buckets {
            unsafe {
                layers[num_buckets - 1 - k].set_len(lens[k]);
            }
        }

        let top = shift + B;
        let base = if top >= bits {
            T::zero()
        } else {
            ((v0 ^ flip) >> top) << top
        };

        let mut pivots = Vec::with_capacity(num_buckets - 1);
        for j in 0..num_buckets - 1 {
            let k = num_buckets - 1 - j;
            pivots.push((base | (T::from(k).unwrap() << shift)) ^ flip);
        }
        pivots
    }

    #[inline(always)]
    pub fn multi_partition<const P: usize>(
        layers: &mut [Vec<T>],
        pivots: &[T],
        pivot_pos: [usize; P],
    ) -> [usize; P] {
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
            ptrs[i] = layers[i].as_mut_ptr();
        }

        let half = pivot_pos[0].next_multiple_of(S::L).saturating_sub(S::L);

        for i in (0..half).step_by(S::L) {
            unsafe {
                S::partition_multi_fast::<P, { simd::ABOVE_GT }>(
                    S::simd_from_slice(layers[0].get_unchecked(i..i + S::L)),
                    &thresholds[0..P - 1],
                    // pivot_pos[0],
                    &mut ptrs,
                    &mut lens,
                );
            }
        }

        for i in (half..n2).step_by(S::L) {
            unsafe {
                S::partition_multi_fast::<P, { simd::ABOVE_GE }>(
                    S::simd_from_slice(layers[0].get_unchecked(i..i + S::L)),
                    &thresholds[0..P - 1],
                    // pivot_pos[0],
                    &mut ptrs,
                    &mut lens,
                );
            }
        }

        if n2 < n {
            unsafe {
                S::partition_multi_slow::<P>(
                    S::simd_from_slice(layers[0].get_unchecked(n2..n)),
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

        lens
    }
}
