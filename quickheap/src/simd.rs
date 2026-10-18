use std::{fmt::Debug, mem::transmute};

use wide::{CmpEq, CmpGt, CmpLt};

/// Marker type selecting the AVX2 (256-bit) SIMD backend for [`ConfigurableSimdQuickHeap`].
///
/// [`ConfigurableSimdQuickHeap`]: crate::ConfigurableSimdQuickHeap
pub struct Avx2;

/// Marker type selecting the AVX-512 (512-bit) SIMD backend for [`ConfigurableSimdQuickHeap`].
///
/// The const generic `CS` controls how compressed results are written:
/// - `Avx512<true>` (the default) uses `_mm512_mask_compressstoreu_epi*`.
/// - `Avx512<true>` uses `_mm512_maskz_compress_epi*` followed by `_mm512_storeu_si512`.
///
/// Requires compiling with `RUSTFLAGS="-C target-feature=+avx512f"` and the `avx512` feature.
///
/// [`ConfigurableSimdQuickHeap`]: crate::ConfigurableSimdQuickHeap
pub struct Avx512<const CS: bool = false>;

/// Marker type selecting the NEON (128-bit) SIMD backend.
#[cfg(target_arch = "aarch64")]
pub struct Neon;

pub const ONLY_PIVOTS_TO_CUR: usize = 0;
pub const ONLY_PIVOTS_TO_NEXT: usize = 1;
pub const SPLIT_PIVOTS_CUR_AND_NEXT: usize = 2;

/// Byte shuffles for `vqtbl1q_u8`: table[keep_mask] moves kept lanes to the front.
#[cfg(target_arch = "aarch64")]
const fn build_neon_shuf<const LANES: usize, const N: usize>() -> [[u8; 16]; N] {
    let bytes = 16 / LANES;
    let mut t = [[0xFFu8; 16]; N];
    let mut m = 0;
    while m < N {
        let mut n = 0;
        let mut lane = 0;
        while lane < LANES {
            if (m >> lane) & 1 == 1 {
                let mut b = 0;
                while b < bytes {
                    t[m][n * bytes + b] = (lane * bytes + b) as u8;
                    b += 1;
                }
                n += 1;
            }
            lane += 1;
        }
        m += 1;
    }
    t
}

#[cfg(target_arch = "aarch64")]
pub(crate) static NEON_SHUF32: [[u8; 16]; 16] = build_neon_shuf::<4, 16>();
#[cfg(target_arch = "aarch64")]
pub(crate) static NEON_SHUF64: [[u8; 16]; 4] = build_neon_shuf::<2, 4>();

/// A SIMD backend strategy for element type `T`.
pub trait SimdElem<T>: 'static {
    /// Number of SIMD lanes.
    const L: usize;
    /// Maximum value for `T`.
    const MAX: T;
    /// The SIMD vector type (e.g. `i32x8` or `i64x4`).
    type Simd: Copy + Debug;

    fn splat(v: T) -> Self::Simd;

    /// # Safety
    /// `slice` must have at least `L` elements accessible (may read past `slice.len()`).
    unsafe fn simd_from_slice(slice: &[T]) -> Self::Simd;

    unsafe fn simd_from_ptr(ptr: *const T) -> Self::Simd;

    /// Returns bitmask where bit `i` = `a[i] < b[i]`.
    fn simd_lt_bitmask(a: Self::Simd, b: Self::Simd) -> u64;
    /// Returns a SIMD register `[0, 1, 2, ..., L-1]`.
    fn lane_indices() -> Self::Simd;
    fn from_usize(n: usize) -> T;
    fn wrapping_add_one(t: T) -> T;

    /// Bit i = lane i of `keys` goes to the "small" side (below threshold, with the alternating-lane tie rule).
    fn small_mask<const MODE: usize>(
        keys: Self::Simd,
        threshold: Self::Simd,
        pivot_pos: usize,
    ) -> u64;

    /*
    fn simd_min(a: Self::Simd, b: Self::Simd) -> Self::Simd;
    fn simd_eq_bitmask(a: Self::Simd, b: Self::Simd) -> u64;
    fn reduce_min(a: Self::Simd) -> T;
    fn simd_max(a: Self::Simd, b: Self::Simd) -> Self::Simd;
    fn reduce_max(a: Self::Simd) -> T;
     */

    /// Partition all `L` lanes of `vals` against `threshold`.
    /// # Safety
    /// `cur_write` and `next_write` must have at least `L` elements of capacity beyond their current write index.
    unsafe fn partition_fast<const MODE: usize>(
        vals: Self::Simd,
        threshold: Self::Simd,
        pivot_pos: usize,
        cur_write: *mut T,
        cur_write_cnt: &mut usize,
        next_write: *mut T,
        next_write_cnt: &mut usize,
    );

    /// Like `partition_fast`, but only the first `len` lanes are in range.
    /// # Safety
    /// Same capacity requirements as `partition_fast`.
    unsafe fn partition_slow<const MODE: usize>(
        vals: Self::Simd,
        len: Self::Simd,
        threshold: Self::Simd,
        pivot_pos: usize,
        cur_write: *mut T,
        cur_write_cnt: &mut usize,
        next_write: *mut T,
        next_write_cnt: &mut usize,
    );

    /// Partition all `L` lanes of `keys` against `threshold`; `vals` follow their keys.
    /// # Safety
    /// All four write pointers must have at least `L` elements of capacity beyond their write index.
    unsafe fn partition_fast_kv(
        keys: Self::Simd,
        vals: Self::Simd,
        threshold: Self::Simd,
        pivot_pos: usize,
        cur_keys: *mut T,
        cur_vals: *mut T,
        cur_write_cnt: &mut usize,
        next_keys: *mut T,
        next_vals: *mut T,
        next_write_cnt: &mut usize,
    );

    /// Like `partition_fast_kv`, but only the first `len` lanes are in range.
    /// # Safety
    /// Same capacity requirements as `partition_fast_kv`.
    unsafe fn partition_slow_kv(
        keys: Self::Simd,
        vals: Self::Simd,
        len: Self::Simd,
        threshold: Self::Simd,
        pivot_pos: usize,
        cur_keys: *mut T,
        cur_vals: *mut T,
        cur_write_cnt: &mut usize,
        next_keys: *mut T,
        next_vals: *mut T,
        next_write_cnt: &mut usize,
    );

    /// Bucket of a lane = number of pivots < value, so pivots.len() == K-1.
    /// Appends each lane to writes[bucket] and bumps cnts[bucket].
    /// Contract: every writes[b] needs >= LANES elements of slack
    /// (full-vector store, only the first popcount lanes are kept).
    unsafe fn partition_multi_fast<const K: usize>(
        vals: Self::Simd,
        pivots: &[Self::Simd],
        rand: usize,
        writes: &mut [*mut T; K],
        cnts: &mut [usize; K],
    );

    /// Like partition_fast, but only the first `len` lanes of `vals` are valid
    /// (`len` is splatted). Used for the tail. Invalid lanes are dropped.
    unsafe fn partition_multi_slow<const K: usize>(
        vals: Self::Simd,
        len: Self::Simd,
        pivots: &[Self::Simd],
        rand: usize,
        writes: &mut [*mut T; K],
        cnts: &mut [usize; K],
    );
}

/// Classify the element t against the decreasing list of pivots.
#[inline(always)]
pub fn push_position<T: Copy + Ord, S: SimdElem<T>>(pivots: &Vec<T>, t: T) -> usize {
    if pivots.len() <= 64 {
        let t_simd = S::splat(t);

        let mut target_layer = 0;
        let mut i = 0;
        while i < pivots.len() {
            // NOTE: This reads beyond the length but within the capacity.
            let vals = unsafe { (pivots.as_ptr().add(i) as *const S::Simd).read_unaligned() };
            target_layer += S::simd_lt_bitmask(t_simd, vals).trailing_ones() as usize;
            i += S::L;
        }
        target_layer.min(pivots.len())
    } else {
        pivots
            .binary_search_by(|p| {
                if *p < t {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            })
            .unwrap_err()
    }
}

#[inline(always)]
pub fn insert_index<T: Copy + Ord, S: SimdElem<T>>(layer: &[T], elem: T) -> usize {
    let elem_simd = S::splat(elem);

    let mut i = 0;
    let mut target_idx = 0;
    while i < layer.len() {
        let vals = unsafe { (layer.as_ptr().add(i) as *const S::Simd).read_unaligned() };
        let ones = S::simd_lt_bitmask(vals, elem_simd).count_ones() as usize;

        /*
        if ones == 0 {
            return target_idx;
        } */

        target_idx += ones;
        i += S::L;
    }
    target_idx
}

#[inline(never)]
pub fn position_min<T: Copy + Ord, S: SimdElem<T>>(v: &mut Vec<T>) -> usize {
    let mut min_pos = [0; 2];
    let mut min_val = [S::MAX; 2];
    for (i, &[l, r]) in v.as_chunks::<2>().0.iter().enumerate() {
        if l < min_val[0] {
            min_val[0] = l;
            min_pos[0] = i * 2;
        }
        if r < min_val[1] {
            min_val[1] = r;
            min_pos[1] = i * 2 + 1;
        }
    }
    if v.len() % 2 == 1 {
        let l = *v.last().unwrap();
        if l < min_val[0] {
            min_val[0] = l;
            min_pos[0] = v.len() - 1;
        }
    }
    if min_val[0] <= min_val[1] {
        min_pos[0]
    } else {
        min_pos[1]
    }
}

/// Returns `(min, index of first occurrence)`. For an empty slice returns `(S::MAX, 0)`.
/*
#[inline(never)]
pub fn simd_min_pos<T: Copy + Ord, S: SimdElem<T>>(s: &[T]) -> (T, usize) {
    let full = s.len() / S::L * S::L;

    // Pass 1: minimum value.
    let mut acc = S::splat(S::MAX);
    let mut i = 0;
    while i < full {
        acc = S::simd_min(acc, unsafe { S::simd_from_ptr(s.as_ptr().add(i)) });
        i += S::L;
    }
    let mut min = S::reduce_min(acc);
    for &x in &s[full..] {
        if x < min {
            min = x;
        }
    }

    // Pass 2: first position equal to the minimum.
    let m = S::splat(min);
    let mut i = 0;
    while i < full {
        let mask = S::simd_eq_bitmask(unsafe { S::simd_from_ptr(s.as_ptr().add(i)) }, m);
        if mask != 0 {
            return (min, i + mask.trailing_zeros() as usize);
        }
        i += S::L;
    }
    for (j, &x) in s[full..].iter().enumerate() {
        if x == min {
            return (min, full + j);
        }
    }
    (min, 0)
} */

/*
/// Returns `(max, index of first occurrence)`, or `None` for an empty slice.
#[inline(never)]
pub fn simd_max_pos<T: Copy + Ord, S: SimdElem<T>>(s: &[T]) -> Option<(T, usize)> {
    if s.is_empty() {
        return None;
    }
    let full = s.len() / S::L * S::L;

    // Pass 1: maximum value. Seeding with s[0] avoids needing a MIN constant.
    let mut acc = S::splat(s[0]);
    let mut i = 0;
    while i < full {
        acc = S::simd_max(acc, unsafe { S::simd_from_ptr(s.as_ptr().add(i)) });
        i += S::L;
    }
    let mut max = S::reduce_max(acc);
    for &x in &s[full..] {
        if x > max {
            max = x;
        }
    }

    // Pass 2: first position equal to the maximum.
    let m = S::splat(max);
    let mut i = 0;
    while i < full {
        let mask = S::simd_eq_bitmask(unsafe { S::simd_from_ptr(s.as_ptr().add(i)) }, m);
        if mask != 0 {
            return Some((max, i + mask.trailing_zeros() as usize));
        }
        i += S::L;
    }
    for (j, &x) in s[full..].iter().enumerate() {
        if x == max {
            return Some((max, full + j));
        }
    }
    unreachable!()
}
 */
// ───────────────────────────── AVX2, 32-bit ─────────────────────────────

#[cfg(target_feature = "avx2")]
macro_rules! impl_simd_elem_32 {
    ($t:ty, $simd:ty, $alt:expr) => {
        impl SimdElem<$t> for Avx2 {
            const L: usize = 8;
            const MAX: $t = <$t>::MAX;
            type Simd = $simd;

            #[inline(always)]
            fn splat(v: $t) -> $simd {
                <$simd>::splat(v)
            }

            #[inline(always)]
            unsafe fn simd_from_slice(slice: &[$t]) -> $simd {
                unsafe { <$simd>::from(*(slice.as_ptr() as *const [$t; 8])) }
            }

            #[inline(always)]
            unsafe fn simd_from_ptr(ptr: *const $t) -> $simd {
                unsafe { <$simd>::from(*(ptr as *const [$t; 8])) }
            }

            #[inline(always)]
            fn simd_lt_bitmask(a: $simd, b: $simd) -> u64 {
                a.simd_lt(b).to_bitmask() as u64
            }

            #[inline(always)]
            fn lane_indices() -> $simd {
                <$simd>::from([0 as $t, 1, 2, 3, 4, 5, 6, 7])
            }

            #[inline(always)]
            fn from_usize(n: usize) -> $t {
                n as $t
            }

            #[inline(always)]
            fn wrapping_add_one(t: $t) -> $t {
                t.wrapping_add(1)
            }

            #[inline(always)]
            fn small_mask<const MODE: usize>(keys: $simd, threshold: $simd, rand: usize) -> u64 {
                let lanes = 0xFFu64;

                if MODE == ONLY_PIVOTS_TO_CUR {
                    return ((keys).simd_lt(threshold).to_bitmask() as u64) & lanes;
                }

                if MODE == ONLY_PIVOTS_TO_NEXT {
                    let eq = (threshold.simd_eq(keys).to_bitmask() as u64);
                    return eq & lanes;
                }

                if MODE == SPLIT_PIVOTS_CUR_AND_NEXT {
                    let lt = (threshold.simd_gt(keys).to_bitmask() as u64);
                    let eq = (threshold.simd_eq(keys).to_bitmask() as u64);
                    return (lt | (eq & EQUAL_MASKS[rand % 16])) & lanes;
                }

                unreachable!();
            }

            #[inline(always)]
            unsafe fn partition_fast<const MODE: usize>(
                vals: Self::Simd,
                threshold: Self::Simd,
                pivot_pos: usize,
                cur_write: *mut $t,
                cur_write_cnt: &mut usize,
                next_write: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::x86_64::*;
                    use std::mem::transmute;

                    let small =
                        <Self as SimdElem<$t>>::small_mask::<MODE>(vals, threshold, pivot_pos)
                            as u8;
                    let large = !small;
                    let vals: __m256i = transmute(vals);

                    let key: __m256i = transmute(crate::simd::UNIQSHUF32[small as usize]);
                    _mm256_storeu_si256(
                        cur_write.add(*cur_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *cur_write_cnt += large.count_ones() as usize;

                    let key: __m256i = transmute(crate::simd::UNIQSHUF32[large as usize]);
                    _mm256_storeu_si256(
                        next_write.add(*next_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *next_write_cnt += small.count_ones() as usize;
                }
            }

            #[inline(always)]
            unsafe fn partition_slow<const MODE: usize>(
                vals: Self::Simd,
                len: Self::Simd,
                threshold: Self::Simd,
                pivot_pos: usize,
                cur_write: *mut $t,
                cur_write_cnt: &mut usize,
                next_write: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::x86_64::*;
                    use std::mem::transmute;

                    let mut small =
                        <Self as SimdElem<$t>>::small_mask::<MODE>(vals, threshold, pivot_pos)
                            as u8;
                    let mut large = !small;
                    let in_range = len
                        .simd_gt(<Self as SimdElem<$t>>::lane_indices())
                        .to_bitmask() as u8;
                    small &= in_range;
                    large &= in_range;
                    let vals: __m256i = transmute(vals);

                    let key: __m256i = transmute(crate::simd::UNIQSHUF32[(!large) as usize]);
                    _mm256_storeu_si256(
                        cur_write.add(*cur_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *cur_write_cnt += large.count_ones() as usize;

                    let key: __m256i = transmute(crate::simd::UNIQSHUF32[(!small) as usize]);
                    _mm256_storeu_si256(
                        next_write.add(*next_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *next_write_cnt += small.count_ones() as usize;
                }
            }

            #[inline(always)]
            unsafe fn partition_fast_kv(
                keys: Self::Simd,
                vals: Self::Simd,
                threshold: Self::Simd,
                pivot_pos: usize,
                cur_keys: *mut $t,
                cur_vals: *mut $t,
                cur_write_cnt: &mut usize,
                next_keys: *mut $t,
                next_vals: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::x86_64::*;
                    use std::mem::transmute;

                    let small = <Self as SimdElem<$t>>::small_mask::<SPLIT_PIVOTS_CUR_AND_NEXT>(
                        keys, threshold, pivot_pos,
                    );
                    let large = !small;
                    let keys: __m256i = transmute(keys);
                    let vals: __m256i = transmute(vals);

                    let key: __m256i = transmute(crate::simd::UNIQSHUF32[small as usize]);
                    _mm256_storeu_si256(
                        cur_keys.add(*cur_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(keys, key),
                    );
                    _mm256_storeu_si256(
                        cur_vals.add(*cur_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *cur_write_cnt += large.count_ones() as usize;

                    let key: __m256i = transmute(crate::simd::UNIQSHUF32[large as usize]);
                    _mm256_storeu_si256(
                        next_keys.add(*next_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(keys, key),
                    );
                    _mm256_storeu_si256(
                        next_vals.add(*next_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *next_write_cnt += small.count_ones() as usize;
                }
            }

            #[inline(always)]
            unsafe fn partition_slow_kv(
                keys: Self::Simd,
                vals: Self::Simd,
                len: Self::Simd,
                threshold: Self::Simd,
                pivot_pos: usize,
                cur_keys: *mut $t,
                cur_vals: *mut $t,
                cur_write_cnt: &mut usize,
                next_keys: *mut $t,
                next_vals: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::x86_64::*;
                    use std::mem::transmute;

                    let mut small = <Self as SimdElem<$t>>::small_mask::<SPLIT_PIVOTS_CUR_AND_NEXT>(
                        keys, threshold, pivot_pos,
                    ) as u8;
                    let mut large = !small;
                    let in_range = len
                        .simd_gt(<Self as SimdElem<$t>>::lane_indices())
                        .to_bitmask() as u8;
                    small &= in_range;
                    large &= in_range;
                    let keys: __m256i = transmute(keys);
                    let vals: __m256i = transmute(vals);

                    let key: __m256i = transmute(crate::simd::UNIQSHUF32[(!large) as usize]);
                    _mm256_storeu_si256(
                        cur_keys.add(*cur_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(keys, key),
                    );
                    _mm256_storeu_si256(
                        cur_vals.add(*cur_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *cur_write_cnt += large.count_ones() as usize;

                    let key: __m256i = transmute(crate::simd::UNIQSHUF32[(!small) as usize]);
                    _mm256_storeu_si256(
                        next_keys.add(*next_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(keys, key),
                    );
                    _mm256_storeu_si256(
                        next_vals.add(*next_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *next_write_cnt += small.count_ones() as usize;
                }
            }

            avx2_partition_multi!($t, UNIQSHUF32, 0xFFu8, u8);
        }
    };
}

// ───────────────────────────── AVX2, 64-bit ─────────────────────────────

#[cfg(target_feature = "avx2")]
macro_rules! impl_simd_elem_64 {
    ($t:ty, $simd:ty, $alt:expr) => {
        impl SimdElem<$t> for Avx2 {
            const L: usize = 4;
            const MAX: $t = <$t>::MAX;
            type Simd = $simd;

            #[inline(always)]
            fn splat(v: $t) -> $simd {
                <$simd>::splat(v)
            }

            #[inline(always)]
            unsafe fn simd_from_slice(slice: &[$t]) -> $simd {
                unsafe { <$simd>::from(*(slice.as_ptr() as *const [$t; 4])) }
            }

            #[inline(always)]
            unsafe fn simd_from_ptr(ptr: *const $t) -> $simd {
                unsafe { <$simd>::from(*(ptr as *const [$t; 4])) }
            }

            #[inline(always)]
            fn simd_lt_bitmask(a: $simd, b: $simd) -> u64 {
                a.simd_lt(b).to_bitmask() as u64
            }

            #[inline(always)]
            fn lane_indices() -> $simd {
                <$simd>::from([0 as $t, 1, 2, 3])
            }

            #[inline(always)]
            fn from_usize(n: usize) -> $t {
                n as $t
            }

            #[inline(always)]
            fn wrapping_add_one(t: $t) -> $t {
                t.wrapping_add(1)
            }

            #[inline(always)]
            fn small_mask<const MODE: usize>(keys: $simd, threshold: $simd, rand: usize) -> u64 {
                let lanes = 0xFu64;

                if MODE == ONLY_PIVOTS_TO_CUR {
                    return ((keys).simd_lt(threshold).to_bitmask() as u64) & lanes;
                }

                if MODE == ONLY_PIVOTS_TO_NEXT {
                    let eq = (threshold.simd_eq(keys).to_bitmask() as u64);
                    return eq & lanes;
                }

                if MODE == SPLIT_PIVOTS_CUR_AND_NEXT {
                    let lt = (threshold.simd_gt(keys).to_bitmask() as u64);
                    let eq = (threshold.simd_eq(keys).to_bitmask() as u64);
                    return (lt | (eq & EQUAL_MASKS[rand % 16])) & lanes;
                }

                unreachable!();
            }

            #[inline(always)]
            unsafe fn partition_fast<const MODE: usize>(
                vals: $simd,
                threshold: $simd,
                pivot_pos: usize,
                cur_write: *mut $t,
                cur_write_cnt: &mut usize,
                next_write: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::x86_64::*;
                    use std::mem::transmute;

                    let small =
                        <Self as SimdElem<$t>>::small_mask::<MODE>(vals, threshold, pivot_pos)
                            as u8;
                    let large = small ^ 0xF;
                    let vals: __m256i = transmute(vals);

                    // UNIQSHUF64[k] keeps the lanes described by keep_pattern = k ^ 0xF.
                    let key: __m256i = transmute(crate::simd::UNIQSHUF64[small as usize]);
                    _mm256_storeu_si256(
                        cur_write.add(*cur_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *cur_write_cnt += large.count_ones() as usize;

                    let key: __m256i = transmute(crate::simd::UNIQSHUF64[large as usize]);
                    _mm256_storeu_si256(
                        next_write.add(*next_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *next_write_cnt += small.count_ones() as usize;
                }
            }

            #[inline(always)]
            unsafe fn partition_slow<const MODE: usize>(
                vals: $simd,
                len: $simd,
                threshold: $simd,
                pivot_pos: usize,
                cur_write: *mut $t,
                cur_write_cnt: &mut usize,
                next_write: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::x86_64::*;
                    use std::mem::transmute;

                    let in_range = (len
                        .simd_gt(<Self as SimdElem<$t>>::lane_indices())
                        .to_bitmask() as u8)
                        & 0xF;
                    let small =
                        <Self as SimdElem<$t>>::small_mask::<MODE>(vals, threshold, pivot_pos)
                            as u8
                            & in_range;
                    let large = (small ^ 0xF) & in_range;

                    let vals: __m256i = transmute(vals);
                    let key: __m256i = transmute(crate::simd::UNIQSHUF64[(large ^ 0xF) as usize]);
                    _mm256_storeu_si256(
                        cur_write.add(*cur_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *cur_write_cnt += large.count_ones() as usize;

                    let key: __m256i = transmute(crate::simd::UNIQSHUF64[(small ^ 0xF) as usize]);
                    _mm256_storeu_si256(
                        next_write.add(*next_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *next_write_cnt += small.count_ones() as usize;
                }
            }

            #[inline(always)]
            unsafe fn partition_fast_kv(
                keys: $simd,
                vals: $simd,
                threshold: $simd,
                pivot_pos: usize,
                cur_keys: *mut $t,
                cur_vals: *mut $t,
                cur_write_cnt: &mut usize,
                next_keys: *mut $t,
                next_vals: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::x86_64::*;
                    use std::mem::transmute;

                    let small = <Self as SimdElem<$t>>::small_mask::<SPLIT_PIVOTS_CUR_AND_NEXT>(
                        keys, threshold, pivot_pos,
                    ) as u8;
                    let large = small ^ 0xF;
                    let keys: __m256i = transmute(keys);
                    let vals: __m256i = transmute(vals);

                    let key: __m256i = transmute(crate::simd::UNIQSHUF64[small as usize]);
                    _mm256_storeu_si256(
                        cur_keys.add(*cur_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(keys, key),
                    );
                    _mm256_storeu_si256(
                        cur_vals.add(*cur_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *cur_write_cnt += large.count_ones() as usize;

                    let key: __m256i = transmute(crate::simd::UNIQSHUF64[large as usize]);
                    _mm256_storeu_si256(
                        next_keys.add(*next_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(keys, key),
                    );
                    _mm256_storeu_si256(
                        next_vals.add(*next_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *next_write_cnt += small.count_ones() as usize;
                }
            }

            #[inline(always)]
            unsafe fn partition_slow_kv(
                keys: $simd,
                vals: $simd,
                len: $simd,
                threshold: $simd,
                pivot_pos: usize,
                cur_keys: *mut $t,
                cur_vals: *mut $t,
                cur_write_cnt: &mut usize,
                next_keys: *mut $t,
                next_vals: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::x86_64::*;
                    use std::mem::transmute;

                    let in_range = (len
                        .simd_gt(<Self as SimdElem<$t>>::lane_indices())
                        .to_bitmask() as u8)
                        & 0xF;
                    let small = <Self as SimdElem<$t>>::small_mask::<SPLIT_PIVOTS_CUR_AND_NEXT>(
                        keys, threshold, pivot_pos,
                    ) as u8
                        & in_range;
                    let large = (small ^ 0xF) & in_range;
                    let keys: __m256i = transmute(keys);
                    let vals: __m256i = transmute(vals);

                    let key: __m256i = transmute(crate::simd::UNIQSHUF64[(large ^ 0xF) as usize]);
                    _mm256_storeu_si256(
                        cur_keys.add(*cur_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(keys, key),
                    );
                    _mm256_storeu_si256(
                        cur_vals.add(*cur_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *cur_write_cnt += large.count_ones() as usize;

                    let key: __m256i = transmute(crate::simd::UNIQSHUF64[(small ^ 0xF) as usize]);
                    _mm256_storeu_si256(
                        next_keys.add(*next_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(keys, key),
                    );
                    _mm256_storeu_si256(
                        next_vals.add(*next_write_cnt) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *next_write_cnt += small.count_ones() as usize;
                }
            }

            avx2_partition_multi!($t, UNIQSHUF64, 0xFu8, u8);
        }
    };
}

// ──────────────────────────── AVX-512, 32-bit ────────────────────────────

#[cfg(target_feature = "avx512f")]
macro_rules! impl_simd_elem_32_avx512 {
    ($t:ty, $simd:ty, $alt:expr) => {
        impl<const CS: bool> SimdElem<$t> for Avx512<CS> {
            const L: usize = 16;
            const MAX: $t = <$t>::MAX;
            type Simd = $simd;

            #[inline(always)]
            fn splat(v: $t) -> $simd {
                <$simd>::splat(v)
            }

            #[inline(always)]
            unsafe fn simd_from_slice(slice: &[$t]) -> $simd {
                unsafe { <$simd>::from(*(slice.as_ptr() as *const [$t; 16])) }
            }

            #[inline(always)]
            unsafe fn simd_from_ptr(ptr: *const $t) -> $simd {
                unsafe { <$simd>::from(*(ptr as *const [$t; 16])) }
            }

            #[inline(always)]
            fn simd_lt_bitmask(a: $simd, b: $simd) -> u64 {
                a.simd_lt(b).to_bitmask() as u64
            }

            #[inline(always)]
            fn lane_indices() -> $simd {
                <$simd>::from([0 as $t, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
            }

            #[inline(always)]
            fn from_usize(n: usize) -> $t {
                n as $t
            }

            #[inline(always)]
            fn wrapping_add_one(t: $t) -> $t {
                t.wrapping_add(1)
            }

            #[inline(always)]
            fn small_mask<const MODE: usize>(keys: $simd, threshold: $simd, rand: usize) -> u64 {
                let lanes = 0xFFFFu64;

                if MODE == ONLY_PIVOTS_TO_CUR {
                    return ((keys).simd_lt(threshold).to_bitmask() as u64) & lanes;
                }

                if MODE == ONLY_PIVOTS_TO_NEXT {
                    let eq = (threshold.simd_eq(keys).to_bitmask() as u64);
                    return eq & lanes;
                }

                if MODE == SPLIT_PIVOTS_CUR_AND_NEXT {
                    let lt = (threshold.simd_gt(keys).to_bitmask() as u64);
                    let eq = (threshold.simd_eq(keys).to_bitmask() as u64);
                    return (lt | (eq & EQUAL_MASKS[rand % 16])) & lanes;
                }

                unreachable!();
            }

            #[inline(always)]
            unsafe fn partition_fast<const MODE: usize>(
                vals: Self::Simd,
                threshold: Self::Simd,
                pivot_pos: usize,
                cur_write: *mut $t,
                cur_write_cnt: &mut usize,
                next_write: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::x86_64::*;
                    use std::mem::transmute;

                    let small =
                        <Self as SimdElem<$t>>::small_mask::<MODE>(vals, threshold, pivot_pos)
                            as u16;
                    let large: u16 = !small;
                    let vals: __m512i = transmute(vals);
                    let large_elems = large.count_ones() as usize;
                    let small_elems = small.count_ones() as usize;

                    if CS {
                        let c_curr = _mm512_maskz_compress_epi32(large, vals);
                        _mm512_storeu_si512(cur_write.add(*cur_write_cnt) as *mut __m512i, c_curr);
                        let c_next = _mm512_maskz_compress_epi32(small, vals);
                        _mm512_storeu_si512(
                            next_write.add(*next_write_cnt) as *mut __m512i,
                            c_next,
                        );
                    } else {
                        _mm512_mask_compressstoreu_epi32(
                            cur_write.add(*cur_write_cnt) as *mut i32,
                            large,
                            vals,
                        );
                        _mm512_mask_compressstoreu_epi32(
                            next_write.add(*next_write_cnt) as *mut i32,
                            small,
                            vals,
                        );
                    }
                    *cur_write_cnt += large_elems;
                    *next_write_cnt += small_elems;
                }
            }

            #[inline(always)]
            unsafe fn partition_slow<const MODE: usize>(
                vals: Self::Simd,
                len: Self::Simd,
                threshold: Self::Simd,
                pivot_pos: usize,
                cur_write: *mut $t,
                cur_write_cnt: &mut usize,
                next_write: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::x86_64::*;
                    use std::mem::transmute;

                    let in_range: u16 = len
                        .simd_gt(<Self as SimdElem<$t>>::lane_indices())
                        .to_bitmask() as u16;

                    let small =
                        <Self as SimdElem<$t>>::small_mask::<MODE>(vals, threshold, pivot_pos)
                            as u16
                            & in_range;
                    let large: u16 = !small & in_range;
                    let vals: __m512i = transmute(vals);
                    let large_elems = large.count_ones() as usize;
                    let small_elems = small.count_ones() as usize;

                    if CS {
                        let c_curr = _mm512_maskz_compress_epi32(large, vals);
                        _mm512_storeu_si512(cur_write.add(*cur_write_cnt) as *mut __m512i, c_curr);
                        let c_next = _mm512_maskz_compress_epi32(small, vals);
                        _mm512_storeu_si512(
                            next_write.add(*next_write_cnt) as *mut __m512i,
                            c_next,
                        );
                    } else {
                        _mm512_mask_compressstoreu_epi32(
                            cur_write.add(*cur_write_cnt) as *mut i32,
                            large,
                            vals,
                        );
                        _mm512_mask_compressstoreu_epi32(
                            next_write.add(*next_write_cnt) as *mut i32,
                            small,
                            vals,
                        );
                    }
                    *cur_write_cnt += large_elems;
                    *next_write_cnt += small_elems;
                }
            }

            #[inline(always)]
            unsafe fn partition_fast_kv(
                keys: Self::Simd,
                vals: Self::Simd,
                threshold: Self::Simd,
                pivot_pos: usize,
                cur_keys: *mut $t,
                cur_vals: *mut $t,
                cur_write_cnt: &mut usize,
                next_keys: *mut $t,
                next_vals: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::x86_64::*;
                    use std::mem::transmute;

                    let small = <Self as SimdElem<$t>>::small_mask::<SPLIT_PIVOTS_CUR_AND_NEXT>(
                        vals, threshold, pivot_pos,
                    ) as u16;
                    let large: u16 = !small;
                    let keys: __m512i = transmute(keys);
                    let vals: __m512i = transmute(vals);
                    let large_elems = large.count_ones() as usize;
                    let small_elems = small.count_ones() as usize;

                    if CS {
                        _mm512_storeu_si512(
                            cur_keys.add(*cur_write_cnt) as *mut __m512i,
                            _mm512_maskz_compress_epi32(large, keys),
                        );
                        _mm512_storeu_si512(
                            cur_vals.add(*cur_write_cnt) as *mut __m512i,
                            _mm512_maskz_compress_epi32(large, vals),
                        );
                        _mm512_storeu_si512(
                            next_keys.add(*next_write_cnt) as *mut __m512i,
                            _mm512_maskz_compress_epi32(small, keys),
                        );
                        _mm512_storeu_si512(
                            next_vals.add(*next_write_cnt) as *mut __m512i,
                            _mm512_maskz_compress_epi32(small, vals),
                        );
                    } else {
                        _mm512_mask_compressstoreu_epi32(
                            cur_keys.add(*cur_write_cnt) as *mut i32,
                            large,
                            keys,
                        );
                        _mm512_mask_compressstoreu_epi32(
                            cur_vals.add(*cur_write_cnt) as *mut i32,
                            large,
                            vals,
                        );
                        _mm512_mask_compressstoreu_epi32(
                            next_keys.add(*next_write_cnt) as *mut i32,
                            small,
                            keys,
                        );
                        _mm512_mask_compressstoreu_epi32(
                            next_vals.add(*next_write_cnt) as *mut i32,
                            small,
                            vals,
                        );
                    }
                    *cur_write_cnt += large_elems;
                    *next_write_cnt += small_elems;
                }
            }

            #[inline(always)]
            unsafe fn partition_slow_kv(
                keys: Self::Simd,
                vals: Self::Simd,
                len: Self::Simd,
                threshold: Self::Simd,
                pivot_pos: usize,
                cur_keys: *mut $t,
                cur_vals: *mut $t,
                cur_write_cnt: &mut usize,
                next_keys: *mut $t,
                next_vals: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::x86_64::*;
                    use std::mem::transmute;

                    let in_range: u16 = len
                        .simd_gt(<Self as SimdElem<$t>>::lane_indices())
                        .to_bitmask() as u16;

                    let small = <Self as SimdElem<$t>>::small_mask::<SPLIT_PIVOTS_CUR_AND_NEXT>(
                        vals, threshold, pivot_pos,
                    ) as u16
                        & in_range;
                    let large: u16 = (!small) & in_range;
                    let keys: __m512i = transmute(keys);
                    let vals: __m512i = transmute(vals);
                    let large_elems = large.count_ones() as usize;
                    let small_elems = small.count_ones() as usize;

                    if CS {
                        _mm512_storeu_si512(
                            cur_keys.add(*cur_write_cnt) as *mut __m512i,
                            _mm512_maskz_compress_epi32(large, keys),
                        );
                        _mm512_storeu_si512(
                            cur_vals.add(*cur_write_cnt) as *mut __m512i,
                            _mm512_maskz_compress_epi32(large, vals),
                        );
                        _mm512_storeu_si512(
                            next_keys.add(*next_write_cnt) as *mut __m512i,
                            _mm512_maskz_compress_epi32(small, keys),
                        );
                        _mm512_storeu_si512(
                            next_vals.add(*next_write_cnt) as *mut __m512i,
                            _mm512_maskz_compress_epi32(small, vals),
                        );
                    } else {
                        _mm512_mask_compressstoreu_epi32(
                            cur_keys.add(*cur_write_cnt) as *mut i32,
                            large,
                            keys,
                        );
                        _mm512_mask_compressstoreu_epi32(
                            cur_vals.add(*cur_write_cnt) as *mut i32,
                            large,
                            vals,
                        );
                        _mm512_mask_compressstoreu_epi32(
                            next_keys.add(*next_write_cnt) as *mut i32,
                            small,
                            keys,
                        );
                        _mm512_mask_compressstoreu_epi32(
                            next_vals.add(*next_write_cnt) as *mut i32,
                            small,
                            vals,
                        );
                    }
                    *cur_write_cnt += large_elems;
                    *next_write_cnt += small_elems;
                }
            }

            avx512_partition_multi!(
                $t,
                0xFFFFu16,
                u16,
                _mm512_maskz_compress_epi32,
                _mm512_mask_compressstoreu_epi32,
                i32
            );
        }
    };
}

// ──────────────────────────── AVX-512, 64-bit ────────────────────────────

#[cfg(target_feature = "avx512f")]
macro_rules! impl_simd_elem_64_avx512 {
    ($t:ty, $simd:ty, $alt:expr) => {
        impl<const CS: bool> SimdElem<$t> for Avx512<CS> {
            const L: usize = 8;
            const MAX: $t = <$t>::MAX;
            type Simd = $simd;

            #[inline(always)]
            fn splat(v: $t) -> $simd {
                <$simd>::splat(v)
            }

            #[inline(always)]
            unsafe fn simd_from_slice(slice: &[$t]) -> $simd {
                unsafe { <$simd>::from(*(slice.as_ptr() as *const [$t; 8])) }
            }

            #[inline(always)]
            unsafe fn simd_from_ptr(ptr: *const $t) -> $simd {
                unsafe { <$simd>::from(*(ptr as *const [$t; 8])) }
            }

            #[inline(always)]
            fn simd_lt_bitmask(a: $simd, b: $simd) -> u64 {
                a.simd_lt(b).to_bitmask() as u64
            }

            #[inline(always)]
            fn lane_indices() -> $simd {
                <$simd>::from([0 as $t, 1, 2, 3, 4, 5, 6, 7])
            }

            #[inline(always)]
            fn from_usize(n: usize) -> $t {
                n as $t
            }

            #[inline(always)]
            fn wrapping_add_one(t: $t) -> $t {
                t.wrapping_add(1)
            }

            #[inline(always)]
            fn small_mask<const MODE: usize>(keys: $simd, threshold: $simd, rand: usize) -> u64 {
                let lanes = 0xFFu64;

                if MODE == ONLY_PIVOTS_TO_CUR {
                    return ((keys).simd_lt(threshold).to_bitmask() as u64) & lanes;
                }

                if MODE == ONLY_PIVOTS_TO_NEXT {
                    let eq = (threshold.simd_eq(keys).to_bitmask() as u64);
                    return eq & lanes;
                }

                if MODE == SPLIT_PIVOTS_CUR_AND_NEXT {
                    let lt = (threshold.simd_gt(keys).to_bitmask() as u64);
                    let eq = (threshold.simd_eq(keys).to_bitmask() as u64);
                    return (lt | (eq & EQUAL_MASKS[rand % 16])) & lanes;
                }

                unreachable!();
            }

            #[inline(always)]
            unsafe fn partition_fast<const MODE: usize>(
                vals: Self::Simd,
                threshold: Self::Simd,
                pivot_pos: usize,
                cur_write: *mut $t,
                cur_write_cnt: &mut usize,
                next_write: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::x86_64::*;
                    use std::mem::transmute;

                    let small =
                        <Self as SimdElem<$t>>::small_mask::<MODE>(vals, threshold, pivot_pos)
                            as u8;
                    let large: u8 = !small;
                    let vals: __m512i = transmute(vals);
                    let large_elems = large.count_ones() as usize;
                    let small_elems = small.count_ones() as usize;

                    if CS {
                        let c_curr = _mm512_maskz_compress_epi64(large, vals);
                        _mm512_storeu_si512(cur_write.add(*cur_write_cnt) as *mut __m512i, c_curr);
                        let c_next = _mm512_maskz_compress_epi64(small, vals);
                        _mm512_storeu_si512(
                            next_write.add(*next_write_cnt) as *mut __m512i,
                            c_next,
                        );
                    } else {
                        _mm512_mask_compressstoreu_epi64(
                            cur_write.add(*cur_write_cnt) as *mut i64,
                            large,
                            vals,
                        );
                        _mm512_mask_compressstoreu_epi64(
                            next_write.add(*next_write_cnt) as *mut i64,
                            small,
                            vals,
                        );
                    }
                    *cur_write_cnt += large_elems;
                    *next_write_cnt += small_elems;
                }
            }

            #[inline(always)]
            unsafe fn partition_slow<const MODE: usize>(
                vals: Self::Simd,
                len: Self::Simd,
                threshold: Self::Simd,
                pivot_pos: usize,
                cur_write: *mut $t,
                cur_write_cnt: &mut usize,
                next_write: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::x86_64::*;
                    use std::mem::transmute;

                    let in_range: u8 = len
                        .simd_gt(<Self as SimdElem<$t>>::lane_indices())
                        .to_bitmask() as u8;

                    let small =
                        <Self as SimdElem<$t>>::small_mask::<MODE>(vals, threshold, pivot_pos)
                            as u8
                            & in_range;
                    let large: u8 = (!small) & in_range;
                    let vals: __m512i = transmute(vals);
                    let large_elems = large.count_ones() as usize;
                    let small_elems = small.count_ones() as usize;

                    if CS {
                        let c_curr = _mm512_maskz_compress_epi64(large, vals);
                        _mm512_storeu_si512(cur_write.add(*cur_write_cnt) as *mut __m512i, c_curr);
                        let c_next = _mm512_maskz_compress_epi64(small, vals);
                        _mm512_storeu_si512(
                            next_write.add(*next_write_cnt) as *mut __m512i,
                            c_next,
                        );
                    } else {
                        _mm512_mask_compressstoreu_epi64(
                            cur_write.add(*cur_write_cnt) as *mut i64,
                            large,
                            vals,
                        );
                        _mm512_mask_compressstoreu_epi64(
                            next_write.add(*next_write_cnt) as *mut i64,
                            small,
                            vals,
                        );
                    }
                    *cur_write_cnt += large_elems;
                    *next_write_cnt += small_elems;
                }
            }

            #[inline(always)]
            unsafe fn partition_fast_kv(
                keys: Self::Simd,
                vals: Self::Simd,
                threshold: Self::Simd,
                pivot_pos: usize,
                cur_keys: *mut $t,
                cur_vals: *mut $t,
                cur_write_cnt: &mut usize,
                next_keys: *mut $t,
                next_vals: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::x86_64::*;
                    use std::mem::transmute;

                    let small = <Self as SimdElem<$t>>::small_mask::<SPLIT_PIVOTS_CUR_AND_NEXT>(
                        vals, threshold, pivot_pos,
                    ) as u8;
                    let large: u8 = !small;
                    let keys: __m512i = transmute(keys);
                    let vals: __m512i = transmute(vals);
                    let large_elems = large.count_ones() as usize;
                    let small_elems = small.count_ones() as usize;

                    if CS {
                        _mm512_storeu_si512(
                            cur_keys.add(*cur_write_cnt) as *mut __m512i,
                            _mm512_maskz_compress_epi64(large, keys),
                        );
                        _mm512_storeu_si512(
                            cur_vals.add(*cur_write_cnt) as *mut __m512i,
                            _mm512_maskz_compress_epi64(large, vals),
                        );
                        _mm512_storeu_si512(
                            next_keys.add(*next_write_cnt) as *mut __m512i,
                            _mm512_maskz_compress_epi64(small, keys),
                        );
                        _mm512_storeu_si512(
                            next_vals.add(*next_write_cnt) as *mut __m512i,
                            _mm512_maskz_compress_epi64(small, vals),
                        );
                    } else {
                        _mm512_mask_compressstoreu_epi64(
                            cur_keys.add(*cur_write_cnt) as *mut i64,
                            large,
                            keys,
                        );
                        _mm512_mask_compressstoreu_epi64(
                            cur_vals.add(*cur_write_cnt) as *mut i64,
                            large,
                            vals,
                        );
                        _mm512_mask_compressstoreu_epi64(
                            next_keys.add(*next_write_cnt) as *mut i64,
                            small,
                            keys,
                        );
                        _mm512_mask_compressstoreu_epi64(
                            next_vals.add(*next_write_cnt) as *mut i64,
                            small,
                            vals,
                        );
                    }
                    *cur_write_cnt += large_elems;
                    *next_write_cnt += small_elems;
                }
            }

            #[inline(always)]
            unsafe fn partition_slow_kv(
                keys: Self::Simd,
                vals: Self::Simd,
                len: Self::Simd,
                threshold: Self::Simd,
                pivot_pos: usize,
                cur_keys: *mut $t,
                cur_vals: *mut $t,
                cur_write_cnt: &mut usize,
                next_keys: *mut $t,
                next_vals: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::x86_64::*;
                    use std::mem::transmute;

                    let in_range: u8 = len
                        .simd_gt(<Self as SimdElem<$t>>::lane_indices())
                        .to_bitmask() as u8;

                    let small = <Self as SimdElem<$t>>::small_mask::<SPLIT_PIVOTS_CUR_AND_NEXT>(
                        keys, threshold, pivot_pos,
                    ) as u8
                        & in_range;
                    let large: u8 = (!small) & in_range;
                    let keys: __m512i = transmute(keys);
                    let vals: __m512i = transmute(vals);
                    let large_elems = large.count_ones() as usize;
                    let small_elems = small.count_ones() as usize;

                    if CS {
                        _mm512_storeu_si512(
                            cur_keys.add(*cur_write_cnt) as *mut __m512i,
                            _mm512_maskz_compress_epi64(large, keys),
                        );
                        _mm512_storeu_si512(
                            cur_vals.add(*cur_write_cnt) as *mut __m512i,
                            _mm512_maskz_compress_epi64(large, vals),
                        );
                        _mm512_storeu_si512(
                            next_keys.add(*next_write_cnt) as *mut __m512i,
                            _mm512_maskz_compress_epi64(small, keys),
                        );
                        _mm512_storeu_si512(
                            next_vals.add(*next_write_cnt) as *mut __m512i,
                            _mm512_maskz_compress_epi64(small, vals),
                        );
                    } else {
                        _mm512_mask_compressstoreu_epi64(
                            cur_keys.add(*cur_write_cnt) as *mut i64,
                            large,
                            keys,
                        );
                        _mm512_mask_compressstoreu_epi64(
                            cur_vals.add(*cur_write_cnt) as *mut i64,
                            large,
                            vals,
                        );
                        _mm512_mask_compressstoreu_epi64(
                            next_keys.add(*next_write_cnt) as *mut i64,
                            small,
                            keys,
                        );
                        _mm512_mask_compressstoreu_epi64(
                            next_vals.add(*next_write_cnt) as *mut i64,
                            small,
                            vals,
                        );
                    }
                    *cur_write_cnt += large_elems;
                    *next_write_cnt += small_elems;
                }
            }

            avx512_partition_multi!(
                $t,
                0xFFu8,
                u8,
                _mm512_maskz_compress_epi64,
                _mm512_mask_compressstoreu_epi64,
                i64
            );
        }
    };
}

// ───────────────────────────── NEON, 32-bit ─────────────────────────────

#[cfg(target_arch = "aarch64")]
macro_rules! impl_simd_elem_32_neon {
    ($t:ty, $simd:ty, $alt:expr) => {
        impl SimdElem<$t> for Neon {
            const L: usize = 4;
            const MAX: $t = <$t>::MAX;
            type Simd = $simd;

            #[inline(always)]
            fn splat(v: $t) -> $simd {
                <$simd>::splat(v)
            }

            #[inline(always)]
            unsafe fn simd_from_slice(slice: &[$t]) -> $simd {
                unsafe { <$simd>::from(*(slice.as_ptr() as *const [$t; 4])) }
            }

            #[inline(always)]
            unsafe fn simd_from_ptr(ptr: *const $t) -> $simd {
                unsafe { <$simd>::from(*(ptr as *const [$t; 4])) }
            }

            #[inline(always)]
            fn simd_lt_bitmask(a: $simd, b: $simd) -> u64 {
                a.simd_lt(b).to_bitmask() as u64
            }

            #[inline(always)]
            fn lane_indices() -> $simd {
                <$simd>::from([0 as $t, 1, 2, 3])
            }

            #[inline(always)]
            fn from_usize(n: usize) -> $t {
                n as $t
            }

            #[inline(always)]
            fn wrapping_add_one(t: $t) -> $t {
                t.wrapping_add(1)
            }

            #[inline(always)]
            fn small_mask<const MODE: usize>(keys: $simd, threshold: $simd, rand: usize) -> u64 {
                let lanes = 0xFu64;

                if MODE == ONLY_PIVOTS_TO_CUR {
                    return ((keys).simd_lt(threshold).to_bitmask() as u64) & lanes;
                }

                if MODE == ONLY_PIVOTS_TO_NEXT {
                    let eq = (threshold.simd_eq(keys).to_bitmask() as u64);
                    return eq & lanes;
                }

                if MODE == SPLIT_PIVOTS_CUR_AND_NEXT {
                    let lt = (threshold.simd_gt(keys).to_bitmask() as u64);
                    let eq = (threshold.simd_eq(keys).to_bitmask() as u64);
                    return (lt | (eq & EQUAL_MASKS[rand % 16])) & lanes;
                }

                unreachable!();
            }

            /*
            #[inline(always)]
            fn simd_min(a: $simd, b: $simd) -> $simd {
                a.min(b)
            }

            #[inline(always)]
            fn simd_eq_bitmask(a: $simd, b: $simd) -> u64 {
                a.simd_eq(b).to_bitmask() as u64
            }

            #[inline(always)]
            fn reduce_min(a: $simd) -> $t {
                a.to_array().into_iter().min().unwrap()
            }

            #[inline(always)]
            fn simd_max(a: $simd, b: $simd) -> $simd {
                a.max(b)
            }

            #[inline(always)]
            fn reduce_max(a: $simd) -> $t {
                a.to_array().into_iter().max().unwrap()
            } */

            #[inline(always)]
            unsafe fn partition_fast<const MODE: usize>(
                vals: $simd,
                threshold: $simd,
                pivot_pos: usize,
                cur_write: *mut $t,
                cur_write_cnt: &mut usize,
                next_write: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::aarch64::*;
                    use std::mem::transmute;

                    let small: u8 =
                        <Self as SimdElem<$t>>::small_mask::<MODE>(vals, threshold, pivot_pos)
                            as u8;
                    let large: u8 = small ^ 0xF;
                    let v: uint8x16_t = transmute(vals);

                    let key = vld1q_u8(NEON_SHUF32[large as usize].as_ptr());
                    vst1q_u8(cur_write.add(*cur_write_cnt) as *mut u8, vqtbl1q_u8(v, key));
                    *cur_write_cnt += large.count_ones() as usize;

                    let key = vld1q_u8(NEON_SHUF32[small as usize].as_ptr());
                    vst1q_u8(
                        next_write.add(*next_write_cnt) as *mut u8,
                        vqtbl1q_u8(v, key),
                    );
                    *next_write_cnt += small.count_ones() as usize;
                }
            }

            #[inline(always)]
            unsafe fn partition_slow<const MODE: usize>(
                vals: $simd,
                len: $simd,
                threshold: $simd,
                pivot_pos: usize,
                cur_write: *mut $t,
                cur_write_cnt: &mut usize,
                next_write: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::aarch64::*;
                    use std::mem::transmute;

                    let in_range: u8 = (len
                        .simd_gt(<Self as SimdElem<$t>>::lane_indices())
                        .to_bitmask() as u8)
                        & 0xF;

                    let small: u8 =
                        <Self as SimdElem<$t>>::small_mask::<MODE>(vals, threshold, pivot_pos)
                            as u8
                            & in_range;
                    let large = (small ^ 0xF) & in_range;
                    let v: uint8x16_t = transmute(vals);

                    let key = vld1q_u8(NEON_SHUF32[large as usize].as_ptr());
                    vst1q_u8(cur_write.add(*cur_write_cnt) as *mut u8, vqtbl1q_u8(v, key));
                    *cur_write_cnt += large.count_ones() as usize;

                    let key = vld1q_u8(NEON_SHUF32[small as usize].as_ptr());
                    vst1q_u8(
                        next_write.add(*next_write_cnt) as *mut u8,
                        vqtbl1q_u8(v, key),
                    );
                    *next_write_cnt += small.count_ones() as usize;
                }
            }

            #[inline(always)]
            unsafe fn partition_fast_kv(
                keys: $simd,
                vals: $simd,
                threshold: $simd,
                pivot_pos: usize,
                cur_keys: *mut $t,
                cur_vals: *mut $t,
                cur_write_cnt: &mut usize,
                next_keys: *mut $t,
                next_vals: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::aarch64::*;
                    use std::mem::transmute;

                    let small = <Self as SimdElem<$t>>::small_mask::<SPLIT_PIVOTS_CUR_AND_NEXT>(
                        keys, threshold, pivot_pos,
                    ) as u8;
                    let large: u8 = small ^ 0xF;
                    let k: uint8x16_t = transmute(keys);
                    let v: uint8x16_t = transmute(vals);

                    let key = vld1q_u8(NEON_SHUF32[large as usize].as_ptr());
                    vst1q_u8(cur_keys.add(*cur_write_cnt) as *mut u8, vqtbl1q_u8(k, key));
                    vst1q_u8(cur_vals.add(*cur_write_cnt) as *mut u8, vqtbl1q_u8(v, key));
                    *cur_write_cnt += large.count_ones() as usize;

                    let key = vld1q_u8(NEON_SHUF32[small as usize].as_ptr());
                    vst1q_u8(
                        next_keys.add(*next_write_cnt) as *mut u8,
                        vqtbl1q_u8(k, key),
                    );
                    vst1q_u8(
                        next_vals.add(*next_write_cnt) as *mut u8,
                        vqtbl1q_u8(v, key),
                    );
                    *next_write_cnt += small.count_ones() as usize;
                }
            }

            #[inline(always)]
            unsafe fn partition_slow_kv(
                keys: $simd,
                vals: $simd,
                len: $simd,
                threshold: $simd,
                pivot_pos: usize,
                cur_keys: *mut $t,
                cur_vals: *mut $t,
                cur_write_cnt: &mut usize,
                next_keys: *mut $t,
                next_vals: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::aarch64::*;
                    use std::mem::transmute;

                    let in_range: u8 = (len
                        .simd_gt(<Self as SimdElem<$t>>::lane_indices())
                        .to_bitmask() as u8)
                        & 0xF;

                    let small = <Self as SimdElem<$t>>::small_mask::<SPLIT_PIVOTS_CUR_AND_NEXT>(
                        keys, threshold, pivot_pos,
                    ) as u8
                        & in_range;
                    let large = (small ^ 0xF) & in_range;
                    let k: uint8x16_t = transmute(keys);
                    let v: uint8x16_t = transmute(vals);

                    let key = vld1q_u8(NEON_SHUF32[large as usize].as_ptr());
                    vst1q_u8(cur_keys.add(*cur_write_cnt) as *mut u8, vqtbl1q_u8(k, key));
                    vst1q_u8(cur_vals.add(*cur_write_cnt) as *mut u8, vqtbl1q_u8(v, key));
                    *cur_write_cnt += large.count_ones() as usize;

                    let key = vld1q_u8(NEON_SHUF32[small as usize].as_ptr());
                    vst1q_u8(
                        next_keys.add(*next_write_cnt) as *mut u8,
                        vqtbl1q_u8(k, key),
                    );
                    vst1q_u8(
                        next_vals.add(*next_write_cnt) as *mut u8,
                        vqtbl1q_u8(v, key),
                    );
                    *next_write_cnt += small.count_ones() as usize;
                }
            }

            neon_partition_multi!($t, NEON_SHUF32, 0xFu8);
        }
    };
}

// ───────────────────────────── NEON, 64-bit ─────────────────────────────

#[cfg(target_arch = "aarch64")]
macro_rules! impl_simd_elem_64_neon {
    ($t:ty, $simd:ty, $alt:expr) => {
        impl SimdElem<$t> for Neon {
            const L: usize = 2;
            const MAX: $t = <$t>::MAX;
            type Simd = $simd;

            #[inline(always)]
            fn splat(v: $t) -> $simd {
                <$simd>::splat(v)
            }

            #[inline(always)]
            unsafe fn simd_from_slice(slice: &[$t]) -> $simd {
                unsafe { <$simd>::from(*(slice.as_ptr() as *const [$t; 2])) }
            }

            #[inline(always)]
            unsafe fn simd_from_ptr(ptr: *const $t) -> $simd {
                unsafe { <$simd>::from(*(ptr as *const [$t; 2])) }
            }

            #[inline(always)]
            fn simd_lt_bitmask(a: $simd, b: $simd) -> u64 {
                a.simd_lt(b).to_bitmask() as u64
            }

            #[inline(always)]
            fn lane_indices() -> $simd {
                <$simd>::from([0 as $t, 1])
            }

            #[inline(always)]
            fn from_usize(n: usize) -> $t {
                n as $t
            }

            #[inline(always)]
            fn wrapping_add_one(t: $t) -> $t {
                t.wrapping_add(1)
            }

            #[inline(always)]
            fn small_mask<const MODE: usize>(keys: $simd, threshold: $simd, rand: usize) -> u64 {
                let lanes = 0xFu64;

                if MODE == ONLY_PIVOTS_TO_CUR {
                    return ((keys).simd_lt(threshold).to_bitmask() as u64) & lanes;
                }

                if MODE == ONLY_PIVOTS_TO_NEXT {
                    let eq = (threshold.simd_eq(keys).to_bitmask() as u64);
                    return eq & lanes;
                }

                if MODE == SPLIT_PIVOTS_CUR_AND_NEXT {
                    let lt = (threshold.simd_gt(keys).to_bitmask() as u64);
                    let eq = (threshold.simd_eq(keys).to_bitmask() as u64);
                    return (lt | (eq & EQUAL_MASKS[rand % 16])) & lanes;
                }

                unreachable!();

                /*
                if threshold != <$simd>::splat(<$t>::MAX) {

                    // (
                    //    (threshold + $alt).simd_gt(keys).to_bitmask() as u64,
                    //) & lanes
                } else {
                    let lt = (threshold.simd_gt(keys).to_bitmask() as u64) & lanes;
                    let eq = (keys.simd_eq(threshold).to_bitmask() as u64) & lanes;
                    lt | (eq & 0b1010)
                }
                 */
            }

            /*
            #[inline(always)]
            fn simd_min(a: $simd, b: $simd) -> $simd {
                a.min(b)
            }

            #[inline(always)]
            fn simd_eq_bitmask(a: $simd, b: $simd) -> u64 {
                a.simd_eq(b).to_bitmask() as u64
            }

            #[inline(always)]
            fn reduce_min(a: $simd) -> $t {
                a.to_array().into_iter().min().unwrap()
            }

            #[inline(always)]
            fn simd_max(a: $simd, b: $simd) -> $simd {
                a.max(b)
            }

            #[inline(always)]
            fn reduce_max(a: $simd) -> $t {
                a.to_array().into_iter().max().unwrap()
            } */

            #[inline(always)]
            unsafe fn partition_fast<const MODE: usize>(
                vals: $simd,
                threshold: $simd,
                pivot_pos: usize,
                cur_write: *mut $t,
                cur_write_cnt: &mut usize,
                next_write: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::aarch64::*;
                    use std::mem::transmute;

                    let small =
                        <Self as SimdElem<$t>>::small_mask::<MODE>(vals, threshold, pivot_pos)
                            as u8;
                    let large: u8 = small ^ 0x3;
                    let v: uint8x16_t = transmute(vals);

                    let key = vld1q_u8(NEON_SHUF64[large as usize].as_ptr());
                    vst1q_u8(cur_write.add(*cur_write_cnt) as *mut u8, vqtbl1q_u8(v, key));
                    *cur_write_cnt += large.count_ones() as usize;

                    let key = vld1q_u8(NEON_SHUF64[small as usize].as_ptr());
                    vst1q_u8(
                        next_write.add(*next_write_cnt) as *mut u8,
                        vqtbl1q_u8(v, key),
                    );
                    *next_write_cnt += small.count_ones() as usize;
                }
            }

            #[inline(always)]
            unsafe fn partition_slow<const MODE: usize>(
                vals: $simd,
                len: $simd,
                threshold: $simd,
                pivot_pos: usize,
                cur_write: *mut $t,
                cur_write_cnt: &mut usize,
                next_write: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::aarch64::*;
                    use std::mem::transmute;

                    let in_range: u8 = (len
                        .simd_gt(<Self as SimdElem<$t>>::lane_indices())
                        .to_bitmask() as u8)
                        & 0x3;

                    let small =
                        <Self as SimdElem<$t>>::small_mask::<MODE>(vals, threshold, pivot_pos)
                            as u8
                            & in_range;
                    let large = (small ^ 0x3) & in_range;
                    let v: uint8x16_t = transmute(vals);

                    let key = vld1q_u8(NEON_SHUF64[large as usize].as_ptr());
                    vst1q_u8(cur_write.add(*cur_write_cnt) as *mut u8, vqtbl1q_u8(v, key));
                    *cur_write_cnt += large.count_ones() as usize;

                    let key = vld1q_u8(NEON_SHUF64[small as usize].as_ptr());
                    vst1q_u8(
                        next_write.add(*next_write_cnt) as *mut u8,
                        vqtbl1q_u8(v, key),
                    );
                    *next_write_cnt += small.count_ones() as usize;
                }
            }

            #[inline(always)]
            unsafe fn partition_fast_kv(
                keys: $simd,
                vals: $simd,
                threshold: $simd,
                pivot_pos: usize,
                cur_keys: *mut $t,
                cur_vals: *mut $t,
                cur_write_cnt: &mut usize,
                next_keys: *mut $t,
                next_vals: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::aarch64::*;
                    use std::mem::transmute;

                    let small: u8 = <Self as SimdElem<$t>>::small_mask::<SPLIT_PIVOTS_CUR_AND_NEXT>(
                        keys, threshold, pivot_pos,
                    ) as u8;
                    let large: u8 = small ^ 0x3;
                    let k: uint8x16_t = transmute(keys);
                    let v: uint8x16_t = transmute(vals);

                    let key = vld1q_u8(NEON_SHUF64[large as usize].as_ptr());
                    vst1q_u8(cur_keys.add(*cur_write_cnt) as *mut u8, vqtbl1q_u8(k, key));
                    vst1q_u8(cur_vals.add(*cur_write_cnt) as *mut u8, vqtbl1q_u8(v, key));
                    *cur_write_cnt += large.count_ones() as usize;

                    let key = vld1q_u8(NEON_SHUF64[small as usize].as_ptr());
                    vst1q_u8(
                        next_keys.add(*next_write_cnt) as *mut u8,
                        vqtbl1q_u8(k, key),
                    );
                    vst1q_u8(
                        next_vals.add(*next_write_cnt) as *mut u8,
                        vqtbl1q_u8(v, key),
                    );
                    *next_write_cnt += small.count_ones() as usize;
                }
            }

            #[inline(always)]
            unsafe fn partition_slow_kv(
                keys: $simd,
                vals: $simd,
                len: $simd,
                threshold: $simd,
                pivot_pos: usize,
                cur_keys: *mut $t,
                cur_vals: *mut $t,
                cur_write_cnt: &mut usize,
                next_keys: *mut $t,
                next_vals: *mut $t,
                next_write_cnt: &mut usize,
            ) {
                unsafe {
                    use core::arch::aarch64::*;
                    use std::mem::transmute;

                    let in_range: u8 = (len
                        .simd_gt(<Self as SimdElem<$t>>::lane_indices())
                        .to_bitmask() as u8)
                        & 0x3;

                    let small: u8 = <Self as SimdElem<$t>>::small_mask::<SPLIT_PIVOTS_CUR_AND_NEXT>(
                        keys, threshold, pivot_pos,
                    ) as u8
                        & in_range;
                    let large = (small ^ 0x3) & in_range;
                    let k: uint8x16_t = transmute(keys);
                    let v: uint8x16_t = transmute(vals);

                    let key = vld1q_u8(NEON_SHUF64[large as usize].as_ptr());
                    vst1q_u8(cur_keys.add(*cur_write_cnt) as *mut u8, vqtbl1q_u8(k, key));
                    vst1q_u8(cur_vals.add(*cur_write_cnt) as *mut u8, vqtbl1q_u8(v, key));
                    *cur_write_cnt += large.count_ones() as usize;

                    let key = vld1q_u8(NEON_SHUF64[small as usize].as_ptr());
                    vst1q_u8(
                        next_keys.add(*next_write_cnt) as *mut u8,
                        vqtbl1q_u8(k, key),
                    );
                    vst1q_u8(
                        next_vals.add(*next_write_cnt) as *mut u8,
                        vqtbl1q_u8(v, key),
                    );
                    *next_write_cnt += small.count_ones() as usize;
                }
            }

            neon_partition_multi!($t, NEON_SHUF64, 0x3u8);
        }
    };
}

/// Multiway partition for avx2, DESCENDING buckets:
/// bucket 0 holds the largest elements, bucket K-1 the smallest.
/// `pivots` must be sorted descending (pivots[0] is the largest), len K-1, splatted.
/// Bucket b = (x > pivots[b]) & !(x > pivots[b-1]); lanes equal to a pivot
/// are split between the two adjacent buckets by EQUAL_MASKS[rand % 16].
/// Each writes[b] needs >= L elements of slack (full 16-byte store).
#[cfg(target_feature = "avx2")]
macro_rules! avx2_partition_multi {
    ($t:ty, $shuf:ident, $full:expr, $mt:ty) => {
        #[inline(always)]
        unsafe fn partition_multi_fast<const K: usize>(
            vals: Self::Simd,
            pivots: &[Self::Simd],
            rand: usize,
            writes: &mut [*mut $t; K],
            cnts: &mut [usize; K],
        ) {
            unsafe {
                use core::arch::x86_64::*;
                use std::mem::transmute;

                debug_assert_eq!(pivots.len() + 1, K);
                let tie: $mt = (EQUAL_MASKS[rand % 16] as $mt) & $full;
                let above = |p: Self::Simd| -> $mt {
                    let gt = (vals.simd_gt(p).to_bitmask() as $mt) & $full;
                    let eq = (vals.simd_eq(p).to_bitmask() as $mt) & $full;
                    gt | (eq & tie)
                };
                let v: __m256i = transmute(vals);

                let mut prev: $mt = 0;
                for b in 0..K {
                    let next: $mt = if b + 1 < K { above(pivots[b]) } else { $full };
                    let m = next & !prev;
                    let key: __m256i = transmute($shuf[((!m) & $full) as usize]);
                    _mm256_storeu_si256(
                        writes[b].add(cnts[b]) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(v, key),
                    );
                    cnts[b] += m.count_ones() as usize;
                    prev = next;
                }
            }
        }

        #[inline(always)]
        unsafe fn partition_multi_slow<const K: usize>(
            vals: Self::Simd,
            len: Self::Simd,
            pivots: &[Self::Simd],
            rand: usize,
            writes: &mut [*mut $t; K],
            cnts: &mut [usize; K],
        ) {
            unsafe {
                use core::arch::x86_64::*;
                use std::mem::transmute;

                debug_assert_eq!(pivots.len() + 1, K);
                let in_range: $mt = (len
                    .simd_gt(<Self as SimdElem<$t>>::lane_indices())
                    .to_bitmask() as $mt)
                    & $full;
                let tie: $mt = (EQUAL_MASKS[rand % 16] as $mt) & $full;
                let above = |p: Self::Simd| -> $mt {
                    let gt = (vals.simd_gt(p).to_bitmask() as $mt) & $full;
                    let eq = (vals.simd_eq(p).to_bitmask() as $mt) & $full;
                    gt | (eq & tie)
                };
                let v: __m256i = transmute(vals);

                let mut prev: $mt = 0;
                for b in 0..K {
                    let next: $mt = if b + 1 < K { above(pivots[b]) } else { $full };
                    let m = next & !prev & in_range;
                    let key: __m256i = transmute($shuf[((!m) & $full) as usize]);
                    _mm256_storeu_si256(
                        writes[b].add(cnts[b]) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(v, key),
                    );
                    cnts[b] += m.count_ones() as usize;
                    prev = next;
                }
            }
        }
    };
}

/// Multiway partition for avx512, DESCENDING buckets:
/// bucket 0 holds the largest elements, bucket K-1 the smallest.
/// `pivots` must be sorted descending (pivots[0] is the largest), len K-1, splatted.
/// Bucket b = (x > pivots[b]) & !(x > pivots[b-1]); lanes equal to a pivot
/// are split between the two adjacent buckets by EQUAL_MASKS[rand % 16].
/// Each writes[b] needs >= L elements of slack (full 16-byte store).
#[cfg(target_feature = "avx512f")]
macro_rules! avx512_partition_multi {
    // $mt: u16 (32-bit) or u8 (64-bit); $cz/$cs: compress intrinsics; $ct: i32/i64
    ($t:ty, $full:expr, $mt:ty, $cz:ident, $cs:ident, $ct:ty) => {
        #[inline(always)]
        unsafe fn partition_multi_fast<const K: usize>(
            vals: Self::Simd,
            pivots: &[Self::Simd],
            rand: usize,
            writes: &mut [*mut $t; K],
            cnts: &mut [usize; K],
        ) {
            unsafe {
                use core::arch::x86_64::*;
                use std::mem::transmute;

                debug_assert_eq!(pivots.len() + 1, K);
                let tie: $mt = (EQUAL_MASKS[rand % 16] as $mt) & $full;
                let above = |p: Self::Simd| -> $mt {
                    let gt = (vals.simd_gt(p).to_bitmask() as $mt) & $full;
                    let eq = (vals.simd_eq(p).to_bitmask() as $mt) & $full;
                    gt | (eq & tie)
                };
                let v: __m512i = transmute(vals);

                let mut prev: $mt = 0;
                for b in 0..K {
                    let next: $mt = if b + 1 < K { above(pivots[b]) } else { $full };
                    let m = next & !prev;
                    let dst = writes[b].add(cnts[b]);
                    if CS {
                        _mm512_storeu_si512(dst as *mut __m512i, $cz(m, v));
                    } else {
                        $cs(dst as *mut $ct, m, v);
                    }
                    cnts[b] += m.count_ones() as usize;
                    prev = next;
                }
            }
        }

        #[inline(always)]
        unsafe fn partition_multi_slow<const K: usize>(
            vals: Self::Simd,
            len: Self::Simd,
            pivots: &[Self::Simd],
            rand: usize,
            writes: &mut [*mut $t; K],
            cnts: &mut [usize; K],
        ) {
            unsafe {
                use core::arch::x86_64::*;
                use std::mem::transmute;

                debug_assert_eq!(pivots.len() + 1, K);
                let in_range: $mt = (len
                    .simd_gt(<Self as SimdElem<$t>>::lane_indices())
                    .to_bitmask() as $mt)
                    & $full;
                let tie: $mt = (EQUAL_MASKS[rand % 16] as $mt) & $full;
                let above = |p: Self::Simd| -> $mt {
                    let gt = (vals.simd_gt(p).to_bitmask() as $mt) & $full;
                    let eq = (vals.simd_eq(p).to_bitmask() as $mt) & $full;
                    gt | (eq & tie)
                };
                let v: __m512i = transmute(vals);

                let mut prev: $mt = 0;
                for b in 0..K {
                    let next: $mt = if b + 1 < K { above(pivots[b]) } else { $full };
                    let m = next & !prev & in_range;
                    let dst = writes[b].add(cnts[b]);
                    if CS {
                        _mm512_storeu_si512(dst as *mut __m512i, $cz(m, v));
                    } else {
                        $cs(dst as *mut $ct, m, v);
                    }
                    cnts[b] += m.count_ones() as usize;
                    prev = next;
                }
            }
        }
    };
}

/// Multiway partition for NEON, DESCENDING buckets:
/// bucket 0 holds the largest elements, bucket K-1 the smallest.
/// `pivots` must be sorted descending (pivots[0] is the largest), len K-1, splatted.
/// Bucket b = (x > pivots[b]) & !(x > pivots[b-1]); lanes equal to a pivot
/// are split between the two adjacent buckets by EQUAL_MASKS[rand % 16].
/// Each writes[b] needs >= L elements of slack (full 16-byte store).
#[cfg(target_arch = "aarch64")]
macro_rules! neon_partition_multi {
    ($t:ty, $shuf:ident, $full:expr) => {
        #[inline(always)]
        unsafe fn partition_multi_fast<const K: usize>(
            vals: Self::Simd,
            pivots: &[Self::Simd],
            rand: usize,
            writes: &mut [*mut $t; K],
            cnts: &mut [usize; K],
        ) {
            unsafe {
                use core::arch::aarch64::*;
                use std::mem::transmute;

                debug_assert_eq!(pivots.len() + 1, K);
                let tie: u8 = (EQUAL_MASKS[rand % 16] as u8) & $full;
                // Bit i = lane i is above pivot p (greater, or equal and picked by tie).
                let above = |p: Self::Simd| -> u8 {
                    let gt = (vals.simd_gt(p).to_bitmask() as u8) & $full;
                    let eq = (vals.simd_eq(p).to_bitmask() as u8) & $full;
                    gt | (eq & tie)
                };
                let v: uint8x16_t = transmute(vals);

                let mut prev: u8 = 0;
                for b in 0..K {
                    let next: u8 = if b + 1 < K { above(pivots[b]) } else { $full };
                    let m = next & !prev;
                    let key = vld1q_u8($shuf[m as usize].as_ptr());
                    vst1q_u8(writes[b].add(cnts[b]) as *mut u8, vqtbl1q_u8(v, key));
                    cnts[b] += m.count_ones() as usize;
                    prev = next;
                }
            }
        }

        #[inline(always)]
        unsafe fn partition_multi_slow<const K: usize>(
            vals: Self::Simd,
            len: Self::Simd,
            pivots: &[Self::Simd],
            rand: usize,
            writes: &mut [*mut $t; K],
            cnts: &mut [usize; K],
        ) {
            unsafe {
                use core::arch::aarch64::*;
                use std::mem::transmute;

                debug_assert_eq!(pivots.len() + 1, K);
                let in_range: u8 = (len
                    .simd_gt(<Self as SimdElem<$t>>::lane_indices())
                    .to_bitmask() as u8)
                    & $full;
                let tie: u8 = (EQUAL_MASKS[rand % 16] as u8) & $full;
                let above = |p: Self::Simd| -> u8 {
                    let gt = (vals.simd_gt(p).to_bitmask() as u8) & $full;
                    let eq = (vals.simd_eq(p).to_bitmask() as u8) & $full;
                    gt | (eq & tie)
                };
                let v: uint8x16_t = transmute(vals);

                let mut prev: u8 = 0;
                for b in 0..K {
                    let next: u8 = if b + 1 < K { above(pivots[b]) } else { $full };
                    let m = next & !prev & in_range;
                    let key = vld1q_u8($shuf[m as usize].as_ptr());
                    vst1q_u8(writes[b].add(cnts[b]) as *mut u8, vqtbl1q_u8(v, key));
                    cnts[b] += m.count_ones() as usize;
                    prev = next;
                }
            }
        }
    };
}

#[cfg(target_feature = "avx2")]
impl_simd_elem_32!(i32, wide::i32x8);
#[cfg(target_feature = "avx2")]
impl_simd_elem_32!(u32, wide::u32x8);
#[cfg(target_feature = "avx2")]
impl_simd_elem_64!(i64, wide::i64x4);
#[cfg(target_feature = "avx2")]
impl_simd_elem_64!(u64, wide::u64x4);

#[cfg(target_feature = "avx512f")]
impl_simd_elem_32_avx512!(i32, wide::i32x16);
#[cfg(target_feature = "avx512f")]
impl_simd_elem_32_avx512!(u32, wide::u32x16);
#[cfg(target_feature = "avx512f")]
impl_simd_elem_64_avx512!(i64, wide::i64x8);
#[cfg(target_feature = "avx512f")]
impl_simd_elem_64_avx512!(u64, wide::u64x8);

#[cfg(target_arch = "aarch64")]
impl_simd_elem_32_neon!(i32, wide::i32x4);
#[cfg(target_arch = "aarch64")]
impl_simd_elem_32_neon!(u32, wide::u32x4);
#[cfg(target_arch = "aarch64")]
impl_simd_elem_64_neon!(i64, wide::i64x2);
#[cfg(target_arch = "aarch64")]
impl_simd_elem_64_neon!(u64, wide::u64x2);

/// Methods to generate random balanced equal masks for the tiebreaker
const fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

const fn balanced(mut x: u64) -> u64 {
    while x.count_ones() > 32 {
        x &= x - 1;
    } // clear lowest set bit
    while x.count_ones() < 32 {
        x |= !x & x.wrapping_add(1);
    } // set lowest zero bit
    x
}

const fn build<const N: usize>() -> [u64; N] {
    let mut t = [0u64; N];
    let mut s = 0x1234_5678_9ABC_DEF0u64;
    let mut i = 0;
    while i < N {
        s = mix(s);
        t[i] = balanced(s);
        i += 1;
    }
    t
}

static EQUAL_MASKS: [u64; 16] = build::<16>();

/// For each of 256 masks of which elements are different than their predecessor,
/// a shuffle that sends those new elements to the beginning.
#[rustfmt::skip]
pub(crate) const UNIQSHUF32: [[i32; 8]; 256] = unsafe {transmute([
0,1,2,3,4,5,6,7,
1,2,3,4,5,6,7,0,
0,2,3,4,5,6,7,0,
2,3,4,5,6,7,0,0,
0,1,3,4,5,6,7,0,
1,3,4,5,6,7,0,0,
0,3,4,5,6,7,0,0,
3,4,5,6,7,0,0,0,
0,1,2,4,5,6,7,0,
1,2,4,5,6,7,0,0,
0,2,4,5,6,7,0,0,
2,4,5,6,7,0,0,0,
0,1,4,5,6,7,0,0,
1,4,5,6,7,0,0,0,
0,4,5,6,7,0,0,0,
4,5,6,7,0,0,0,0,
0,1,2,3,5,6,7,0,
1,2,3,5,6,7,0,0,
0,2,3,5,6,7,0,0,
2,3,5,6,7,0,0,0,
0,1,3,5,6,7,0,0,
1,3,5,6,7,0,0,0,
0,3,5,6,7,0,0,0,
3,5,6,7,0,0,0,0,
0,1,2,5,6,7,0,0,
1,2,5,6,7,0,0,0,
0,2,5,6,7,0,0,0,
2,5,6,7,0,0,0,0,
0,1,5,6,7,0,0,0,
1,5,6,7,0,0,0,0,
0,5,6,7,0,0,0,0,
5,6,7,0,0,0,0,0,
0,1,2,3,4,6,7,0,
1,2,3,4,6,7,0,0,
0,2,3,4,6,7,0,0,
2,3,4,6,7,0,0,0,
0,1,3,4,6,7,0,0,
1,3,4,6,7,0,0,0,
0,3,4,6,7,0,0,0,
3,4,6,7,0,0,0,0,
0,1,2,4,6,7,0,0,
1,2,4,6,7,0,0,0,
0,2,4,6,7,0,0,0,
2,4,6,7,0,0,0,0,
0,1,4,6,7,0,0,0,
1,4,6,7,0,0,0,0,
0,4,6,7,0,0,0,0,
4,6,7,0,0,0,0,0,
0,1,2,3,6,7,0,0,
1,2,3,6,7,0,0,0,
0,2,3,6,7,0,0,0,
2,3,6,7,0,0,0,0,
0,1,3,6,7,0,0,0,
1,3,6,7,0,0,0,0,
0,3,6,7,0,0,0,0,
3,6,7,0,0,0,0,0,
0,1,2,6,7,0,0,0,
1,2,6,7,0,0,0,0,
0,2,6,7,0,0,0,0,
2,6,7,0,0,0,0,0,
0,1,6,7,0,0,0,0,
1,6,7,0,0,0,0,0,
0,6,7,0,0,0,0,0,
6,7,0,0,0,0,0,0,
0,1,2,3,4,5,7,0,
1,2,3,4,5,7,0,0,
0,2,3,4,5,7,0,0,
2,3,4,5,7,0,0,0,
0,1,3,4,5,7,0,0,
1,3,4,5,7,0,0,0,
0,3,4,5,7,0,0,0,
3,4,5,7,0,0,0,0,
0,1,2,4,5,7,0,0,
1,2,4,5,7,0,0,0,
0,2,4,5,7,0,0,0,
2,4,5,7,0,0,0,0,
0,1,4,5,7,0,0,0,
1,4,5,7,0,0,0,0,
0,4,5,7,0,0,0,0,
4,5,7,0,0,0,0,0,
0,1,2,3,5,7,0,0,
1,2,3,5,7,0,0,0,
0,2,3,5,7,0,0,0,
2,3,5,7,0,0,0,0,
0,1,3,5,7,0,0,0,
1,3,5,7,0,0,0,0,
0,3,5,7,0,0,0,0,
3,5,7,0,0,0,0,0,
0,1,2,5,7,0,0,0,
1,2,5,7,0,0,0,0,
0,2,5,7,0,0,0,0,
2,5,7,0,0,0,0,0,
0,1,5,7,0,0,0,0,
1,5,7,0,0,0,0,0,
0,5,7,0,0,0,0,0,
5,7,0,0,0,0,0,0,
0,1,2,3,4,7,0,0,
1,2,3,4,7,0,0,0,
0,2,3,4,7,0,0,0,
2,3,4,7,0,0,0,0,
0,1,3,4,7,0,0,0,
1,3,4,7,0,0,0,0,
0,3,4,7,0,0,0,0,
3,4,7,0,0,0,0,0,
0,1,2,4,7,0,0,0,
1,2,4,7,0,0,0,0,
0,2,4,7,0,0,0,0,
2,4,7,0,0,0,0,0,
0,1,4,7,0,0,0,0,
1,4,7,0,0,0,0,0,
0,4,7,0,0,0,0,0,
4,7,0,0,0,0,0,0,
0,1,2,3,7,0,0,0,
1,2,3,7,0,0,0,0,
0,2,3,7,0,0,0,0,
2,3,7,0,0,0,0,0,
0,1,3,7,0,0,0,0,
1,3,7,0,0,0,0,0,
0,3,7,0,0,0,0,0,
3,7,0,0,0,0,0,0,
0,1,2,7,0,0,0,0,
1,2,7,0,0,0,0,0,
0,2,7,0,0,0,0,0,
2,7,0,0,0,0,0,0,
0,1,7,0,0,0,0,0,
1,7,0,0,0,0,0,0,
0,7,0,0,0,0,0,0,
7,0,0,0,0,0,0,0,
0,1,2,3,4,5,6,0,
1,2,3,4,5,6,0,0,
0,2,3,4,5,6,0,0,
2,3,4,5,6,0,0,0,
0,1,3,4,5,6,0,0,
1,3,4,5,6,0,0,0,
0,3,4,5,6,0,0,0,
3,4,5,6,0,0,0,0,
0,1,2,4,5,6,0,0,
1,2,4,5,6,0,0,0,
0,2,4,5,6,0,0,0,
2,4,5,6,0,0,0,0,
0,1,4,5,6,0,0,0,
1,4,5,6,0,0,0,0,
0,4,5,6,0,0,0,0,
4,5,6,0,0,0,0,0,
0,1,2,3,5,6,0,0,
1,2,3,5,6,0,0,0,
0,2,3,5,6,0,0,0,
2,3,5,6,0,0,0,0,
0,1,3,5,6,0,0,0,
1,3,5,6,0,0,0,0,
0,3,5,6,0,0,0,0,
3,5,6,0,0,0,0,0,
0,1,2,5,6,0,0,0,
1,2,5,6,0,0,0,0,
0,2,5,6,0,0,0,0,
2,5,6,0,0,0,0,0,
0,1,5,6,0,0,0,0,
1,5,6,0,0,0,0,0,
0,5,6,0,0,0,0,0,
5,6,0,0,0,0,0,0,
0,1,2,3,4,6,0,0,
1,2,3,4,6,0,0,0,
0,2,3,4,6,0,0,0,
2,3,4,6,0,0,0,0,
0,1,3,4,6,0,0,0,
1,3,4,6,0,0,0,0,
0,3,4,6,0,0,0,0,
3,4,6,0,0,0,0,0,
0,1,2,4,6,0,0,0,
1,2,4,6,0,0,0,0,
0,2,4,6,0,0,0,0,
2,4,6,0,0,0,0,0,
0,1,4,6,0,0,0,0,
1,4,6,0,0,0,0,0,
0,4,6,0,0,0,0,0,
4,6,0,0,0,0,0,0,
0,1,2,3,6,0,0,0,
1,2,3,6,0,0,0,0,
0,2,3,6,0,0,0,0,
2,3,6,0,0,0,0,0,
0,1,3,6,0,0,0,0,
1,3,6,0,0,0,0,0,
0,3,6,0,0,0,0,0,
3,6,0,0,0,0,0,0,
0,1,2,6,0,0,0,0,
1,2,6,0,0,0,0,0,
0,2,6,0,0,0,0,0,
2,6,0,0,0,0,0,0,
0,1,6,0,0,0,0,0,
1,6,0,0,0,0,0,0,
0,6,0,0,0,0,0,0,
6,0,0,0,0,0,0,0,
0,1,2,3,4,5,0,0,
1,2,3,4,5,0,0,0,
0,2,3,4,5,0,0,0,
2,3,4,5,0,0,0,0,
0,1,3,4,5,0,0,0,
1,3,4,5,0,0,0,0,
0,3,4,5,0,0,0,0,
3,4,5,0,0,0,0,0,
0,1,2,4,5,0,0,0,
1,2,4,5,0,0,0,0,
0,2,4,5,0,0,0,0,
2,4,5,0,0,0,0,0,
0,1,4,5,0,0,0,0,
1,4,5,0,0,0,0,0,
0,4,5,0,0,0,0,0,
4,5,0,0,0,0,0,0,
0,1,2,3,5,0,0,0,
1,2,3,5,0,0,0,0,
0,2,3,5,0,0,0,0,
2,3,5,0,0,0,0,0,
0,1,3,5,0,0,0,0,
1,3,5,0,0,0,0,0,
0,3,5,0,0,0,0,0,
3,5,0,0,0,0,0,0,
0,1,2,5,0,0,0,0,
1,2,5,0,0,0,0,0,
0,2,5,0,0,0,0,0,
2,5,0,0,0,0,0,0,
0,1,5,0,0,0,0,0,
1,5,0,0,0,0,0,0,
0,5,0,0,0,0,0,0,
5,0,0,0,0,0,0,0,
0,1,2,3,4,0,0,0,
1,2,3,4,0,0,0,0,
0,2,3,4,0,0,0,0,
2,3,4,0,0,0,0,0,
0,1,3,4,0,0,0,0,
1,3,4,0,0,0,0,0,
0,3,4,0,0,0,0,0,
3,4,0,0,0,0,0,0,
0,1,2,4,0,0,0,0,
1,2,4,0,0,0,0,0,
0,2,4,0,0,0,0,0,
2,4,0,0,0,0,0,0,
0,1,4,0,0,0,0,0,
1,4,0,0,0,0,0,0,
0,4,0,0,0,0,0,0,
4,0,0,0,0,0,0,0,
0,1,2,3,0,0,0,0,
1,2,3,0,0,0,0,0,
0,2,3,0,0,0,0,0,
2,3,0,0,0,0,0,0,
0,1,3,0,0,0,0,0,
1,3,0,0,0,0,0,0,
0,3,0,0,0,0,0,0,
3,0,0,0,0,0,0,0,
0,1,2,0,0,0,0,0,
1,2,0,0,0,0,0,0,
0,2,0,0,0,0,0,0,
2,0,0,0,0,0,0,0,
0,1,0,0,0,0,0,0,
1,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,
])};

/// Masks for 32-bit shuffle instructions on 64-bit data.
#[rustfmt::skip]
pub(crate) const UNIQSHUF64: [[i32; 8]; 16] = unsafe {
transmute([
0, 1, 2, 3, 4, 5, 6, 7, //0000
2, 3, 4, 5, 6, 7, 0, 0, //1000
0, 1, 4, 5, 6, 7, 0, 0, //0100
4, 5, 6, 7, 0, 0, 0, 0, //1100
0, 1, 2, 3, 6, 7, 0, 0, //0010
2, 3, 6, 7, 0, 0, 0, 0, //1010
0, 1, 6, 7, 0, 0, 0, 0, //0110
6, 7, 0, 0, 0, 0, 0, 0, //1110
0, 1, 2, 3, 4, 5, 0, 0, //0001
2, 3, 4, 5, 0, 0, 0, 0, //1001
0, 1, 4, 5, 0, 0, 0, 0, //0101
4, 5, 0, 0, 0, 0, 0, 0, //1101
0, 1, 2, 3, 0, 0, 0, 0, //0011
2, 3, 0, 0, 0, 0, 0, 0, //1011
0, 1, 0, 0, 0, 0, 0, 0, //0111
0, 0, 0, 0, 0, 0, 0, 0, //1111
])
};
