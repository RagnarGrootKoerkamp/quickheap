use crate::workloads::Elem;
use std::mem::transmute;

#[inline(always)]
pub fn push_position<T: SimdElem>(pivots: &Vec<T>, t: T) -> usize {
    // Baseline:
    // return pivots.iter().map(|x| (t <= **x) as usize).sum::<usize>();

    let v = unsafe { pivots.get_unchecked(..pivots.len().next_multiple_of(T::L)) };

    if v.len() <= 64 {
        let t_simd = T::splat(t);

        let mut target_layer = 0;
        let mut i = 0;
        while i < v.len() {
            let vals = unsafe { T::simd_from_slice(v.get_unchecked(i..i + T::L)) };
            // TODO: Compare SIMD register against 0
            target_layer += (T::simd_le_bitmask(t_simd, vals) as u8).trailing_ones() as usize;
            i += T::L;
        }
        target_layer = target_layer.min(pivots.len());

        let l2 = pivots
            .binary_search_by(|p| {
                if *p < t {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            })
            .unwrap_err();
        assert_eq!(target_layer, l2, "pivots {pivots:?} v {v:?}, t {t:?}");

        target_layer
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

#[inline(never)]
pub fn position_min<T: SimdElem>(v: &mut Vec<T>) -> usize {
    // Baseline:
    // let mut min = T::MAX;
    // let mut pos = 0;
    // for i in 0..v.len() {
    //     if v[i] <= min {
    //         min = v[i];
    //         pos = i;
    //     }
    // }
    // return pos;

    let mut min_pos = [0; 2];
    let mut min_val = [T::MAX; 2];
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

/// Trait for element types supported by `SimdQuickHeap`.
///
/// Implemented for `i32`, `u32`, `i64`, and `u64`.
pub trait SimdElem: Elem + Copy + 'static {
    /// Number of SIMD lanes (8 for 32-bit types, 4 for 64-bit types).
    const L: usize;
    /// Maximum value for this type.
    const MAX: Self;
    /// The SIMD vector type (e.g. `i32x8` or `i64x4`).
    type Simd: Copy;

    fn splat(v: Self) -> Self::Simd;
    /// # Safety
    /// `slice` must have at least `L` elements accessible (may read past `slice.len()`).
    unsafe fn simd_from_slice(slice: &[Self]) -> Self::Simd;
    /// Returns bitmask where bit `i` = `a[i] <= b[i]`.
    fn simd_le_bitmask(a: Self::Simd, b: Self::Simd) -> u64;
    /// Returns a SIMD register `[0, 1, 2, ..., L-1]`.
    fn lane_indices() -> Self::Simd;
    fn from_usize(n: usize) -> Self;
    fn wrapping_add_one(self) -> Self;

    /// Partition all `L` lanes of `vals` against `threshold`.
    /// Lanes `>= threshold` go to `v`, lanes `< threshold` go to `w`.
    /// # Safety
    /// `v` and `w` must have at least `L` elements of capacity beyond their current write index.
    unsafe fn partition_fast(
        vals: Self::Simd,
        threshold: Self::Simd,
        v: &mut [Self],
        v_idx: &mut usize,
        w: &mut [Self],
        w_idx: &mut usize,
    );

    /// Like `partition_fast`, but only the first `len` lanes are in range.
    /// # Safety
    /// Same capacity requirements as `partition_fast`.
    unsafe fn partition_slow(
        vals: Self::Simd,
        len: Self::Simd,
        threshold: Self::Simd,
        v: &mut [Self],
        v_idx: &mut usize,
        w: &mut [Self],
        w_idx: &mut usize,
    );
}

macro_rules! impl_simd_elem_32 {
    ($t:ty, $simd:ty) => {
        impl SimdElem for $t {
            const L: usize = 8;
            const MAX: Self = <$t>::MAX;
            type Simd = $simd;

            #[inline(always)]
            fn splat(v: Self) -> $simd {
                <$simd>::splat(v)
            }

            #[inline(always)]
            unsafe fn simd_from_slice(slice: &[Self]) -> $simd {
                unsafe { <$simd>::from_array(*(slice.as_ptr() as *const [Self; 8])) }
            }

            #[inline(always)]
            fn simd_le_bitmask(a: $simd, b: $simd) -> u64 {
                use std::simd::cmp::SimdPartialOrd;
                a.simd_le(b).to_bitmask()
            }

            #[inline(always)]
            fn lane_indices() -> $simd {
                <$simd>::from_array([0 as $t, 1, 2, 3, 4, 5, 6, 7])
            }

            #[inline(always)]
            fn from_usize(n: usize) -> Self {
                n as $t
            }

            #[inline(always)]
            fn wrapping_add_one(self) -> Self {
                self.wrapping_add(1)
            }

            #[inline(always)]
            unsafe fn partition_fast(
                vals: $simd,
                threshold: $simd,
                v: &mut [Self],
                v_idx: &mut usize,
                w: &mut [Self],
                w_idx: &mut usize,
            ) {
                unsafe {
                    use core::arch::x86_64::*;
                    use std::mem::transmute;
                    use std::simd::cmp::SimdPartialOrd;

                    // bit i = lane i is small (< threshold)
                    let small = threshold.simd_gt(vals).to_bitmask() as u8;
                    let large = !small;
                    let vals: __m256i = transmute(vals);

                    // Write large (>= threshold) to v: exclude small lanes.
                    let key: __m256i = transmute(crate::simd::UNIQSHUF32[small as usize]);
                    _mm256_storeu_si256(
                        v.as_mut_ptr().add(*v_idx) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *v_idx += large.count_ones() as usize;

                    // Write small (< threshold) to w: exclude large lanes.
                    let key: __m256i = transmute(crate::simd::UNIQSHUF32[large as usize]);
                    _mm256_storeu_si256(
                        w.as_mut_ptr().add(*w_idx) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *w_idx += small.count_ones() as usize;
                }
            }

            #[inline(always)]
            unsafe fn partition_slow(
                vals: $simd,
                len: $simd,
                threshold: $simd,
                v: &mut [Self],
                v_idx: &mut usize,
                w: &mut [Self],
                w_idx: &mut usize,
            ) {
                unsafe {
                    use core::arch::x86_64::*;
                    use std::mem::transmute;
                    use std::simd::cmp::SimdPartialOrd;

                    let mut small = vals.simd_lt(threshold).to_bitmask() as u8;
                    let mut large = vals.simd_ge(threshold).to_bitmask() as u8;
                    let in_range = len.simd_gt(Self::lane_indices()).to_bitmask() as u8;
                    small &= in_range;
                    large &= in_range;

                    let vals: __m256i = transmute(vals);

                    // Exclude mask = complement of keep mask.
                    let key: __m256i = transmute(crate::simd::UNIQSHUF32[(!large) as usize]);
                    _mm256_storeu_si256(
                        v.as_mut_ptr().add(*v_idx) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *v_idx += large.count_ones() as usize;

                    let key: __m256i = transmute(crate::simd::UNIQSHUF32[(!small) as usize]);
                    _mm256_storeu_si256(
                        w.as_mut_ptr().add(*w_idx) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *w_idx += small.count_ones() as usize;
                }
            }
        }
    };
}

macro_rules! impl_simd_elem_64 {
    ($t:ty, $simd:ty) => {
        impl SimdElem for $t {
            const L: usize = 4;
            const MAX: Self = <$t>::MAX;
            type Simd = $simd;

            #[inline(always)]
            fn splat(v: Self) -> $simd {
                <$simd>::splat(v)
            }

            #[inline(always)]
            unsafe fn simd_from_slice(slice: &[Self]) -> $simd {
                unsafe { <$simd>::from_array(*(slice.as_ptr() as *const [Self; 4])) }
            }

            #[inline(always)]
            fn simd_le_bitmask(a: $simd, b: $simd) -> u64 {
                use std::simd::cmp::SimdPartialOrd;
                a.simd_le(b).to_bitmask()
            }

            #[inline(always)]
            fn lane_indices() -> $simd {
                <$simd>::from_array([0 as $t, 1, 2, 3])
            }

            #[inline(always)]
            fn from_usize(n: usize) -> Self {
                n as $t
            }

            #[inline(always)]
            fn wrapping_add_one(self) -> Self {
                self.wrapping_add(1)
            }

            #[inline(always)]
            unsafe fn partition_fast(
                vals: $simd,
                threshold: $simd,
                v: &mut [Self],
                v_idx: &mut usize,
                w: &mut [Self],
                w_idx: &mut usize,
            ) {
                unsafe {
                    use core::arch::x86_64::*;
                    use std::mem::transmute;
                    use std::simd::cmp::SimdPartialOrd;

                    // 4-bit mask: bit i = lane i is small (< threshold).
                    let small = (threshold.simd_gt(vals).to_bitmask() as u8) & 0xF;
                    let large = small ^ 0xF;
                    let vals: __m256i = transmute(vals);

                    // UNIQSHUF64[k] keeps the lanes described by keep_pattern = k ^ 0xF.
                    // To keep large lanes (keep_pattern = large): index = large ^ 0xF = small.
                    let key: __m256i = transmute(crate::simd::UNIQSHUF64[small as usize]);
                    _mm256_storeu_si256(
                        v.as_mut_ptr().add(*v_idx) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *v_idx += large.count_ones() as usize;

                    // To keep small lanes (keep_pattern = small): index = small ^ 0xF = large.
                    let key: __m256i = transmute(crate::simd::UNIQSHUF64[large as usize]);
                    _mm256_storeu_si256(
                        w.as_mut_ptr().add(*w_idx) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *w_idx += small.count_ones() as usize;
                }
            }

            #[inline(always)]
            unsafe fn partition_slow(
                vals: $simd,
                len: $simd,
                threshold: $simd,
                v: &mut [Self],
                v_idx: &mut usize,
                w: &mut [Self],
                w_idx: &mut usize,
            ) {
                unsafe {
                    use core::arch::x86_64::*;
                    use std::mem::transmute;
                    use std::simd::cmp::SimdPartialOrd;

                    let mut small = (vals.simd_lt(threshold).to_bitmask() as u8) & 0xF;
                    let mut large = (vals.simd_ge(threshold).to_bitmask() as u8) & 0xF;
                    let in_range = (len.simd_gt(Self::lane_indices()).to_bitmask() as u8) & 0xF;
                    small &= in_range;
                    large &= in_range;

                    let vals: __m256i = transmute(vals);

                    // To keep large lanes: index = large ^ 0xF.
                    let key: __m256i = transmute(crate::simd::UNIQSHUF64[(large ^ 0xF) as usize]);
                    _mm256_storeu_si256(
                        v.as_mut_ptr().add(*v_idx) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *v_idx += large.count_ones() as usize;

                    // To keep small lanes: index = small ^ 0xF.
                    let key: __m256i = transmute(crate::simd::UNIQSHUF64[(small ^ 0xF) as usize]);
                    _mm256_storeu_si256(
                        w.as_mut_ptr().add(*w_idx) as *mut __m256i,
                        _mm256_permutevar8x32_epi32(vals, key),
                    );
                    *w_idx += small.count_ones() as usize;
                }
            }
        }
    };
}

impl_simd_elem_32!(i32, std::simd::i32x8);
impl_simd_elem_64!(i64, std::simd::i64x4);

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
