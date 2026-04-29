#[doc(hidden)]
pub mod pivot_strategies;
mod simd;

use std::marker::PhantomData;

pub use simd::Avx2;
#[cfg(target_feature = "avx512f")]
pub use simd::Avx512;

#[cfg(not(target_feature = "avx512f"))]
pub type Simd = Avx2;
#[cfg(target_feature = "avx512f")]
pub type Simd = Avx512;

pub trait Elem: Copy + Ord {}
impl<T: Copy + Ord> Elem for T {}
pub use simd::SimdElem;

pub struct ConfigurableSimdQuickHeap<
    T: Elem,
    S: simd::SimdElem<T> = Simd,
    P: pivot_strategies::PivotStrategy = pivot_strategies::MedianOfM<3>,
    const N: usize = 16,
    const SORT: bool = true,
> {
    /// A decreasing array of the pivots for all layers.
    /// buckets[i] >= pivots[i] >= buckets[i+1]
    /// Values equal to pivots[i] can be in layer i or i+1.
    /// The first layer does not have a pivot in this array.
    ///
    /// The effective number of layers is always 1 longer than `pivots`.
    ///
    /// This will have enough underlying capacity for out-of-bounds SIMD reads.
    pivots: Vec<T>,
    /// The values in each layer.
    /// pivots[i-1] >= elements of buckets[i] >= pivots[i]
    /// Values equal to pivots[i] can be in layer i or i-1.
    ///
    /// This can be longer than `layer` to reuse allocations.
    buckets: Vec<Vec<T>>,

    _p: PhantomData<P>,
    _backend: PhantomData<S>,
}

pub type SimdQuickHeap<T> =
    ConfigurableSimdQuickHeap<T, Simd, pivot_strategies::MedianOfM<3>, 16, true>;

impl<
    T: Elem,
    S: simd::SimdElem<T>,
    P: pivot_strategies::PivotStrategy,
    const N: usize,
    const SORT: bool,
> Default for ConfigurableSimdQuickHeap<T, S, P, N, SORT>
{
    fn default() -> Self {
        Self {
            pivots: Vec::with_capacity(128),
            buckets: (0..128).map(|_| vec![]).collect(),
            _p: PhantomData,
            _backend: PhantomData,
        }
    }
}

impl<
    T: Elem,
    S: simd::SimdElem<T>,
    P: pivot_strategies::PivotStrategy,
    const N: usize,
    const SORT: bool,
> ConfigurableSimdQuickHeap<T, S, P, N, SORT>
{
    pub fn push(&mut self, t: T) {
        let target_layer = simd::push_position::<T, S>(&self.pivots, t);
        let layer = &mut self.buckets[target_layer];
        layer.reserve(S::L + 1);
        if SORT && target_layer == self.pivots.len() && layer.len() < N {
            // Count the number of larger elements in the prefix and insert the new element after them.
            let pos = layer.partition_point(|&x| x > t);
            layer.insert(pos, t);
            // TODO: SIMD
        } else {
            layer.push(t);
        }
    }
    pub fn pop(&mut self) -> Option<T> {
        let layer = self.pivots.len();
        // Only the top layer can be empty.
        if layer == 0 && self.buckets[0].is_empty() {
            return None;
        }
        // Split the current layer as long as it is too large.
        if self.buckets[self.pivots.len()].len() > N {
            while self.buckets[self.pivots.len()].len() > N {
                self.partition();
            }
            if SORT {
                // Sort final layer decreasing.
                let layer = &mut self.buckets[self.pivots.len()];
                layer.sort_unstable_by_key(|&x| std::cmp::Reverse(x));
            }
        }
        // Find and extract the minimum.
        let layer = &mut self.buckets[self.pivots.len()];
        let min = if SORT {
            layer.pop().unwrap()
        } else {
            let min_pos = simd::position_min::<T, S>(layer);
            layer.swap_remove(min_pos)
        };

        // Update the active layer.
        if layer.is_empty() && self.pivots.len() > 0 {
            self.pivots.pop();
        }
        Some(min)
    }

    #[inline(never)]
    fn partition(&mut self) {
        // Reserve space for an additional L layers when needed.
        let layer = self.pivots.len();
        if layer + 2 * S::L >= self.pivots.capacity() {
            self.pivots.reserve(S::L);
        }
        if layer + 1 == self.buckets.len() {
            self.buckets.push(vec![]);
        }
        // Alias the current layer (to be split) and the next layer.
        let [cur_layer, next_layer] = &mut self.buckets[layer..=layer + 1] else {
            unreachable!()
        };
        let n = cur_layer.len();

        // Sample a pivot using the pivot strategy
        let (pivot, pivot_pos) = P::pick(&cur_layer);
        self.pivots.push(pivot);

        // Reserve space in the next layer,
        // and make sure the current layer can hold a spare SIMD register.
        cur_layer.reserve(S::L);
        next_layer.clear();
        next_layer.reserve(n + S::L);

        unsafe { cur_layer.set_len(n + S::L) };
        unsafe { next_layer.set_len(n + S::L) };

        let n2 = n.next_multiple_of(S::L).saturating_sub(S::L);

        // Partition a list into two using SIMD.
        let mut cur_len = 0;
        let mut next_len = 0;
        let half = (pivot_pos + 1).min(n2).next_multiple_of(S::L);
        let threshold = S::splat(S::wrapping_add_one(pivot));
        for i in (0..half).step_by(S::L) {
            unsafe {
                S::partition_fast(
                    S::simd_from_slice(cur_layer.get_unchecked(i..i + S::L)),
                    threshold,
                    cur_layer,
                    &mut cur_len,
                    next_layer,
                    &mut next_len,
                );
            }
        }
        let threshold = S::splat(pivot);
        for i in (half..n2).step_by(S::L) {
            unsafe {
                S::partition_fast(
                    S::simd_from_slice(cur_layer.get_unchecked(i..i + S::L)),
                    threshold,
                    cur_layer,
                    &mut cur_len,
                    next_layer,
                    &mut next_len,
                );
            }
        }
        if n2 < n {
            let threshold = if pivot_pos >= n2 {
                S::splat(S::wrapping_add_one(pivot))
            } else {
                S::splat(pivot)
            };
            unsafe {
                S::partition_slow(
                    S::simd_from_slice(cur_layer.get_unchecked(n2..n2 + S::L)),
                    S::splat(S::from_usize(n - n2)),
                    threshold,
                    cur_layer,
                    &mut cur_len,
                    next_layer,
                    &mut next_len,
                );
            }
        }

        debug_assert!(next_len > 0);

        unsafe {
            cur_layer.set_len(cur_len);
            next_layer.set_len(next_len);
        }

        // If we extracted all elements to the next layer
        // because the pivot was the largest one,
        // undo and try again.
        if cur_len == 0 {
            std::mem::swap(cur_layer, next_layer);
            self.pivots.pop().unwrap();
        }
    }
}
