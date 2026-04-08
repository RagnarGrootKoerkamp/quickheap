use crate::impls::NoHeap;
use crate::simd::SimdElem;
use crate::workloads::Elem;

use super::simd;
use super::Heap;
use std::array::from_fn;

pub struct SimdQuickHeap<T: SimdElem, const N: usize, const M: usize> {
    /// A decreasing array of the pivots for all layers.
    /// buckets[i] >= pivots[i] >= buckets[i+1]
    /// Values equal to pivots[i] can be in layer i or i+1.
    /// The first layer does not have a pivot in this array.
    ///
    /// The effective number of layers is always 1 longer than `pivots`.
    ///
    /// This will have enough underlying capacity for out-of-bounds SIMD reads.
    pub(crate) pivots: Vec<T>,
    /// The values in each layer.
    /// pivots[i-1] >= elements of buckets[i] >= pivots[i]
    /// Values equal to pivots[i] can be in layer i or i-1.
    ///
    /// This can be longer than `layer` to reuse allocations.
    pub(crate) buckets: Vec<Vec<T>>,
}

impl<T: SimdElem, const N: usize, const M: usize> Heap<T> for SimdQuickHeap<T, N, M> {
    type Casted<T2: Elem> = NoHeap;

    fn default() -> Self {
        Self {
            pivots: Vec::with_capacity(128),
            buckets: vec![vec![]; 128],
        }
    }
    #[inline(never)]
    fn push(&mut self, t: T) {
        let target_layer = simd::push_position(&self.pivots, t);
        self.buckets[target_layer].reserve(T::L + 1);
        self.buckets[target_layer].push(t);
    }
    #[inline(never)]
    fn pop(&mut self) -> Option<T> {
        let layer = self.pivots.len();
        // Only the top layer can be empty.
        if layer == 0 && self.buckets[0].is_empty() {
            return None;
        }
        // Split the current layer as long as it is too large.
        while self.buckets[self.pivots.len()].len() > N {
            self.partition();
        }
        // Find and extract the minimum.
        let layer = &mut self.buckets[self.pivots.len()];
        let min_pos = simd::position_min(layer);
        let min = layer.swap_remove(min_pos);

        // Update the active layer.
        if layer.is_empty() && self.pivots.len() > 0 {
            self.pivots.pop();
        }
        Some(min)
    }
}

impl<T: SimdElem, const N: usize, const M: usize> SimdQuickHeap<T, N, M> {
    #[inline(never)]
    pub(crate) fn partition(&mut self) {
        // Reserve space for an additional L layers when needed.
        let layer = self.pivots.len();
        if layer + 2 * T::L >= self.pivots.capacity() {
            self.pivots.reserve(T::L);
        }
        if layer + 1 == self.buckets.len() {
            self.buckets.push(vec![]);
        }
        // Alias the current layer (to be split) and the next layer.
        let [cur_layer, next_layer] = &mut self.buckets[layer..=layer + 1] else {
            unreachable!()
        };
        let n = cur_layer.len();

        // Select M random pivots and compute their median.
        let mut pivots: [(T, usize); M] = from_fn(|_| {
            let pos = rand::random_range(0..n);
            (cur_layer[pos], pos)
        });
        pivots.sort_by_key(|(pivot, _)| *pivot);
        // Pivots are exclusive.
        let pivot = pivots[M / 2].0;
        let pivot_pos = pivots[M / 2].1;
        self.pivots.push(pivot);

        // Reserve space in the next layer,
        // and make sure the current layer can hold a spare SIMD register.
        cur_layer.reserve(T::L);
        next_layer.clear();
        next_layer.reserve(n + T::L);

        unsafe { cur_layer.set_len(n + T::L) };
        unsafe { next_layer.set_len(n + T::L) };

        let n2 = n.next_multiple_of(T::L).saturating_sub(T::L);

        // Partition a list into two using SIMD.
        let mut cur_len = 0;
        let mut next_len = 0;
        let half = (pivot_pos + 1).min(n2).next_multiple_of(T::L);
        let threshold = T::splat(pivot.wrapping_add_one());
        for i in (0..half).step_by(T::L) {
            unsafe {
                T::partition_fast(
                    T::simd_from_slice(cur_layer.get_unchecked(i..i + T::L)),
                    threshold,
                    cur_layer,
                    &mut cur_len,
                    next_layer,
                    &mut next_len,
                );
            }
        }
        let threshold = T::splat(pivot);
        for i in (half..n2).step_by(T::L) {
            unsafe {
                T::partition_fast(
                    T::simd_from_slice(cur_layer.get_unchecked(i..i + T::L)),
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
                T::splat(pivot.wrapping_add_one())
            } else {
                T::splat(pivot)
            };
            unsafe {
                T::partition_slow(
                    T::simd_from_slice(cur_layer.get_unchecked(n2..n2 + T::L)),
                    T::splat(T::from_usize(n - n2)),
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
            return;
        }
    }
}
