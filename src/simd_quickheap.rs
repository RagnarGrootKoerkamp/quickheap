use crate::impls::NoHeap;
use crate::pivot_strategies::PivotStrategy;
use crate::simd::{SimdElem, position_min, push_position};
use crate::workloads::Elem;

use super::Heap;
use std::marker::PhantomData;

pub struct SimdQuickHeap<T: Elem, S: SimdElem<T>, P: PivotStrategy, const N: usize> {
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

    _p: PhantomData<P>,
    _backend: PhantomData<S>,
}

impl<T: Elem, S: SimdElem<T>, P: PivotStrategy, const N: usize> Heap<T>
    for SimdQuickHeap<T, S, P, N>
{
    type CountedHeap = NoHeap;

    fn default() -> Self {
        Self {
            pivots: Vec::with_capacity(128),
            buckets: vec![vec![]; 128],
            _p: PhantomData,
            _backend: PhantomData,
        }
    }
    fn push(&mut self, t: T) {
        let target_layer = push_position::<T, S>(&self.pivots, t);
        self.buckets[target_layer].reserve(S::L + 1);
        self.buckets[target_layer].push(t);
    }
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
        let min_pos = position_min::<T, S>(layer);
        let min = layer.swap_remove(min_pos);

        // Update the active layer.
        if layer.is_empty() && self.pivots.len() > 0 {
            self.pivots.pop();
        }
        Some(min)
    }
}

impl<T: Elem, S: SimdElem<T>, P: PivotStrategy, const N: usize> SimdQuickHeap<T, S, P, N> {
    #[inline(never)]
    pub(crate) fn partition(&mut self) {
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
