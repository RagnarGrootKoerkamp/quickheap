use crate::ElemT;
use crate::KvCore;
use crate::partitioning::VecPartitioning;
use crate::simd;
use crate::simd::SimdElem;
use crate::util::CountingBloomFilter;
use crate::util::ToBytes;
use crate::util::XxHashKHash;

use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

/// Caller has to ensure that every val is only inserted once
pub trait DecreaseKeyStrategy<T, S>
where
    T: ElemT + Debug,
    S: SimdElem<T>,
{
    fn decrease_key(&mut self, heap: &mut KvCore<T, S>, old_key: T, new_key: T, val: T);
    fn before_pop(&mut self, heap: &mut KvCore<T, S>);
    fn before_partition(&mut self, heap: &mut KvCore<T, S>);
    fn after_partition(&mut self, idx: usize, pivot: T, pivot_val: T, pivot_pos: usize);

    fn layer_remove(&mut self);
    fn layer_push(&mut self);
}

#[derive(Default)]
pub struct NoDecreaseKey;

impl<T, S> DecreaseKeyStrategy<T, S> for NoDecreaseKey
where
    T: ElemT + Debug,
    S: SimdElem<T>,
{
    fn decrease_key(&mut self, _: &mut KvCore<T, S>, _: T, _: T, _: T) {
        unreachable!("decrease_key is not supported by NoDecreaseKey");
    }

    fn before_pop(&mut self, _: &mut KvCore<T, S>) {}

    fn before_partition(&mut self, _: &mut KvCore<T, S>) {}

    fn layer_remove(&mut self) {}

    fn layer_push(&mut self) {}

    fn after_partition(&mut self, _: usize, _: T, _: T, _: usize) {}
}

/// Strategy that reinserts new decreased keys and tracks the already popped values, s.t. no value is popped twice
#[derive(Default)]
pub struct SimpleDK<T> {
    popped_vals: HashSet<T>,
}

impl<T, S> DecreaseKeyStrategy<T, S> for SimpleDK<T>
where
    T: Hash + ElemT + Debug + Default,
    S: SimdElem<T>,
{
    fn decrease_key(&mut self, heap: &mut KvCore<T, S>, _: T, new_key: T, val: T) {
        // Just push the new key and val again
        heap.insert_elem(new_key, val);
    }

    fn before_pop(&mut self, heap: &mut KvCore<T, S>) {
        // Drop stale entries whose value was already popped
        while let Some((_, val)) = heap.peek(self) {
            if !self.popped_vals.contains(&val) {
                break;
            }
            heap.remove_smallest_elem();
        }

        // Insert the val that is popped currently
        if let Some((_, val)) = heap.peek(self) {
            self.popped_vals.insert(val);
        }
    }

    fn before_partition(&mut self, _: &mut KvCore<T, S>) {}

    fn layer_remove(&mut self) {}
    fn layer_push(&mut self) {}
    fn after_partition(&mut self, _: usize, _: T, _: T, _: usize) {}
}

/// Strategy that inserts deletion tokens and cleans up partitions by sorting
pub struct DelTokenSort<T, S> {
    deleted_vals: Vec<Vec<T>>,
    deleted_keys: Vec<Vec<T>>,
    _s: std::marker::PhantomData<S>,
}

impl<T, S> Default for DelTokenSort<T, S> {
    fn default() -> Self {
        Self {
            deleted_keys: (0..128).map(|_| vec![]).collect(),
            deleted_vals: (0..128).map(|_| vec![]).collect(),
            _s: std::marker::PhantomData,
        }
    }
}

impl<T, S> DecreaseKeyStrategy<T, S> for DelTokenSort<T, S>
where
    T: ElemT + Debug,
    S: SimdElem<T>,
{
    fn decrease_key(&mut self, heap: &mut KvCore<T, S>, old_key: T, new_key: T, val: T) {
        // Insert the new elem
        heap.insert_elem(new_key, val);

        // Insert the deletion token into the separate buffer
        let target_layer = heap.layer_of(old_key, val);
        // let target_layer = simd::push_position::<T, S>(&heap.pivots, old_key);
        debug_assert!(target_layer < self.deleted_keys.len());
        debug_assert!(target_layer < self.deleted_vals.len());

        self.deleted_keys[target_layer].push(old_key);
        self.deleted_vals[target_layer].push(val);
    }

    fn before_pop(&mut self, heap: &mut KvCore<T, S>) {
        // Clean up lowest buffer if it contains a deletion token
        if self.deleted_keys[heap.pivots.len()].len() > 0 {
            self.clean_up(heap);
        }
    }

    fn before_partition(&mut self, heap: &mut KvCore<T, S>) {
        // Clean up partition if necessary
        let layer = heap.pivots.len();

        let layer_size = heap.keys[layer].len();
        let deletions = self.deleted_keys[layer].len();

        if deletions * 3 < layer_size {
            return;
        }

        self.clean_up(heap);
    }

    fn layer_remove(&mut self) {
        self.deleted_keys.pop();
        self.deleted_vals.pop();
    }

    fn layer_push(&mut self) {
        self.deleted_keys.push(vec![]);
        self.deleted_vals.push(vec![]);
    }

    fn after_partition(&mut self, idx: usize, pivot: T, pivot_val: T, pivot_pos: usize) {
        if idx + 1 == self.deleted_keys.len() {
            debug_assert!(idx + 1 == self.deleted_vals.len());
            self.deleted_keys.push(vec![]);
            self.deleted_vals.push(vec![]);
        }

        // Alias the current layer (to be split) and the next layer.
        let [cur_key_layer, next_key_layer] = &mut self.deleted_keys[idx..=idx + 1] else {
            unreachable!()
        };

        let [cur_val_layer, next_val_layer] = &mut self.deleted_vals[idx..=idx + 1] else {
            unreachable!()
        };

        let n = cur_key_layer.len();
        debug_assert!(cur_val_layer.len() == n);

        // Clear the next layer
        next_key_layer.clear();
        next_val_layer.clear();

        VecPartitioning::<T, S>::partition_kv(
            cur_key_layer,
            cur_val_layer,
            next_key_layer,
            next_val_layer,
            pivot,
            pivot_val,
            pivot_pos,
        );
    }
}

impl<T, S> DelTokenSort<T, S>
where
    T: ElemT + Debug,
    S: SimdElem<T>,
{
    fn clean_up(&mut self, heap: &mut KvCore<T, S>) {
        let layer = heap.pivots.len();

        if self.deleted_keys[layer].is_empty() || heap.keys[layer].is_empty() {
            return;
        }

        // First, sort all keys and the values accordingly
        KvCore::<T, S>::sort_key_and_values(&mut heap.keys[layer], &mut heap.values[layer]);
        KvCore::<T, S>::sort_key_and_values(
            &mut self.deleted_keys[layer],
            &mut self.deleted_vals[layer],
        );

        let mut idx_norm = heap.keys[layer].len() - 1;
        let mut idx_del = self.deleted_keys[layer].len() - 1;

        loop {
            if self.deleted_keys[layer][idx_del] > heap.keys[layer][idx_norm] {
                if idx_norm == 0 {
                    break;
                }

                idx_norm -= 1;
                continue;
            }

            if self.deleted_keys[layer][idx_del] < heap.keys[layer][idx_norm] {
                if idx_del == 0 {
                    break;
                }

                idx_del -= 1;
                continue;
            }

            // Both keys are equal
            if self.deleted_vals[layer][idx_del] > heap.values[layer][idx_norm] {
                if idx_norm == 0 {
                    break;
                }

                idx_norm -= 1;
                continue;
            }

            if self.deleted_vals[layer][idx_del] < heap.values[layer][idx_norm] {
                if idx_del == 0 {
                    break;
                }

                idx_del -= 1;
                continue;
            }

            // Both keys and both values are equal -> Remove element and deletion token
            self.deleted_keys[layer].remove(idx_del);
            self.deleted_vals[layer].remove(idx_del);
            heap.keys[layer].remove(idx_norm);
            heap.values[layer].remove(idx_norm);

            if idx_del == 0 || idx_norm == 0 {
                break;
            }

            idx_del -= 1;
            idx_norm -= 1;
        }

        assert!(self.deleted_keys[layer].len() == 0);
        assert!(self.deleted_vals[layer].len() == 0);

        if heap.keys[layer].len() == 0 {
            heap.pop_layer();
        }
    }
}

/// Strategy that inserts deletion tokens and cleans up partitions using bloom filters
pub struct DelTokenCountingBloom<T, S, const ELEM_BYTES: usize> {
    deleted_vals: Vec<Vec<T>>,
    deleted_keys: Vec<Vec<T>>,
    _s: std::marker::PhantomData<S>,
}

impl<T, S, const ELEM_BYTES: usize> Default for DelTokenCountingBloom<T, S, ELEM_BYTES> {
    fn default() -> Self {
        Self {
            deleted_keys: (0..128).map(|_| vec![]).collect(),
            deleted_vals: (0..128).map(|_| vec![]).collect(),
            _s: std::marker::PhantomData,
        }
    }
}

impl<T, S, const ELEM_BYTES: usize> DecreaseKeyStrategy<T, S>
    for DelTokenCountingBloom<T, S, ELEM_BYTES>
where
    T: ElemT + Debug,
    (T, T): ToBytes<ELEM_BYTES>,
    S: SimdElem<T>,
{
    fn decrease_key(&mut self, heap: &mut KvCore<T, S>, old_key: T, new_key: T, val: T) {
        // Insert the new elem
        heap.insert_elem(new_key, val);

        // Insert the deletion token into the separate buffer
        // let target_layer = simd::push_position::<T, S>(&heap.pivots, old_key);
        let target_layer = heap.layer_of(old_key, val);
        debug_assert!(target_layer < self.deleted_keys.len());
        debug_assert!(target_layer < self.deleted_vals.len());

        self.deleted_keys[target_layer].push(old_key);
        self.deleted_vals[target_layer].push(val);
    }

    fn before_pop(&mut self, heap: &mut KvCore<T, S>) {
        // Clean up lowest buffer if it contains a deletion token
        if self.deleted_keys[heap.pivots.len()].len() > 0 {
            self.clean_up(heap);
        }
    }

    fn before_partition(&mut self, heap: &mut KvCore<T, S>) {
        // Clean up partition if necessary
        let layer = heap.pivots.len();

        let layer_size = heap.keys[layer].len();
        let deletions = self.deleted_keys[layer].len();

        if deletions * 3 < layer_size {
            return;
        }

        self.clean_up(heap);
    }

    fn layer_remove(&mut self) {
        self.deleted_keys.pop();
        self.deleted_vals.pop();
    }

    fn layer_push(&mut self) {
        self.deleted_keys.push(vec![]);
        self.deleted_vals.push(vec![]);
    }

    fn after_partition(&mut self, idx: usize, pivot: T, pivot_val: T, pivot_pos: usize) {
        if idx + 1 == self.deleted_keys.len() {
            debug_assert!(idx + 1 == self.deleted_vals.len());
            self.deleted_keys.push(vec![]);
            self.deleted_vals.push(vec![]);
        }

        // Alias the current layer (to be split) and the next layer.
        let [cur_key_layer, next_key_layer] = &mut self.deleted_keys[idx..=idx + 1] else {
            unreachable!()
        };

        let [cur_val_layer, next_val_layer] = &mut self.deleted_vals[idx..=idx + 1] else {
            unreachable!()
        };

        let n = cur_key_layer.len();
        debug_assert!(cur_val_layer.len() == n);

        // Clear the next layer
        next_key_layer.clear();
        next_val_layer.clear();

        VecPartitioning::<T, S>::partition_kv(
            cur_key_layer,
            cur_val_layer,
            next_key_layer,
            next_val_layer,
            pivot,
            pivot_val,
            pivot_pos,
        );
    }
}

impl<T, S, const ELEM_BYTES: usize> DelTokenCountingBloom<T, S, ELEM_BYTES>
where
    T: ElemT + Debug,
    (T, T): ToBytes<ELEM_BYTES>,
    S: SimdElem<T>,
{
    fn clean_up(&mut self, heap: &mut KvCore<T, S>) {
        let layer = heap.pivots.len();

        if self.deleted_keys[layer].is_empty() || heap.keys[layer].is_empty() {
            return;
        }

        let mut cbf = CountingBloomFilter::<XxHashKHash, (T, T), 4, 128, 4, ELEM_BYTES>::new(1);

        // Sort the deletions
        KvCore::<T, S>::sort_key_and_values(
            &mut self.deleted_keys[layer],
            &mut self.deleted_vals[layer],
        );

        for i in 0..self.deleted_keys[layer].len() {
            let elem = (self.deleted_keys[layer][i], self.deleted_vals[layer][i]);
            cbf.insert(elem);
        }

        // Check all elements
        let mut remaining = self.deleted_keys[layer].len();
        for i in (0..heap.keys[layer].len()).rev() {
            let elem = (heap.keys[layer][i], heap.values[layer][i]);

            if !cbf.contains(elem) {
                // Element can be kept, it is definitely not deleted
                continue;
            }

            // Element maybe has a deletion token
            // Search the deletions to find
            let mut has_token = false; // TODO: Find something faster for that
            for i in 0..self.deleted_keys[layer].len() {
                if self.deleted_keys[layer][i] == elem.0 && self.deleted_vals[layer][i] == elem.1 {
                    has_token = true;
                    break;
                }
            }

            if !has_token {
                // Element can be kept, it has no matching token
                continue;
            }

            // Remove the deleted elem
            remaining -= 1;
            cbf.remove(elem);
            heap.values[layer].remove(i);
            heap.keys[layer].remove(i);

            if remaining == 0 {
                break;
            }
        }

        // Each token has been matched
        assert!(remaining == 0);

        // Clear the used tokens
        self.deleted_keys[layer].clear();
        self.deleted_vals[layer].clear();

        // If layer is now empty, remove it
        if heap.keys[layer].len() == 0 {
            heap.pop_layer();
        }
    }
}

#[cfg(test)]
mod test {
    /// Tiny xorshift PRNG: deterministic, no dependencies.
    pub struct Rng(pub u64);
    impl Rng {
        pub fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        pub fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    macro_rules! decrease_key_tests {
        ($modname:ident, $simd:ty, $dk:ty) => {
            mod $modname {
                use crate::decrease_key::test::Rng; // absolute path, independent of nesting
                use crate::*;

                type Heap = KvSimdQuickHeap<i64, $simd, $dk>;

                fn new_heap() -> Heap {
                    Heap::new()
                }

                #[test]
                fn empty_heap() {
                    let mut h = new_heap();
                    assert!(h.is_empty());
                    assert_eq!(h.pop(), None);
                    assert_eq!(h.peek(), None);
                }

                #[test]
                fn pop_returns_key_and_value_sorted() {
                    let mut h = new_heap();
                    h.push(30, 3);
                    h.push(10, 1);
                    h.push(20, 2);
                    assert_eq!(h.len(), 3);

                    assert_eq!(h.pop(), Some((10, 1)));
                    assert_eq!(h.pop(), Some((20, 2)));
                    assert_eq!(h.pop(), Some((30, 3)));
                    assert_eq!(h.pop(), None);
                }

                #[test]
                fn decrease_key_moves_value_to_front() {
                    let mut h = new_heap();
                    h.push(10, 1);
                    h.push(20, 2);
                    h.push(30, 3);

                    h.decrease_key(20, 5, 2);
                    assert_eq!(h.len(), 3, "decrease_key must not change len");

                    assert_eq!(h.peek(), Some((5, 2)));
                    assert_eq!(h.pop(), Some((5, 2)));
                    assert_eq!(h.pop(), Some((10, 1)));
                    assert_eq!(h.pop(), Some((30, 3)));
                    assert_eq!(h.pop(), None);
                    assert!(h.is_empty());
                }

                #[test]
                fn decrease_key_on_current_min() {
                    let mut h = new_heap();
                    h.push(10, 1);
                    h.push(20, 2);
                    h.decrease_key(10, 3, 1);
                    assert_eq!(h.pop(), Some((3, 1)));
                    assert_eq!(h.pop(), Some((20, 2)));
                    assert_eq!(h.pop(), None);
                }

                #[test]
                fn repeated_decrease_on_same_value() {
                    let mut h = new_heap();
                    h.push(1_000, 42);
                    h.push(500, 7);

                    let mut cur = 1_000;
                    for new in (1..500).rev().step_by(37) {
                        h.decrease_key(cur, new, 42);
                        cur = new;
                        assert_eq!(h.len(), 2);
                    }

                    assert_eq!(h.pop(), Some((cur, 42)));
                    assert_eq!(h.pop(), Some((500, 7)));
                    assert_eq!(h.pop(), None);
                }

                /// Enough elements to force several partition() calls,
                /// then decrease keys of elements living in deeper layers.
                #[test]
                fn decrease_key_across_layers() {
                    let mut h = new_heap();
                    let n = 1_000i64;
                    for v in 0..n {
                        h.push(10_000 + v * 3, v);
                    }
                    assert_eq!(h.peek(), Some((10_000, 0)));

                    for v in (n - 10)..n {
                        h.decrease_key(10_000 + v * 3, v - n, v);
                    }
                    assert_eq!(h.len(), n as usize);

                    let mut last = i64::MIN;
                    let mut count = 0;
                    while let Some((k, _)) = h.pop() {
                        assert!(k >= last, "order violated: {k} < {last}");
                        last = k;
                        count += 1;
                    }
                    assert_eq!(count, n);
                }

                #[test]
                fn randomized_against_reference_model() {
                    const N: usize = 5_000;
                    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
                    let mut h = new_heap();

                    let mut model: Vec<Option<i64>> = vec![None; N];
                    let mut next_fresh = 0usize;

                    for _ in 0..50_000 {
                        match rng.below(10) {
                            0..=3 if next_fresh < N => {
                                let key = rng.below(1_000_000) as i64 + 1;
                                h.push(key, next_fresh as i64);
                                model[next_fresh] = Some(key);
                                next_fresh += 1;
                            }
                            4..=6 => {
                                let v = rng.below(N as u64) as usize;
                                if let Some(old) = model[v] {
                                    if old > 0 {
                                        let new = rng.below(old as u64) as i64;
                                        h.decrease_key(old, new, v as i64);
                                        model[v] = Some(new);
                                    }
                                }
                            }
                            _ => {
                                let expected_key = model.iter().flatten().min().copied();
                                match h.pop() {
                                    None => assert_eq!(expected_key, None),
                                    Some((k, v)) => {
                                        let v = v as usize;
                                        assert_eq!(Some(k), expected_key, "non-minimal key");
                                        assert_eq!(model[v], Some(k), "stale/unknown entry");
                                        model[v] = None;
                                    }
                                }
                            }
                        }
                        assert_eq!(h.len(), model.iter().flatten().count());
                    }

                    let mut last = i64::MIN;
                    while let Some((k, v)) = h.pop() {
                        let v = v as usize;
                        assert!(k >= last, "heap order violated: {k} < {last}");
                        assert_eq!(model[v], Some(k));
                        model[v] = None;
                        last = k;
                    }
                    assert!(model.iter().all(|x| x.is_none()), "values lost in heap");
                }
            }
        };
    }

    #[cfg(target_feature = "avx2")]
    mod avx2_dk_test {
        decrease_key_tests!(simple_dk, crate::Avx2, crate::decrease_key::SimpleDK<i64>);
        decrease_key_tests!(
            del_token_sort,
            crate::Avx2,
            crate::decrease_key::DelTokenSort<i64, crate::Avx2>
        );
        decrease_key_tests!(
            del_token_cnt_bloom,
            crate::Avx2,
            crate::decrease_key::DelTokenCountingBloom<i64, crate::Avx2, 16>
        );
    }

    #[cfg(target_feature = "avx512f")]
    mod avx512_dk_test {
        decrease_key_tests!(simple_dk, crate::Avx512, crate::decrease_key::SimpleDK<i64>);
        decrease_key_tests!(del_token_sort, crate::Avx512, crate::decrease_key::DelTokenSort<i64, crate::Avx512>);
        decrease_key_tests!(del_token_cnt_bloom, crate::Avx512, crate::decrease_key::DelTokenCountingBloom<i64, crate::Avx512, 16>);
    }

    #[cfg(target_arch = "aarch64")]
    mod neon_dk_test {
        decrease_key_tests!(simple_dk, crate::Neon, crate::decrease_key::SimpleDK<i64>);
        decrease_key_tests!(del_token_sort, crate::Neon, crate::decrease_key::DelTokenSort<i64, crate::Neon>);
        decrease_key_tests!(del_token_cnt_bloom, crate::Neon, crate::decrease_key::DelTokenCountingBloom<i64, crate::Neon, 16>);
    }
}
