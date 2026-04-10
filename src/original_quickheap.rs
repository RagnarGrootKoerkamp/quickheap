use std::collections::VecDeque;

use crate::workloads::Elem;

use super::Heap;

#[derive(Debug)]
pub struct OriginalQuickHeap<T: Elem> {
    pub data: VecDeque<T>,
    /// Pivots, from large to small.
    pub pivot_positions: Vec<usize>,
    /// Offset to the pivot positions. Incremented on pop.
    pub offset: usize,
}

impl<T: Elem> Heap<T> for OriginalQuickHeap<T> {
    type Casted<T2: Elem> = OriginalQuickHeap<T2>;

    fn default() -> Self {
        Self {
            data: VecDeque::new(),
            pivot_positions: vec![0],
            offset: 0,
        }
    }
    #[inline(never)]
    fn push(&mut self, t: T) {
        self.pivot_positions[0] += 1;
        self.data.push_back(t);
        let mut idx = self.data.len() - 1;
        let mut layer = 1;
        while layer < self.pivot_positions.len()
            && t < self.data[self.pivot_positions[layer] - self.offset]
        {
            // rotate pivot pos -> pivot pos + 1 -> idx=t -> pivot pos
            let pivot_pos = self.pivot_positions[layer] - self.offset;
            let p = self.data[pivot_pos];
            let x = self.data[pivot_pos + 1];
            self.data[idx] = x;
            self.data[pivot_pos] = t;
            self.data[pivot_pos + 1] = p;
            self.pivot_positions[layer] += 1;
            idx = pivot_pos;
            layer += 1;
        }
    }
    #[inline(never)]
    fn pop(&mut self) -> Option<T> {
        // Only the top layer can be empty.
        if self.data.is_empty() {
            return None;
        }
        // Split the current layer as long as it is too large.
        while self.pivot_positions.last().unwrap() - self.offset > 1 {
            self.partition();
        }
        // Find and extract the minimum.
        let x = self.data.pop_front().unwrap();
        self.offset += 1;
        if self.offset > *self.pivot_positions.last().unwrap() {
            self.pivot_positions.pop();
        }
        return Some(x);
    }
}

impl<T: Elem> OriginalQuickHeap<T> {
    #[inline(never)]
    fn partition(&mut self) {
        let n = self.pivot_positions.last().unwrap() - self.offset;
        assert!(n >= 2);
        // choose a random index as pivot
        let pivot_idx = rand::random_range(0..n);
        let pivot = self.data[pivot_idx];
        {
            // swap pivot to the end of range
            self.data[pivot_idx] = self.data[n - 1];
            self.data[n - 1] = pivot;
        }
        // Partition the range 0..n-1 by the pivot value using Hoare's algorithm.
        let mut i = 0;
        let mut j = n - 2;
        loop {
            while i < n - 1 && self.data[i] <= pivot {
                i += 1;
            }
            while j > i && self.data[j] >= pivot {
                j -= 1;
            }
            if i >= j {
                break;
            }
            (self.data[i], self.data[j]) = (self.data[j], self.data[i]);
            i += 1;
            j -= 1;
        }
        // swap pivot to its final position
        let pivot_pos = i;
        (self.data[pivot_pos], self.data[n - 1]) = (self.data[n - 1], self.data[pivot_pos]);
        self.pivot_positions.push(pivot_pos + self.offset);
    }
}
