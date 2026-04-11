use crate::{
    Heap,
    workloads::{CountComparisons, CountingHeap, Elem},
};
pub struct CustomBinaryHeap<T: Elem> {
    elements: Vec<T>,
    size: usize,
}

impl<T: Elem> Heap<T> for CustomBinaryHeap<T> {
    type CountedHeap = CountingHeap<T, CustomBinaryHeap<CountComparisons<T>>>;

    fn default() -> Self {
        Self {
            elements: Vec::with_capacity(1000),
            size: 0,
        }
    }

    fn push(&mut self, t: T) {
        self.elements.push(t);
        self.size += 1;
        self.bubble_up_it(self.size - 1);
    }

    fn pop(&mut self) -> Option<T> {
        if self.size == 0 {
            return None;
        }

        let res = self.elements[0];
        self.size -= 1;

        self.elements.swap_remove(0);
        self.bubble_down_it(0);

        Some(res)
    }
}

impl<T: Elem> CustomBinaryHeap<T> {
    #[inline(always)]
    fn bubble_up_it(&mut self, pos: usize) {
        if pos == 0 {
            return;
        }

        let mut parent_pos = pos;
        let mut child_pos = pos;

        while child_pos > 0 {
            parent_pos = (parent_pos - 1) / 2;

            if self.elements[child_pos] >= self.elements[parent_pos] {
                break;
            }

            self.elements.swap(child_pos, parent_pos);

            child_pos = parent_pos;
        }
    }

    #[inline(always)]
    fn bubble_down_it(&mut self, pos: usize) {
        if pos >= self.size {
            return;
        }

        let mut parent_pos = pos;
        let mut left_child;
        let mut right_child;

        while parent_pos < self.size {
            left_child = 2 * parent_pos + 1;
            right_child = 2 * parent_pos + 2;

            let smaller_left =
                left_child < self.size && self.elements[parent_pos] > self.elements[left_child];
            let smaller_right =
                right_child < self.size && self.elements[parent_pos] > self.elements[right_child];

            if !smaller_left && !smaller_right {
                return;
            }

            if !smaller_right
                || (smaller_left && self.elements[left_child] < self.elements[right_child])
            {
                self.elements.swap(parent_pos, left_child);
                parent_pos = left_child;
            } else {
                self.elements.swap(parent_pos, right_child);
                parent_pos = right_child;
            }
        }
    }
}

//     fn bubble_up_rec(&mut self, pos: usize) {
//         if pos == 0 {
//             return;
//         }

//         let parent_pos = (pos - 1) / 2;

//         if self.elements[pos] < self.elements[parent_pos] {
//             self.swap(pos, parent_pos);
//             self.bubble_up_rec(parent_pos);
//         }
//     }
//     fn bubble_down_rec(&mut self, pos: usize) {
//         if pos >= self.size {
//             return;
//         }

//         let left_child = 2 * pos + 1;
//         let right_child = 2 * pos + 2;

//         let smaller_left = left_child < self.size && self.elements[pos] > self.elements[left_child];
//         let smaller_right =
//             right_child < self.size && self.elements[pos] > self.elements[right_child];

//         if !smaller_left && !smaller_right {
//             return;
//         }

//         if !smaller_right
//             || (smaller_left && self.elements[left_child] < self.elements[right_child])
//         {
//             self.swap(pos, left_child);
//             self.bubble_down_rec(left_child);
//         } else {
//             self.swap(pos, right_child);
//             self.bubble_down_rec(right_child);
//         }
//     }
// }
