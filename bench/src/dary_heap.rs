use crate::{
    Heap,
    workloads::{CountComparisons, CountingHeap, Elem},
};

pub struct CustomDaryHeap<T: Elem, const D: usize> {
    elements: Vec<T>,
    size: usize,
}

impl<T: Elem, const D: usize> Heap<T> for CustomDaryHeap<T, D> {
    type CountedHeap = CountingHeap<T, CustomDaryHeap<CountComparisons<T>, D>>;

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
        // self.bubble_down_it(0); ?
    }

    fn pop(&mut self) -> Option<T> {
        if self.size == 0 {
            return None;
        }

        let res = self.elements[0];

        // self.elements.swap_remove(0);
        // print!("Initial:         ");
        // self.print();
        self.elements.swap(0, self.size - 1);
        self.elements.pop();
        // print!("Removed element: ");
        // self.print();

        self.size -= 1;

        if self.size == 0 {
            return Some(res);
        }

        self.bubble_down_it(0);
        // print!("End:             ");
        // self.print();

        Some(res)
    }
}

impl<T: Elem, const D: usize> CustomDaryHeap<T, D> {
    pub fn print(&self) {
        println!("{:?}", self.elements);
    }

    fn bubble_up_it(&mut self, pos: usize) {
        if pos == 0 {
            return;
        }

        let mut parent_idx = pos;
        let mut child_idx = pos;

        while child_idx > 0 {
            parent_idx = (parent_idx - 1) / D;

            if self.elements[child_idx] >= self.elements[parent_idx] {
                break;
            }

            self.elements.swap(child_idx, parent_idx);

            child_idx = parent_idx;
        }
    }

    fn bubble_down_it(&mut self, pos: usize) {
        let mut parent_idx = pos;
        let parent = self.elements[parent_idx];

        let mut smallest_child_idx: usize;
        let mut smallest_child: T;

        let mut current_idx: usize;
        let mut current_child: T;

        while parent_idx < self.size {
            // println!("Checking parent idx: {:?} val: {:?}", parent_idx, parent);

            current_idx = D * parent_idx + 1; // Index of first child

            if current_idx >= self.size {
                return; // Parent is a leaf
            }

            smallest_child_idx = current_idx;
            smallest_child = self.elements[current_idx];

            for _ in 0..D {
                // println!("Check child: {}", current_idx);
                if current_idx >= self.size {
                    break; // No more child left
                }

                current_child = self.elements[current_idx];
                // println!("cur {:?} <? sm {:?}", current_child, smallest_child);
                if current_child < smallest_child {
                    smallest_child_idx = current_idx;
                    smallest_child = current_child;
                    // println!("New smallest child: {:?}", smallest_child);
                }

                current_idx += 1;
            }

            // println!("sm {:?} >= par {:?}", smallest_child, parent);
            if smallest_child >= parent {
                return;
            }

            self.elements.swap(smallest_child_idx, parent_idx);
            parent_idx = smallest_child_idx;

            // print!("Swapped:         ");
            // self.print();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Heap, dary_heap::CustomDaryHeap};

    #[test]
    fn test_dary_heap() {
        let mut heap: CustomDaryHeap<i32, 4> = CustomDaryHeap::<i32, 4>::default();

        heap.push(1);
        heap.push(-12);
        heap.push(-16);
        heap.push(22);
        heap.push(45);
        heap.push(12);
        heap.push(9);
        heap.push(-112);
        heap.push(-1611);
        heap.push(2);
        heap.push(14);
        heap.push(14);

        assert_eq!(heap.pop().unwrap(), -1611);
        assert_eq!(heap.pop().unwrap(), -112);
        assert_eq!(heap.pop().unwrap(), -16);
        assert_eq!(heap.pop().unwrap(), -12);
        assert_eq!(heap.pop().unwrap(), 1);
        assert_eq!(heap.pop().unwrap(), 2);
        assert_eq!(heap.pop().unwrap(), 9);
        assert_eq!(heap.pop().unwrap(), 12);
        assert_eq!(heap.pop().unwrap(), 14);
        assert_eq!(heap.pop().unwrap(), 14);
        assert_eq!(heap.pop().unwrap(), 22);
        assert_eq!(heap.pop().unwrap(), 45);

        heap.print();
    }
}
