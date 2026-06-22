use std::collections::LinkedList;

pub struct Block<T, const K: usize> {
    size: usize,
    data: [T; K],
}

impl<T: Default + Copy, const K: usize> Block<T, K> {
    fn default() -> Self {
        Self {
            size: 0,
            data: [T::default(); K],
        }
    }

    fn insert(&mut self, elem: T) {
        assert!(self.size < K);
        self.data[self.size] = elem;
        self.size += 1;
    }

    fn remove(&mut self, i: usize) -> T {
        assert!(i < K);
        assert!(self.size > 0);

        let elem = self.data[i];
        self.data[i] = self.data[self.size - 1];
        self.size -= 1;

        elem
    }

    fn get(&self, i: usize) -> T {
        assert!(i < self.size);
        self.data[i]
    }

    fn size(&self) -> usize {
        self.size
    }

    fn full(&self) -> bool {
        self.size == K
    }
}

pub trait Bucket<T> {
    fn default() -> Self;
    fn insert(&mut self, elem: T);
    fn join(&mut self, other: &mut Self);
    // fn partition(&mut self) -> (Self, Self);
}

impl<T: Copy + Default> Bucket<T> for Vec<T> {
    fn insert(&mut self, elem: T) {
        self.push(elem);
    }

    fn join(&mut self, other: &mut Self) {
        self.extend(*other);
    }
    // fn partition(&mut self) -> ;
}

impl<T: Copy + Default, const K: usize> Bucket<T> for Vec<Block<T, K>> {
    fn insert(&mut self, elem: T) {
        let full = self[self.len()].full();
        if !full {
            let len = self.len();
            self[len - 1].insert(elem);
            return;
        }
        self.push(Block::<T, K>::default());
    }

    fn join(&mut self, other: Self) {}
}

impl<T: Copy + Default, const K: usize> Bucket<T> for LinkedList<Block<T, K>> {
    fn insert(&mut self, elem: T) {
        //self.
    }
    fn join(&mut self, other: &mut Self) {
        self.append(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_block() {
        let mut block = Block::<i32, 16>::default();
        assert!(block.size() == 0);

        for i in 0..16 {
            block.insert(i);
        }

        for i in 0..16 {
            assert!(block.get(i) == i as i32)
        }
    }

    #[test]
    fn delete_from_block() {
        let mut block = Block::<i32, 16>::default();
        assert!(block.size() == 0);

        for i in 0..16 {
            block.insert(i);
        }

        // 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
        block.remove(2);
        // 0 1 15 3 4 5 6 7 8 9 10 11 12 13 14
        block.remove(14);
        // 0 1 15 3 4 5 6 7 8 9 10 11 12 13
        block.remove(6);
        // 0 1 15 3 4 5 13 7 8 9 10 11 12

        assert!(block.size() == 13);
        assert!(block.get(0) == 0);
        assert!(block.get(2) == 15);
        assert!(block.get(6) == 13);
        assert!(block.get(7) == 7);
        assert!(block.get(12) == 12);
    }
}
