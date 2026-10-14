#[cfg(feature = "layer_stats")]
pub mod layer_stats {
    use std::cell::RefCell;

    thread_local! {
        static HIST: RefCell<Vec<u64>> = RefCell::new(vec![0; 256]);
    }

    #[inline(always)]
    pub fn record(layers: usize) {
        HIST.with(|h| {
            let mut h = h.borrow_mut();
            if layers >= h.len() {
                h.resize(layers + 1, 0);
            }
            h[layers] += 1;
        });
    }

    pub fn reset() {
        HIST.with(|h| h.borrow_mut().iter_mut().for_each(|x| *x = 0));
    }

    /// Median number of layers over all recorded operations.
    pub fn median() -> usize {
        HIST.with(|h| {
            let h = h.borrow();
            let total: u64 = h.iter().sum();
            if total == 0 {
                return 0;
            }
            let mut acc = 0;
            for (layers, &cnt) in h.iter().enumerate() {
                acc += cnt;
                if acc * 2 >= total {
                    return layers as usize;
                }
            }
            unreachable!()
        })
    }
}

#[cfg(feature = "partition_stats")]
pub mod partition_stats {
    use std::cell::RefCell;

    #[derive(Debug, Clone, Copy, Default)]
    pub struct PartitionStats {
        pub partitions: usize,
        total_partitioned_elements: u128,
        pub redos_cur_layer_empty: usize,
        pub redos_next_layer_empty: usize,
        pub equal_buckets_found: usize,
        pub avg_num_elem_partitioned: usize,
    }

    thread_local! {
        static DATA: RefCell<PartitionStats> = RefCell::new(PartitionStats { partitions: 0, redos_cur_layer_empty: 0, redos_next_layer_empty: 0, equal_buckets_found: 0, total_partitioned_elements: 0, avg_num_elem_partitioned: 0 });
    }

    pub fn reset() {
        DATA.with(|d| {
            let mut d = d.borrow_mut();
            d.partitions = 0;
            d.redos_cur_layer_empty = 0;
            d.redos_next_layer_empty = 0;
            d.equal_buckets_found = 0;
        });
    }

    #[inline(always)]
    pub fn partition(elems: usize) {
        DATA.with(|d| {
            let mut d = d.borrow_mut();
            d.partitions += 1;
            d.total_partitioned_elements += elems as u128;
        });
    }

    #[inline(always)]
    pub fn equal_bucket() {
        DATA.with(|d| {
            let mut d = d.borrow_mut();
            d.equal_buckets_found += 1;
        });
    }

    #[inline(always)]
    pub fn redo_cur_layer() {
        DATA.with(|d| {
            let mut d = d.borrow_mut();
            d.redos_cur_layer_empty += 1;
        });
    }

    #[inline(always)]
    pub fn redo_next_layer() {
        DATA.with(|d| {
            let mut d = d.borrow_mut();
            d.redos_next_layer_empty += 1;
        });
    }

    pub fn result() -> PartitionStats {
        DATA.with(|d| {
            let mut d = d.borrow_mut();
            d.avg_num_elem_partitioned =
                (d.total_partitioned_elements / d.partitions as u128) as usize;
            *d
        })
    }
}
