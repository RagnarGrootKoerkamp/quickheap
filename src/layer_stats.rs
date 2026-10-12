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
