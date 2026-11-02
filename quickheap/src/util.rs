use std::marker::PhantomData;
use xxhash_rust::xxh3::xxh3_128_with_seed;

pub trait KHash<const BYTES: usize> {
    fn new(key: &[u8], seed: u64) -> Self;
    fn base_hash_1(&self) -> u64;
    fn base_hash_2(&self) -> u64;

    fn index(&self, i: u64) -> usize {
        let h = self
            .base_hash_1()
            .wrapping_add(i.wrapping_mul(self.base_hash_2()))
            .wrapping_add(i.wrapping_mul(i).wrapping_mul(i)); // Additional, optional i^3 term to reduce collisions
        // Lemire fast range reduction
        (((h as u128) * ((BYTES * 8) as u128)) >> 64) as usize
    }
}

pub struct XxHashKHash {
    h1: u64,
    h2: u64,
}
impl<const M: usize> KHash<M> for XxHashKHash {
    fn new(key: &[u8], seed: u64) -> Self {
        let h = xxh3_128_with_seed(key, seed);
        Self {
            h1: h as u64,
            h2: (h >> 64) as u64 | 1,
        }
    }

    fn base_hash_1(&self) -> u64 {
        self.h1
    }

    fn base_hash_2(&self) -> u64 {
        self.h2
    }
}

pub trait ToBytes<const M: usize> {
    fn to_bytes(&self) -> [u8; M];
}
macro_rules! impl_bytes {
    ($t:ty) => {
        impl ToBytes<{ std::mem::size_of::<$t>() }> for $t {
            fn to_bytes(&self) -> [u8; std::mem::size_of::<$t>()] {
                (*self).to_le_bytes()
            }
        }
    };
}

impl_bytes!(i32);
impl_bytes!(i64);
impl_bytes!(u32);
impl_bytes!(u64);

macro_rules! impl_bytes_tup {
    ($t:ty) => {
        impl ToBytes<{ 2 * std::mem::size_of::<$t>() }> for ($t, $t) {
            fn to_bytes(&self) -> [u8; 2 * std::mem::size_of::<$t>()] {
                const N: usize = std::mem::size_of::<$t>();
                let mut out = [0u8; 2 * N];
                out[..N].copy_from_slice(&self.0.to_le_bytes());
                out[N..].copy_from_slice(&self.1.to_le_bytes());
                out
            }
        }
    };
}

impl_bytes_tup!(i32);
impl_bytes_tup!(i64);
impl_bytes_tup!(u32);
impl_bytes_tup!(u64);

pub struct BloomFilter<
    HASHER,
    ELEM_T,
    const K: usize,
    const FILTER_BYTES: usize,
    const ELEM_BYTES: usize,
> {
    data: Vec<u64>,
    seed: u64,
    _hash: PhantomData<HASHER>,
    _elem: PhantomData<ELEM_T>,
}

impl<
    HASHER: KHash<FILTER_BYTES>,
    ELEM_T: ToBytes<ELEM_BYTES>,
    const FILTER_BYTES: usize,
    const ELEM_BYTES: usize,
    const K: usize,
> BloomFilter<HASHER, ELEM_T, K, FILTER_BYTES, ELEM_BYTES>
{
    const VALID: () = assert!(
        FILTER_BYTES >= 1 && K >= 1,
        "FILTER_BYTES and K must be >= 1"
    );

    pub fn new(seed: u64) -> Self {
        let () = Self::VALID;
        Self {
            data: vec![0; FILTER_BYTES.div_ceil(8)], // FILTER_BYTES*8 bits
            seed,
            _hash: PhantomData,
            _elem: PhantomData,
        }
    }

    pub fn reset(&mut self) {
        self.data.fill(0);
    }

    pub fn insert(&mut self, x: ELEM_T) {
        let h = HASHER::new(&x.to_bytes(), self.seed);
        for i in 0..K {
            let pos = h.index(i as u64);
            self.data[pos / 64] |= 1u64 << (pos % 64);
        }
    }

    pub fn contains(&self, x: ELEM_T) -> bool {
        let h = HASHER::new(&x.to_bytes(), self.seed);
        (0..K).all(|i| {
            let pos = h.index(i as u64);
            (self.data[pos / 64] >> (pos % 64)) & 1 == 1
        })
    }
}

pub struct CountingBloomFilter<
    HASHER,
    ELEM_T,
    const K: usize,
    const FILTER_BYTES: usize,
    const COUNTER_BITS: usize,
    const ELEM_BYTES: usize,
> {
    data: Vec<u64>,
    seed: u64,
    _hash: PhantomData<HASHER>,
    _elem: PhantomData<ELEM_T>,
}

impl<
    HASHER: KHash<FILTER_BYTES>,
    ELEM_T: Copy + ToBytes<ELEM_BYTES>,
    const FILTER_BYTES: usize,
    const COUNTER_BITS: usize,
    const ELEM_BYTES: usize,
    const K: usize,
> CountingBloomFilter<HASHER, ELEM_T, K, FILTER_BYTES, COUNTER_BITS, ELEM_BYTES>
{
    // Power of two <= 64 => counters never straddle a word boundary.
    const VALID: () = assert!(
        FILTER_BYTES >= 1
            && K >= 1
            && COUNTER_BITS >= 1
            && COUNTER_BITS <= 64
            && COUNTER_BITS.is_power_of_two(),
        "COUNTER_BITS must be a power of two <= 64; FILTER_BYTES, K >= 1"
    );
    const MASK: u64 = u64::MAX >> (64 - COUNTER_BITS);
    const PER_WORD: usize = 64 / COUNTER_BITS;
    const COUNTERS: usize = FILTER_BYTES * 8;

    pub fn new(seed: u64) -> Self {
        let () = Self::VALID;
        Self {
            data: vec![0; Self::COUNTERS.div_ceil(Self::PER_WORD)],
            seed,
            _hash: PhantomData,
            _elem: PhantomData,
        }
    }

    pub fn reset(&mut self) {
        self.data.fill(0);
    }

    fn get(&self, pos: usize) -> u64 {
        let sh = (pos % Self::PER_WORD) * COUNTER_BITS;
        (self.data[pos / Self::PER_WORD] >> sh) & Self::MASK
    }

    fn set(&mut self, pos: usize, v: u64) {
        let sh = (pos % Self::PER_WORD) * COUNTER_BITS;
        let w = &mut self.data[pos / Self::PER_WORD];
        *w = (*w & !(Self::MASK << sh)) | (v << sh);
    }

    pub fn insert(&mut self, x: ELEM_T) {
        let h = HASHER::new(&x.to_bytes(), self.seed);
        for i in 0..K {
            let pos = h.index(i as u64);
            let c = self.get(pos);
            if c < Self::MASK {
                self.set(pos, c + 1); // saturated stays saturated
            }
        }
    }

    /// Returns false (and changes nothing) if x is definitely not contained.
    pub fn remove(&mut self, x: ELEM_T) -> bool {
        if !self.contains(x) {
            return false;
        }
        let h = HASHER::new(&x.to_bytes(), self.seed);
        for i in 0..K {
            let pos = h.index(i as u64);
            let c = self.get(pos);
            if c > 0 && c < Self::MASK {
                self.set(pos, c - 1); // never decrement a saturated counter
            }
        }
        true
    }

    pub fn contains(&self, x: ELEM_T) -> bool {
        let h = HASHER::new(&x.to_bytes(), self.seed);
        (0..K).all(|i| self.get(h.index(i as u64)) > 0)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // 1024 bits, K=4
    type Bf = BloomFilter<XxHashKHash, u64, 4, 128, 8>;

    #[test]
    fn bf_empty_contains_nothing() {
        let f = Bf::new(0);
        assert!((0..1000u64).all(|x| !f.contains(x)));
    }

    #[test]
    fn bf_no_false_negatives() {
        let mut f = Bf::new(0);
        (0..100u64).for_each(|x| f.insert(x));
        assert!((0..100u64).all(|x| f.contains(x)));
    }

    #[test]
    fn bf_sets_between_1_and_k_bits() {
        let mut f = Bf::new(0);
        f.insert(123);
        let ones: u32 = f.data.iter().map(|w| w.count_ones()).sum();
        assert!((1..=4).contains(&ones));
    }

    #[test]
    fn bf_idempotent_insert() {
        let mut f = Bf::new(0);
        f.insert(7);
        let before = f.data.clone();
        f.insert(7);
        assert_eq!(before, f.data);
    }

    #[test]
    fn bf_reset_clears() {
        let mut f = Bf::new(0);
        f.insert(42);
        f.reset();
        assert!(f.data.iter().all(|&w| w == 0));
        assert!(!f.contains(42));
    }

    #[test]
    fn bf_size_not_multiple_of_8_bytes() {
        let mut f = BloomFilter::<XxHashKHash, u32, 3, 5, 4>::new(1); // 40 bits
        assert_eq!(f.data.len(), 1);
        (0..10u32).for_each(|x| f.insert(x));
        assert!(f.data[0] >> 40 == 0);
        assert!((0..10u32).all(|x| f.contains(x)));
    }

    #[test]
    fn bf_large_filter_uses_all_words() {
        let mut f = BloomFilter::<XxHashKHash, u64, 3, 1024, 8>::new(0);
        assert_eq!(f.data.len(), 128);
        (0..2000u64).for_each(|x| f.insert(x));
        assert!(f.data.iter().filter(|&&w| w != 0).count() > 100);
    }

    #[test]
    fn bf_seeds_differ() {
        let (mut a, mut b) = (Bf::new(1), Bf::new(2));
        a.insert(5);
        b.insert(5);
        assert_ne!(a.data, b.data);
    }

    #[test]
    fn bf_false_positive_rate_is_sane() {
        let mut f = Bf::new(0);
        (0..100u64).for_each(|x| f.insert(x));
        let fp = (1000..11_000u64).filter(|x| f.contains(*x)).count();
        assert!(fp < 300, "fp: {fp}/10000"); // expected ~1%
    }

    // 1024 counters, 4-bit (max 15), K=4
    type Cbf = CountingBloomFilter<XxHashKHash, u64, 4, 128, 4, 8>;

    fn total(f: &Cbf) -> u64 {
        (0..1024).map(|p| f.get(p)).sum()
    }

    #[test]
    fn cbf_layout() {
        let f = Cbf::new(0);
        assert_eq!(Cbf::PER_WORD, 16);
        assert_eq!(f.data.len(), 64);
        assert_eq!(Cbf::MASK, 15);
    }

    #[test]
    fn cbf_empty_contains_nothing() {
        let f = Cbf::new(0);
        assert!((0..1000u64).all(|x| !f.contains(x)));
    }

    #[test]
    fn cbf_insert_increments_k_times() {
        let mut f = Cbf::new(0);
        f.insert(123);
        assert_eq!(total(&f), 4);
        assert!(f.contains(123));
    }

    #[test]
    fn cbf_remove_restores_empty() {
        let mut f = Cbf::new(0);
        f.insert(42);
        assert!(f.remove(42));
        assert!(f.data.iter().all(|&w| w == 0));
        assert!(!f.contains(42));
    }

    #[test]
    fn cbf_duplicate_insert_needs_two_removes() {
        let mut f = Cbf::new(0);
        f.insert(9);
        f.insert(9);
        f.remove(9);
        assert!(f.contains(9));
        f.remove(9);
        assert!(!f.contains(9));
    }

    #[test]
    fn cbf_remove_on_empty_is_noop() {
        let mut f = Cbf::new(0);
        assert!(!f.remove(5));
        assert!(f.data.iter().all(|&w| w == 0));
    }

    #[test]
    fn cbf_remove_keeps_other_elements() {
        let mut f = Cbf::new(0);
        (0..20u64).for_each(|x| f.insert(x));
        (0..10u64).for_each(|x| {
            f.remove(x);
        });
        assert!((10..20u64).all(|x| f.contains(x)));
    }

    #[test]
    fn cbf_saturates_and_stays_saturated() {
        let mut f = Cbf::new(0);
        for _ in 0..20 {
            f.insert(7);
        }
        for _ in 0..20 {
            f.remove(7);
        }
        assert!(f.contains(7)); // saturated counters are never decremented
    }

    #[test]
    fn cbf_neighbouring_counters_independent() {
        let mut f = Cbf::new(0);
        // 15/16 straddle a word boundary
        f.set(15, 5);
        f.set(16, 9);
        f.set(15, 6);
        assert_eq!((f.get(14), f.get(15), f.get(16), f.get(17)), (0, 6, 9, 0));
        assert_eq!(f.data[0], 6u64 << 60);
        assert_eq!(f.data[1], 9);
    }

    #[test]
    fn cbf_reset_clears() {
        let mut f = Cbf::new(0);
        f.insert(3);
        f.reset();
        assert_eq!(total(&f), 0);
    }

    #[test]
    fn cbf_counter_bits_extremes() {
        let mut a = CountingBloomFilter::<XxHashKHash, u32, 2, 1, 64, 4>::new(1);
        assert_eq!(a.data.len(), 8);
        (0..5u32).for_each(|x| a.insert(x));
        assert!((0..5u32).all(|x| a.contains(x)));

        let mut b = CountingBloomFilter::<XxHashKHash, u32, 2, 4, 1, 4>::new(1);
        assert_eq!(b.data.len(), 1); // 32 one-bit counters
        b.insert(1);
        assert!(b.contains(1));
        assert!(b.remove(1));
        assert!(b.contains(1)); // Contains because of saturation
    }

    #[test]
    fn cbf_seeds_differ() {
        let (mut a, mut b) = (Cbf::new(1), Cbf::new(2));
        a.insert(5);
        b.insert(5);
        assert_ne!(a.data, b.data);
    }
}
