use fastrand;
use std::mem::MaybeUninit;

pub trait VecPivotStrategy<T: Copy + Ord> {
    fn pick(layer: &[T]) -> (T, usize);
}

pub struct MedianOfM<const M: usize>;

impl<T: Copy + Ord, const M: usize> VecPivotStrategy<T> for MedianOfM<M> {
    #[inline(always)]
    fn pick(layer: &[T]) -> (T, usize) {
        let n = layer.len();
        let mut rng = fastrand::Rng::new();
        let mut pivots: [(T, usize); M] = std::array::from_fn(|_| {
            let pos = rng.usize(..n);
            (layer[pos], pos)
        });
        let (_, med, _) = pivots.select_nth_unstable(M / 2);
        *med
    }
}

pub struct LayerSizeDependentPivot;

impl LayerSizeDependentPivot {
    const S_MAX: usize = 512;

    #[inline(always)]
    fn sample_size(n: usize) -> usize {
        match n {
            ..=256 => 3,
            ..=1024 => 7,
            ..=4096 => 9,
            _ => (((n as f64).cbrt() * 0.75) as usize | 1).clamp(13, Self::S_MAX),
        }
    }

    #[inline(always)]
    fn pick_large<T: Copy + Ord>(layer: &[T], s: usize) -> (T, usize) {
        let n = layer.len();
        let mut rng = fastrand::Rng::new();
        let mut buf = [const { MaybeUninit::<(T, usize)>::uninit() }; Self::S_MAX];
        for slot in &mut buf[..s] {
            let pos = rng.usize(..n);
            slot.write((layer[pos], pos));
        }
        // SAFETY: the first `s` slots were initialised above.
        let sample =
            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<(T, usize)>(), s) };
        let (_, med, _) = sample.select_nth_unstable(s / 2);
        *med
    }
}

impl<T: Copy + Ord> VecPivotStrategy<T> for LayerSizeDependentPivot {
    #[inline(always)]
    fn pick(layer: &[T]) -> (T, usize) {
        match Self::sample_size(layer.len()) {
            3 => MedianOfM::<3>::pick(layer),
            7 => MedianOfM::<7>::pick(layer),
            s => Self::pick_large(layer, s),
        }
    }
}
