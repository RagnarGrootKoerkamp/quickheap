use std::fs::File;
use std::hint::black_box;
use std::io::BufWriter;
use std::io::Write;
use std::time;

use bench::workloads::Elem;
use quickheap::partitioning::Partition;
use quickheap::partitioning::VecPartitioning;
use quickheap::pivots::MedianOfM;
use quickheap::simd::{self, SPLIT_PIVOTS_CUR_AND_NEXT};
use quickheap::{ElemT, SimdElem};
use quickheap_dev::ElemConstraint;
use quickheap_dev::buckets::block_arena::BlockArena;
use quickheap_dev::buckets::block_partitioning::FlatPartitioning;
use quickheap_dev::buckets::{BlockedBucket, Bucket};
use quickheap_dev::buckets::{
    block_partitioning::BlockedPartitioning, list_block_bucket::ListBlockBucket,
    vec_block_bucket::VecBlockBucket, vec_bucket::VecBucket,
};

const REPEATS: usize = 5;

const EQUAL: bool = false;

fn generate_layer_input<T: Elem>(n: usize) -> (Vec<T>, T, usize) {
    let mut rng = fastrand::Rng::new();
    let stride = T::stride();

    let values: Vec<T>;
    if EQUAL {
        let val = <T as Elem>::clamp(T::from(rng.u64(..stride)));
        values = vec![val; n];
    } else {
        values = std::iter::repeat_with(|| <T as Elem>::clamp(T::from(rng.u64(..stride))))
            .take(n)
            .collect::<Vec<_>>();
    }

    let (pivot, pivot_pos) = MedianOfM::<3>::pick(&values);

    (values, pivot, pivot_pos)
}

// All prep functions take `data` by value. That way the returned closure owns
// everything and does not borrow from the caller, so it can be returned from
// `prep` as a single concrete type `R` (no higher-ranked lifetime problems).
//
// The closures return their layers, so that dropping them happens *after*
// the timed region and the results can be passed through `black_box`.

fn prep_vec_partition<T: ElemT + Elem, S: SimdElem<T>>(
    data: Vec<T>,
    pivot: T,
    pivot_pos: usize,
    _: *mut BlockArena<T, 240, 256>,
) -> impl FnOnce() -> (Vec<T>, Vec<T>) {
    let mut cur_layer = data;
    let mut next_layer: Vec<T> = vec![];

    move || {
        VecPartitioning::<T, S>::partition::<SPLIT_PIVOTS_CUR_AND_NEXT>(
            &mut cur_layer,
            &mut next_layer,
            pivot,
            pivot_pos,
        );
        (cur_layer, next_layer)
    }
}

fn prep_flat_partition<T: ElemT + Elem + Default, S: SimdElem<T>>(
    data: Vec<T>,
    pivot: T,
    pivot_pos: usize,
    arena: *mut BlockArena<T, 240, 256>,
) -> impl FnOnce() -> (VecBucket<T>, VecBucket<T>) {
    let mut cur_layer = VecBucket::<T>::default(arena);
    let mut next_layer = VecBucket::<T>::default(arena);

    for d in &data {
        <VecBucket<T> as Bucket<T, 240, 256>>::push(&mut cur_layer, *d);
    }

    move || {
        FlatPartitioning::<S>::partition::<SPLIT_PIVOTS_CUR_AND_NEXT>(
            &mut cur_layer,
            &mut next_layer,
            pivot,
            pivot_pos,
        );
        (cur_layer, next_layer)
    }
}

fn prep_vec_block_partition<T: Default + ElemT + Elem, S: SimdElem<T>>(
    data: Vec<T>,
    pivot: T,
    pivot_pos: usize,
    arena: *mut BlockArena<T, 240, 256>,
) -> impl FnOnce() -> (VecBlockBucket<T, 240, 256>, VecBlockBucket<T, 240, 256>) {
    let mut cur_layer = VecBlockBucket::<T, 240, 256>::default(arena);
    let mut next_layer = VecBlockBucket::<T, 240, 256>::default(arena);

    for d in &data {
        cur_layer.push(*d);
    }

    move || {
        BlockedPartitioning::<S, 240, 256>::partition::<SPLIT_PIVOTS_CUR_AND_NEXT>(
            &mut cur_layer,
            &mut next_layer,
            pivot,
            pivot_pos,
        );
        (cur_layer, next_layer)
    }
}

fn prep_list_block_partition<T: Default + ElemT + Elem + ElemConstraint, S: SimdElem<T>>(
    data: Vec<T>,
    pivot: T,
    pivot_pos: usize,
    arena: *mut BlockArena<T, 240, 256>,
) -> impl FnOnce() -> (ListBlockBucket<T, 240, 256>, ListBlockBucket<T, 240, 256>) {
    let mut cur_layer = ListBlockBucket::<T, 240, 256>::default(arena);
    let mut next_layer = ListBlockBucket::<T, 240, 256>::default(arena);

    for d in &data {
        cur_layer.push(*d);
    }

    move || {
        BlockedPartitioning::<S, 240, 256>::partition::<SPLIT_PIVOTS_CUR_AND_NEXT>(
            &mut cur_layer,
            &mut next_layer,
            pivot,
            pivot_pos,
        );
        (cur_layer, next_layer)
    }
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    let mid = samples.len() / 2;
    if samples.len() % 2 == 0 {
        (samples[mid - 1] + samples[mid]) / 2.0
    } else {
        samples[mid]
    }
}

// Generics: T (element), S (SIMD backend, only used for its name),
// P (prep function), R (closure returned by prep), O (closure output).
// Call as `bench_partition::<i64, simd::Avx2, _, _, _>(...)`.
//
// Runs REPEATS measurements on fresh copies of `data` and writes only the
// median (ns per element) to the file.
fn bench_partition<T: Elem + Copy, S, P, R, O>(
    n: usize,
    data: &[T],
    pivot: T,
    pivot_pos: usize,
    arena: *mut BlockArena<T, 240, 256>,
    name: &str,
    writer: &mut BufWriter<File>,
    prep: P,
) where
    P: Fn(Vec<T>, T, usize, *mut BlockArena<T, 240, 256>) -> R,
    R: FnOnce() -> O,
{
    let mut samples: Vec<f64> = Vec::with_capacity(REPEATS);

    for _ in 0..REPEATS {
        // Setup (not timed)
        let f = prep(data.to_vec(), pivot, pivot_pos, arena);

        // Timed region
        let now = time::Instant::now();
        let out = black_box(f());
        let elapsed = now.elapsed().as_nanos();

        // Drop the layers outside the timed region
        drop(black_box(out));

        samples.push(black_box(elapsed) as f64 / n as f64);
    }

    let norm = median(&mut samples);

    let line = format!(
        "{},{},{},{},{},{:.2}\n",
        std::any::type_name::<T>(),
        std::any::type_name::<S>(),
        "OnlyPartition",
        n,
        name,
        norm
    )
    .replace("quickheap::simd::", "");

    writer.write_all(line.as_bytes()).unwrap();
}

fn run<T: Default + ElemConstraint + ElemT + Elem, S: SimdElem<T>>(writer: &mut BufWriter<File>) {
    let mut free_arena = BlockArena::<T, 240, 256>::new();
    let ns: Vec<usize> = (10..=26).step_by(2).map(|i| (2usize).pow(i)).collect();

    for n in ns {
        let (data, pivot, pivot_pos) = generate_layer_input::<T>(n);

        bench_partition::<T, S, _, _, _>(
            n,
            &data,
            pivot,
            pivot_pos,
            &mut free_arena,
            "vec_partition",
            writer,
            prep_vec_partition::<T, S>,
        );

        bench_partition::<T, S, _, _, _>(
            n,
            &data,
            pivot,
            pivot_pos,
            &mut free_arena,
            "vec_bucket_partition",
            writer,
            prep_flat_partition::<T, S>,
        );

        bench_partition::<T, S, _, _, _>(
            n,
            &data,
            pivot,
            pivot_pos,
            &mut free_arena,
            "vec_block_partition",
            writer,
            prep_vec_block_partition::<T, S>,
        );

        bench_partition::<T, S, _, _, _>(
            n,
            &data,
            pivot,
            pivot_pos,
            &mut free_arena,
            "list_block_partition",
            writer,
            prep_list_block_partition::<T, S>,
        );

        // TODO: Memory Insights
        writer.flush().unwrap();
    }
}

fn main() {
    #[cfg(target_arch = "aarch64")]
    let suff = "_neon";

    #[cfg(not(target_arch = "aarch64"))]
    let suff = "_avx";

    let path = String::from(env!("CARGO_MANIFEST_DIR"))
        + "/evals/data/experiments/partitions"
        + suff
        + ".csv";
    let file = File::create(path).unwrap();
    let mut writer = BufWriter::with_capacity(1 << 15, file);

    writer
        .write_all("elem_type,simd_backend,workload,n,heap,nano_seconds\n".as_bytes())
        .unwrap();

    #[cfg(target_feature = "avx2")]
    {
        run::<i32, simd::Avx2>(&mut writer);
        run::<i64, simd::Avx2>(&mut writer);
    }

    #[cfg(target_feature = "avx512f")]
    {
        run::<i32, simd::Avx512>(&mut writer);
        run::<i64, simd::Avx512>(&mut writer);
    }

    #[cfg(target_arch = "aarch64")]
    {
        run::<i32, simd::Neon>(&mut writer);
        run::<i64, simd::Neon>(&mut writer);
    }
}
