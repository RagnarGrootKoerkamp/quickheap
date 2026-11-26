use quickheap::pivots;

use std::fs::File;
use std::io::BufWriter;
use std::io::Write;

use bench::workloads::Elem;

const REPEATS: usize = 201;
const BAD_THRESHOLD: f64 = 0.9;

fn generate_layer_input<T: Elem>(n: usize) -> Vec<T> {
    let mut rng = fastrand::Rng::new();
    let stride = T::stride();

    std::iter::repeat_with(|| <T as Elem>::clamp(T::from(rng.u64(..stride))))
        .take(n)
        .collect()
}

/// Binary Shannon entropy (in bits) of a split into `a` and `b` elements.
/// 1.0 = perfectly balanced, 0.0 = completely degenerate.
fn binary_entropy(a: f64, b: f64) -> f64 {
    let total = a + b;
    if total == 0.0 {
        return 0.0;
    }
    let term = |c: f64| {
        if c == 0.0 {
            0.0
        } else {
            let p = c / total;
            -p * p.log2()
        }
    };
    term(a) + term(b)
}

struct Stats {
    mean: f64,
    p05: f64,
    frac_bad: f64,
}

fn summarize(entropies: &mut Vec<f64>) -> Stats {
    let k = entropies.len() as f64;
    let mean = entropies.iter().sum::<f64>() / k;
    let var = entropies.iter().map(|h| (h - mean).powi(2)).sum::<f64>() / (k - 1.0);

    let frac_bad = entropies.iter().filter(|&&h| h < BAD_THRESHOLD).count() as f64 / k;

    entropies.sort_by(|a, b| a.total_cmp(b));
    let p05 = entropies[((entropies.len() as f64) * 0.05) as usize];

    Stats {
        mean,
        p05,
        frac_bad,
    }
}

fn run_entropy<P: pivots::VecPivotStrategy<i64>>(
    name: &str,
    values: &Vec<i64>,
    writer: &mut BufWriter<File>,
) {
    let n = values.len();
    let mut entropies = Vec::with_capacity(REPEATS);

    for _ in 0..REPEATS {
        let (p, _pos) = P::pick(values);

        let mut a = 0usize; // elements smaller than the pivot
        let mut b = 0usize; // elements larger than the pivot
        for &val in values {
            if val < p {
                a += 1;
            } else if val > p {
                b += 1;
            }
        }
        entropies.push(binary_entropy(a as f64, b as f64));
    }

    let s = summarize(&mut entropies);
    writeln!(
        writer,
        "{name},{n},{:.3},{:.3},{:.3}",
        s.mean, s.p05, s.frac_bad
    )
    .unwrap();
}

fn main() {
    let path = String::from(env!("CARGO_MANIFEST_DIR")) + "/evals/data/experiments/entropy.csv";
    let file = File::create(path).unwrap();
    let mut writer = BufWriter::with_capacity(1 << 15, file);
    writer
        .write_all("strategy,n,mean,p05,frac_bad\n".as_bytes())
        .unwrap();

    let ns: Vec<usize> = (10..=25).map(|i| 1usize << i).collect();

    for n in ns {
        let vals = generate_layer_input::<i64>(n);

        run_entropy::<pivots::MedianOfM<3>>("MedianOfM3", &vals, &mut writer);
        run_entropy::<pivots::MedianOfM<7>>("MedianOfM7", &vals, &mut writer);
        run_entropy::<pivots::LayerSizeDependentPivot>("LayerSizeDependent", &vals, &mut writer);

        writer.flush().unwrap();
    }
}
