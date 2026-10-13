use rand::rng;
use std::time;

use quickheap::ConfigurableSimdQuickHeap;

use quickheap::Avx512;
use quickheap::buckets::vec_bucket::VecBucket;
use quickheap::pivot_strategies::MedianOfM;
use quickheap::rebalancing_strategies::NoRebalancing;

use rand::seq::SliceRandom;

fn main() {
    println!("Comparison Best / Worst Initialization");

    const REP: usize = 5;

    const N: usize = 2 << 25;
    const M: usize = 2 << 22;

    const K: usize = 16;

    let mut val_rng = fastrand::Rng::new();
    let mut shuffle_rng = rng();

    let mut best_times = [0; REP];
    let mut worst_times = [0; REP];
    let mut real_times = [0; REP];

    for k in 0..REP {
        let mut init_values = std::iter::repeat_with(|| val_rng.i64(i64::MIN..i64::MAX))
            .take(N as usize)
            .collect::<Vec<_>>();

        init_values.sort();

        let mut bad_layers = vec![];
        let mut bad_layer = vec![];
        let mut cur_bad = 0;

        let mut good_layers = vec![];
        let mut good_layer = vec![];
        let mut cur_good = 0;
        let mut cur_good_size = K;

        for i in 0..N {
            // Insert to good
            good_layer.push(init_values[i]);
            cur_good += 1;

            if cur_good == cur_good_size {
                cur_good_size *= 2;

                good_layer.shuffle(&mut shuffle_rng);
                good_layers.push(good_layer.clone());

                good_layer.clear();
                cur_good = 0;
            }

            // Insert to bad
            bad_layer.push(init_values[i]);
            cur_bad += 1;

            if cur_bad == K {
                bad_layer.shuffle(&mut shuffle_rng);
                bad_layers.push(bad_layer.clone());

                bad_layer.clear();
                cur_bad = 0;
            }
        }

        if cur_good != 0 {
            good_layer.shuffle(&mut shuffle_rng);
            good_layers.push(good_layer);
        }

        if cur_bad != 0 {
            bad_layer.shuffle(&mut shuffle_rng);
            bad_layers.push(bad_layer);
        }

        good_layers.reverse();
        bad_layers.reverse();

        let mut worst_h = ConfigurableSimdQuickHeap::<
            i64,
            VecBucket<i64>,
            Avx512,
            MedianOfM<3>,
            NoRebalancing,
            K,
            true,
            false,
        >::from_vecs(bad_layers);

        let mut best_h = ConfigurableSimdQuickHeap::<
            i64,
            VecBucket<i64>,
            Avx512,
            MedianOfM<3>,
            NoRebalancing,
            K,
            true,
            false,
        >::from_vecs(good_layers);

        init_values.shuffle(&mut shuffle_rng);
        let mut real_h = ConfigurableSimdQuickHeap::<
            i64,
            VecBucket<i64>,
            Avx512,
            MedianOfM<3>,
            NoRebalancing,
            K,
            true,
            false,
        >::from_vecs(vec![init_values]);

        // println!("Initialization finished.");

        let values = std::iter::repeat_with(|| val_rng.i64(i64::MIN..i64::MAX))
            .take(M as usize)
            .collect::<Vec<_>>();

        let start_2 = time::Instant::now();
        for val in &values {
            worst_h.pop().unwrap();
            worst_h.push(*val);
        }
        worst_times[k] = start_2.elapsed().as_nanos();
        // println!("Worst Case finished.");

        let start_1 = time::Instant::now();
        for val in &values {
            best_h.pop().unwrap();
            best_h.push(*val);
        }
        best_times[k] = start_1.elapsed().as_nanos();
        // println!("Best Case finished.");

        let e = real_h.pop().unwrap();
        real_h.push(e);

        let start_3 = time::Instant::now();
        for val in &values {
            real_h.pop().unwrap();
            real_h.push(*val);
        }
        real_times[k] = start_3.elapsed().as_nanos();
        // println!("Real Case finished.");
    }

    best_times.sort();
    worst_times.sort();
    real_times.sort();

    let norm_best = best_times[REP / 2] / M as u128;
    let norm_worst = worst_times[REP / 2] / M as u128;
    let norm_real = real_times[REP / 2] / M as u128;

    println!("Initialized with {} elements", N);
    println!("Measured with {} elements", M);
    println!("Time per Element Best Case {}ns", norm_best);
    println!("Time per Element Worst Case {}ns", norm_worst);
    println!("Time per Element Real Case {}ns", norm_real);
    println!(
        "Max Speedup Best {:.2}",
        norm_worst as f64 / norm_best as f64
    );
    println!(
        "Max Speedup Real {:.2}",
        norm_worst as f64 / norm_real as f64
    );
}
