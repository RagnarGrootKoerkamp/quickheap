use crate::buckets::Bucket;

pub trait EqualBucketTest {
    fn check<T: Copy + Ord + Default, B: Bucket<T>>(bucket: &B) -> (bool, T);
}

pub struct EqualBucketSamplingTest<const K: usize> {}
impl<const K: usize> EqualBucketTest for EqualBucketSamplingTest<K> {
    fn check<T: Copy + Ord + Default, B: Bucket<T>>(bucket: &B) -> (bool, T) {
        let positions: Vec<usize> = (0..K)
            .map(|_| rand::random_range(0..bucket.len()))
            .collect();

        let mut sample: Vec<T> = positions.into_iter().map(|i| bucket.get(i)).collect();
        sample.sort_unstable();

        let mut most_common: T = sample[0];
        let mut last_elem: T = sample[0];

        let mut most_common_count = 0;
        let mut current_count = 0;

        for i in 0..K {
            if sample[i] == last_elem {
                current_count += 1;
                continue;
            }

            if current_count > most_common_count {
                most_common = last_elem;
                most_common_count = current_count;
            }

            current_count = 1;
            last_elem = sample[i];
        }

        let most_common_ratio: f64 = most_common_count as f64 / K as f64;

        // TODO: Try different values as threshold
        if most_common_ratio > 0.2 {
            return (true, most_common);
        }

        (false, T::default())
    }
}

#[cfg(test)]
mod tests {
    use crate::buckets::{
        Bucket,
        equal_buckets::{EqualBucketSamplingTest, EqualBucketTest},
        vec_bucket::VecBucket,
    };

    #[test]
    fn test_equal_sampling() {
        let mut b = VecBucket::<u64>::default();

        for i in 0..32 {
            b.push(i);
        }

        for _ in 0..20 {
            b.push(3);
        }

        b.print();

        let (test, elem) = EqualBucketSamplingTest::<8>::check::<u64, VecBucket<u64>>(&b);

        assert!(
            test,
            "Random sampling is not stable, try to run the test again."
        );
        assert!(elem == 3);
    }
}

// Equal-buckets:
// 1 Check if many equal elements in bucket
//   - Q: Which buckets to check? Smaller? Bigger? Something in between?
//   - Different methods possible
//   - E.g. sampling, counting, hashing, etc.
//   - In our model (no tuples) -> Could collapse to counter & number
// 2 Partition bucket into 3: less, equal, greater
//   - Use something similar to the existing partitioning
// 3 Update pivots accordingly
//   - Allow only equal elements for this bucket to be inserted
// 4 If equal -> no further partitioning necessary, removing / inserting is just simple push / pop
