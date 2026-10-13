pub trait MultiwayPartition<S, T, B> {
    fn partition(cur_layer: &mut B, next_layers: &[&mut B], pivots: &[T]);
}
