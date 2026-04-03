#![feature(portable_simd, vec_from_fn)]
pub mod impls;
pub mod scalar_quickheap;
mod simd;
pub mod simd_quickheap;
pub mod workloads;

pub trait Heap<T> {
    fn default() -> Self;
    fn push(&mut self, t: T);
    fn pop(&mut self) -> Option<T>;
}
