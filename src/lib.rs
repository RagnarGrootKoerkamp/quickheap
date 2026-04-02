#![feature(portable_simd)]
mod impls;
mod simd;
pub mod simd_quickheap;

pub trait Heap<T> {
    fn default() -> Self;
    fn push(&mut self, t: T);
    fn pop(&mut self) -> Option<T>;
}
