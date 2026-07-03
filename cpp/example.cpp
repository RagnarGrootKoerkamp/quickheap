#include "quickheap.hpp"

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

int main() {
  // Basic push/pop, mirroring the Rust doctest in src/lib.rs.
  {
    quickheap::Heap<std::uint64_t> q;
    q.push(4);
    q.push(1);
    q.push(7);
    std::cout << "pop -> " << q.pop() << " (expected 1)\n";
    q.push(7);
    q.push(3);
    std::cout << "pop -> " << q.pop() << " (expected 3)\n";
    std::cout << "pop -> " << q.pop() << " (expected 4)\n";
    std::cout << "pop -> " << q.pop() << " (expected 7)\n";
    std::cout << "pop -> " << q.pop() << " (expected 7)\n";
    std::cout << "empty -> " << std::boolalpha << q.empty() << "\n";
  }

  // Sort a batch of random 32-bit values by pushing them all, then popping
  // them off in ascending order.
  {
    std::mt19937 rng(42);
    std::uniform_int_distribution<std::int32_t> dist(-1000, 1000);

    quickheap::Heap<std::int32_t> q;
    for (int i = 0; i < 20; ++i) {
      q.push(dist(rng));
    }

    std::cout << "sorted: ";
    while (auto v = q.try_pop()) {
      std::cout << *v << " ";
    }
    std::cout << "\n";
  }

  return 0;
}
