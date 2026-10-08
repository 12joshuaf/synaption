#pragma once
#include <vector>
#include <thread>
#include <algorithm>
#include <cstddef>

namespace synaption {

// Minimum work size below which spawning threads isn't worth the overhead.
// Small layers (like a 4-neuron XOR hidden layer) run faster single-threaded;
// tune this down if you want to force-test multithreading on a tiny network.
constexpr size_t kParallelThreshold = 64;

// Splits [0, count) into contiguous chunks and runs fn(start, end) on each,
// across up to hardware_concurrency() threads. Blocks until all finish.
// Falls back to single-threaded execution below kParallelThreshold or on
// single-core machines.
template <typename Func>
void parallel_for(size_t count, Func&& fn) {
    if (count == 0) return;

    if (count < kParallelThreshold) {
        fn(size_t(0), count);
        return;
    }

    size_t num_threads = std::min<size_t>(
        std::max<size_t>(1, std::thread::hardware_concurrency()), count);

    if (num_threads <= 1) {
        fn(size_t(0), count);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    size_t chunk = (count + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk;
        size_t end = std::min(start + chunk, count);
        if (start >= end) break;
        threads.emplace_back([&fn, start, end]() { fn(start, end); });
    }
    for (auto& th : threads) th.join();
}

} // namespace synaption
