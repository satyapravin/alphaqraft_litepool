// Copyright 2024 Alphaqraft
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <atomic>

#include "orderbook.h"

#if defined(_MSC_VER)
    #include <intrin.h>
    #define CPU_PAUSE() _mm_pause()
#elif defined(__x86_64__) || defined(__i386__)
    #include <immintrin.h>
    #define CPU_PAUSE() _mm_pause()
#elif defined(__aarch64__)
    #define CPU_PAUSE() asm volatile("yield" ::: "memory")
#else
    #define CPU_PAUSE() std::this_thread::yield()
#endif

namespace RLTrader {
class alignas(64) LockFreeOrderBookBuffer {
private:
    static constexpr size_t BUFFER_SIZE = 100;
    static constexpr size_t CACHE_LINE_SIZE = 64;

    struct alignas(CACHE_LINE_SIZE) Slot {
        OrderBook book;
        std::atomic<size_t> sequence;

        Slot() : sequence(0) {
        }
    };

    struct alignas(CACHE_LINE_SIZE) AlignedAtomic {
        std::atomic<size_t> value;
        char padding[CACHE_LINE_SIZE - sizeof(std::atomic<size_t>)]{};

        AlignedAtomic() : value(0) {}
    };

    alignas(CACHE_LINE_SIZE) std::array<Slot, BUFFER_SIZE> buffer;
    AlignedAtomic write_index;
    AlignedAtomic read_index;

public:
    LockFreeOrderBookBuffer() {
        // Initialize all slots in the buffer
        for (size_t i = 0; i < BUFFER_SIZE; ++i) {
            buffer[i].sequence.store(i, std::memory_order_relaxed);
        }
        write_index.value.store(0, std::memory_order_relaxed);
        read_index.value.store(0, std::memory_order_relaxed);
    }

    OrderBook& get_write_slot(size_t& current_write) {
        while (true) {
            current_write = write_index.value.load(std::memory_order_relaxed);
            size_t sequence = buffer[current_write].sequence.load(std::memory_order_acquire);

            if (sequence == current_write) {
                return buffer[current_write].book;
            }

            #if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
            #elif defined(__aarch64__) || defined(_M_ARM64)
                asm volatile("yield");
            #else
                std::atomic_thread_fence(std::memory_order_acquire);
            #endif
        }
    }

    void commit_write(size_t current_write) {
        buffer[current_write].sequence.store(current_write + BUFFER_SIZE,
                                           std::memory_order_release);
        write_index.value.store((current_write + 1) % BUFFER_SIZE,
                              std::memory_order_release);
    }

    OrderBook& get_read_slot(size_t& current_read) {
        while (true) {
            current_read = read_index.value.load(std::memory_order_relaxed);
            size_t sequence = buffer[current_read].sequence.load(std::memory_order_acquire);

            if (sequence == current_read + BUFFER_SIZE) {
                return buffer[current_read].book;
            }

            #if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
            #elif defined(__aarch64__) || defined(_M_ARM64)
                asm volatile("yield");
            #else
                std::atomic_thread_fence(std::memory_order_acquire);
            #endif
        }
    }

    void commit_read(size_t current_read) {
        buffer[current_read].sequence.store(current_read,
                                          std::memory_order_release);
        read_index.value.store((current_read + 1) % BUFFER_SIZE,
                             std::memory_order_release);
    }
};

}
