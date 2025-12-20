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
#include <array>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace RLTrader {

// Fixed-size circular buffer with no heap allocations
template <typename T, size_t N>
class CircularBuffer {
public:
    CircularBuffer() : count_(0), head_(0) {
        buffer_.fill(T{});
    }

    void push(const T& value) {
        buffer_[head_] = value;
        head_ = (head_ + 1) % N;
        if (count_ < N) ++count_;
    }

    // Get item by lag (0 = most recent, 1 = one before, etc.)
    [[nodiscard]] const T& get(size_t lag) const {
        if (lag >= count_) {
            throw std::out_of_range("Lag exceeds buffer size");
        }
        size_t idx = (head_ + N - 1 - lag) % N;
        return buffer_[idx];
    }

    // Get item by lag (mutable)
    T& get(size_t lag) {
        if (lag >= count_) {
            throw std::out_of_range("Lag exceeds buffer size");
        }
        size_t idx = (head_ + N - 1 - lag) % N;
        return buffer_[idx];
    }

    [[nodiscard]] size_t size() const { return count_; }
    [[nodiscard]] constexpr size_t capacity() const { return N; }
    [[nodiscard]] bool full() const { return count_ == N; }
    [[nodiscard]] bool empty() const { return count_ == 0; }

    // Access underlying array for iteration
    const std::array<T, N>& data() const { return buffer_; }

private:
    std::array<T, N> buffer_;
    size_t count_;
    size_t head_;
};

// Legacy TemporalBuffer interface for backward compatibility
template <typename T>
class TemporalBuffer {
public:
    explicit TemporalBuffer(uint32_t lags) : size_(lags + 1), count_(0), head_(0) {
        buffer_.resize(size_);
    }

    void add(const T& value) {
        buffer_[head_] = value;
        head_ = (head_ + 1) % size_;
        if (count_ < size_) ++count_;
    }

    T& get(uint32_t lag) {
        if (lag >= count_) {
            throw std::out_of_range("Lag is out of range");
        }
        size_t idx = (head_ + size_ - 1 - lag) % size_;
        return buffer_[idx];
    }

    [[nodiscard]] const T& get(uint32_t lag) const {
        if (lag >= count_) {
            throw std::out_of_range("Lag is out of range");
        }
        size_t idx = (head_ + size_ - 1 - lag) % size_;
        return buffer_[idx];
    }

private:
    uint32_t size_;
    uint32_t count_;
    uint32_t head_;
    std::vector<T> buffer_;
};

}
