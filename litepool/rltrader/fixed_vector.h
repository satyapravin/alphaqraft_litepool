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
#include <cstdint>
#include <array>

namespace RLTrader {
    template<typename T, size_t N>
    class FixedVector {
        std::array<T, N> data;

    public:
        FixedVector() { data.fill(T{}); }
        
        T& operator[](size_t i) { return data[i]; }
        const T& operator[](size_t i) const { return data[i]; }
        
        [[nodiscard]] constexpr size_t size() const { return N; }
        [[nodiscard]] constexpr size_t capacity() const { return N; }
        [[nodiscard]] bool empty() const { return N == 0; }

        // Iterator support
        T* begin() { return data.data(); }
        T* end() { return data.data() + N; }
        [[nodiscard]] const T* begin() const { return data.data(); }
        [[nodiscard]] const T* end() const { return data.data() + N; }
        
        // Data access
        T* data_ptr() { return data.data(); }
        [[nodiscard]] const T* data_ptr() const { return data.data(); }
    };
}
