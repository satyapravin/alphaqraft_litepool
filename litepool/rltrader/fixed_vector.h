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
