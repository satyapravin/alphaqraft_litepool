#pragma once
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cstdint>
#include "fixed_vector.h"

namespace RLTrader {

// Circular table for storing rows of fixed-size data
template<size_t NUM_COLS = 20>
class CircularTable {
public:
    explicit CircularTable(uint32_t rows) 
        : num_rows_(rows), current_row_(0), count_(0), buffer_(rows) {
    }

    void addRow(const FixedVector<double, NUM_COLS>& row) {
        std::copy(row.begin(), row.end(), buffer_[current_row_].begin());
        current_row_ = (current_row_ + 1) % num_rows_;
        if (count_ < num_rows_) ++count_;
    }

    // Get row by lag (0 = most recent, 1 = one before, etc.)
    [[nodiscard]] const FixedVector<double, NUM_COLS>& get(uint32_t lag) const {
        if (lag >= count_) {
            throw std::out_of_range("Lag exceeds available rows");
        }
        uint32_t idx = (current_row_ + num_rows_ - 1 - lag) % num_rows_;
        return buffer_[idx];
    }

    [[nodiscard]] uint32_t size() const { return count_; }
    [[nodiscard]] uint32_t capacity() const { return num_rows_; }
    [[nodiscard]] bool full() const { return count_ == num_rows_; }

private:
    uint32_t num_rows_;
    uint32_t current_row_;
    uint32_t count_;
    std::vector<FixedVector<double, NUM_COLS>> buffer_;
};

// Legacy TemporalTable for backward compatibility
class TemporalTable {
public:
    explicit TemporalTable(uint32_t rows) 
        : num_rows_(rows), current_row_(0), count_(0), buffer_(rows) {
    }

    void addRow(FixedVector<double, 20>& row) {
        std::copy(row.begin(), row.end(), buffer_[current_row_].begin());
        current_row_ = (current_row_ + 1) % num_rows_;
        if (count_ < num_rows_) ++count_;
    }

    [[nodiscard]] uint32_t get_lagged_row(uint32_t lag) const {
        if (lag >= count_) throw std::runtime_error("lag greater than available rows");
        return (current_row_ + num_rows_ - 1 - lag) % num_rows_;
    }

    const FixedVector<double, 20>& get(uint32_t lag) const {
        uint32_t idx = get_lagged_row(lag);
        return buffer_[idx];
    }

    [[nodiscard]] uint32_t size() const { return count_; }

private:
    uint32_t num_rows_;
    uint32_t current_row_;
    uint32_t count_;
    std::vector<FixedVector<double, 20>> buffer_;
};

}
