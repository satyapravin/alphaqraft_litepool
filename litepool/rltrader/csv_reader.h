// csv_reader.h
#pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <unordered_map>

namespace RLTrader {
    struct DataRow {
        DataRow(const DataRow& other) : id(other.id), data(other.data) {}
    
        DataRow(DataRow&& other) noexcept : id(other.id), data(std::move(other.data)) {}
    
        DataRow(long long anId, const std::unordered_map<std::string, double>& dataMap) 
            : id(anId), data(dataMap) {}

        DataRow& operator=(const DataRow& other) {
            if (this != &other) {
                id = other.id;
                data = other.data;
            }
            return *this;
        }
    
        DataRow& operator=(DataRow&& other) noexcept {
            if (this != &other) {
                id = other.id;
                data = std::move(other.data);
            }
            return *this;
        }

        long long id = 0;
        std::unordered_map<std::string, double> data{};

        double getBestBidPrice() const {
            return data.find("bids[0].price")->second;
        }

        double getBestAskPrice() const {
            return data.find("asks[0].price")->second;
        }
    };

    class CsvReader {
    private:
        class Iterator {
        private:
            const std::vector<DataRow>* rowsPtr;
            size_t current = 0;  // Changed from -1 to 0

        public:
            Iterator(): rowsPtr(nullptr), current(0) {}

            void populate(const std::vector<DataRow>* ptr) {
                rowsPtr = ptr;
                current = 0;  // Initialize to 0
            }

            [[nodiscard]] bool hasNext() const {
                return rowsPtr && current < rowsPtr->size();
            }

            const DataRow& next() {
                if (!hasNext()) {
                    throw std::out_of_range("No more elements");
                }
                return (*rowsPtr)[current++];
            }

            [[nodiscard]] long long getTimeStamp() const {
                if (!rowsPtr || current == 0) {
                    throw std::runtime_error("Invalid iterator state");
                }
                return (*rowsPtr)[current - 1].id;
            }

            [[nodiscard]] double getDouble(const std::string& key) const {
                if (!rowsPtr || current == 0) {
                    throw std::runtime_error("Invalid iterator state");
                }
                return (*rowsPtr)[current - 1].data.at(key);
            }

            [[nodiscard]] const DataRow& currentRow() const {
                if (!rowsPtr || current == 0) {
                    throw std::runtime_error("Invalid iterator state");
                }
                return (*rowsPtr)[current - 1];
            }

            void reset() {
                current = 0;
            }
        };

        std::ifstream filestream;
        std::string filename;
        Iterator iterator;
        std::vector<std::string> headers;
        std::vector<DataRow> rows;
        static std::vector<double> parseLineToDoubles(const std::string& line);
        void readCSV(int start_line);
        bool more_data;
        int start_read;
        int max_read;
        int num_reads;

    public:
        CsvReader(const std::string& filename, int start_read, int max_read);
        ~CsvReader() {
            if (filestream.is_open()) {
                filestream.close();
            }
        }
        
        // Delete copy constructor and assignment
        CsvReader(const CsvReader&) = delete;
        CsvReader& operator=(const CsvReader&) = delete;
        
        // Allow move
        CsvReader(CsvReader&&) noexcept = default;
        CsvReader& operator=(CsvReader&&) noexcept = default;

        bool hasNext();
        const DataRow& next();
        const DataRow& current() const;
        long long getTimeStamp() const;
        double getDouble(const std::string& keyname) const;
        void reset();
    };
}
