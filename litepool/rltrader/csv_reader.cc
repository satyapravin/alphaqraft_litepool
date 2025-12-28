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

#include <sstream>
#include <random>
#include <iostream>
#include <glog/logging.h>
#include "csv_reader.h"

using namespace RLTrader;

CsvReader::CsvReader(const std::string& fname, int start_read_lines):filename(fname), // NOLINT(*-pass-by-value)
                                                                                         more_data(true), start_read(start_read_lines),
                                                                                         num_reads(0), rows{} {
    // Seed RNG with random_device for unpredictable randomization
    // This is done once at construction, not on every reset
    std::random_device rd;
    rng_.seed(rd());
}

void CsvReader::parseHeaderLine(const std::string& line) {
    // Parse header line to extract column names (skip first 4 columns)
    std::istringstream headerStream(line);
    std::string header;
    std::getline(headerStream, header, ','); // Skip exchange
    std::getline(headerStream, header, ','); // Skip symbol
    std::getline(headerStream, header, ','); // Skip timestamp (will be used as ID)
    std::getline(headerStream, header, ','); // Skip local_timestamp

    headers.clear();
    while (std::getline(headerStream, header, ',')) {
        headers.push_back(header);
    }
}

bool CsvReader::hasNext() {
    bool retval = this->iterator.hasNext();
    if (!retval) {
        if (more_data) {
            // Read next line from file
            // The iterator's current is at rows.size() (exhausted)
            // After adding a line, current < rows.size() should be true
            if (readNextLine()) {
                // Re-populate iterator to ensure pointer is valid after vector might have reallocated
                // But preserve current position (don't reset to 0)
                size_t saved_current = this->iterator.getCurrent();
                this->iterator.populate(&rows);
                this->iterator.setCurrent(saved_current);
                return this->iterator.hasNext();
            } else {
                // No more data available
                more_data = false;
                return false;
            }
        } else {
            // No more data available - episode should end
            return false;
        }
    }

    return retval;
}

const DataRow& CsvReader::next() {
    return this->iterator.next();
}

const DataRow& CsvReader::current() const {
    return this->iterator.currentRow();
}

long long CsvReader::getTimeStamp() const {
    return this->iterator.getTimeStamp();
}

double CsvReader::getDouble(const std::string& keyName) const {
    return this->iterator.getDouble(keyName);
}

long long CsvReader::peekFirstTimestamp() const {
    // Peek at first row without consuming it
    // After reset, iterator is at position 0, so rows[0] is the first row
    // CRITICAL: Check if rows is empty or if iterator is not populated
    if (rows.empty()) {
        throw std::runtime_error("No data available to peek - rows is empty");
    }
    // Also check if iterator is pointing to valid data
    if (!iterator.hasNext()) {
        throw std::runtime_error("No data available to peek - iterator has no next");
    }
    // Access first row directly (iterator is at position 0 after reset)
    return rows[0].id;
}

void CsvReader::rewindIterator() {
    // Rewind iterator to beginning without resetting the whole reader
    iterator.reset();
}

void CsvReader::reset() {
    // === Reset State ===
    num_reads = 0;
    iterator.reset();
    more_data = true;
    iterator.populate(nullptr);
    // Don't clear rows here - readCSV() will clear it, avoiding double-clear and slow destruction
    
    // === Generate Random Start Position ===
    // Use persistent RNG (seeded in constructor) for consistent randomization
    std::uniform_int_distribution<> distr(0, start_read);
    int start_line = distr(rng_);
    
    // === Ensure File Is Open ===
    if (!this->filestream.is_open()) {
        this->filestream.open(filename, std::ios::in);
        if (!this->filestream.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
    }
    filestream.clear();
    
    // === Parse Header If First Time ===
    if (headers.empty()) {
        filestream.seekg(0, std::ios::beg);
        std::string header_line;
        if (!std::getline(filestream, header_line)) {
            throw std::runtime_error("Failed to read header line from: " + filename);
        }
        parseHeaderLine(header_line);
        header_end_pos = filestream.tellg();
        filestream.clear();
    }
    
    // === Seek to Random Position (Line-by-Line for Exact Positioning) ===
    filestream.seekg(header_end_pos, std::ios::beg);
    filestream.clear();
    
    // Use line-by-line seeking for exact positioning (required for backtesting)
    std::string line;
    for (int i = 0; i < start_line && std::getline(filestream, line); ++i) {
        // Skip lines until we reach the random start position
    }
    
    // === Read First Line ===
    // Clear rows and read the first line at the random start position
    std::vector<DataRow>().swap(rows);
    readNextLine();
    
    this->iterator.populate(&rows);
    
    // === Validate Reset ===
    if (rows.empty()) {
        more_data = false;
    }
}

bool CsvReader::readNextLine() {
    if (!filestream.is_open()) {
        more_data = false;
        return false;
    }

    // Clear stream state before reading (EOF/fail bits from previous ops)
    if (filestream.eof()) {
        filestream.clear();
    }
    
    // Check stream state before reading
    if (!filestream.good() && filestream.eof()) {
        more_data = false;
        return false;
    }

    std::string line;
    if (!std::getline(filestream, line)) {
        // EOF reached
        more_data = false;
        return false;
    }

    // Parse the line
    ++num_reads;
    std::istringstream lineStream(line);
    std::string cell;

    // Skip exchange (1st column)
    if (!std::getline(lineStream, cell, ',')) {
        // Empty or malformed line - try next line
        return readNextLine();
    }

    // Skip symbol (2nd column)
    if (!std::getline(lineStream, cell, ',')) {
        return readNextLine();
    }

    // Use timestamp (3rd column) as ID
    if (!std::getline(lineStream, cell, ',')) {
        return readNextLine();
    }
    
    long long id = std::stoll(cell);
    
    // Parse remaining columns as doubles
    std::vector<double> values = parseLineToDoubles(line);

    if (values.size() != headers.size()) {
        // Malformed line - skip
        return readNextLine();
    }

    std::unordered_map<std::string, double> data;
    for (size_t i = 0; i < values.size(); ++i) {
        data[headers[i]] = values[i];
    }

    rows.emplace_back(id, data);
    return true;
}

std::vector<double> CsvReader::parseLineToDoubles(const std::string& line) {
    std::istringstream stream(line);
    std::string cell;
    std::vector<double> results;
    
    // Skip exchange (1st column)
    if (!std::getline(stream, cell, ',')) {
        return results;  // Empty line
    }
    
    // Skip symbol (2nd column)
    if (!std::getline(stream, cell, ',')) {
        return results;  // Malformed line
    }
    
    // Skip timestamp (3rd column)
    if (!std::getline(stream, cell, ',')) {
        return results;  // Malformed line
    }
    
    // Skip local_timestamp (4th column)
    if (!std::getline(stream, cell, ',')) {
        return results;  // Malformed line
    }

    // Parse remaining columns as doubles
    int col_index = 4;  // Track column index for diagnostics
    while (std::getline(stream, cell, ',')) {
            // Trim whitespace
            cell.erase(0, cell.find_first_not_of(" \t\n\r"));
            if (!cell.empty()) {
                cell.erase(cell.find_last_not_of(" \t\n\r") + 1);
            }
            
            // Skip empty cells
            if (cell.empty()) {
                results.push_back(0.0);
                col_index++;
                continue;
            }
            
        results.push_back(std::stod(cell));
        col_index++;
    }
    return results;
}

