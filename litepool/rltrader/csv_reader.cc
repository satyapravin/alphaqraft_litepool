#include <sstream>
#include <random>
#include <iostream>
#include <glog/logging.h>
#include "csv_reader.h"

using namespace RLTrader;

CsvReader::CsvReader(const std::string& fname, int start_read_lines):filename(fname), // NOLINT(*-pass-by-value)
                                                                                         more_data(true), start_read(start_read_lines),
                                                                                         num_reads(0), rows{} {
}

bool CsvReader::hasNext() {
    bool retval = this->iterator.hasNext();
    if (!retval) {
        if (more_data) {
            // Continue reading sequentially from current file position
            // (start_line parameter is ignored when headers already exist)
            // Guard: Clear EOF flag before reading to ensure we can read more data
            if (filestream.eof()) {
                filestream.clear();  // Clear EOF flag
            }
            
            // CRITICAL: Check if file stream is actually readable before calling readCSV
            // If file is at EOF and stream is bad, don't try to read
            if (!filestream.good() && filestream.eof()) {
                more_data = false;
                return false;
            }
            
            // Store current more_data state before readCSV
            bool had_more_data = more_data;
            this->readCSV(0);
            this->iterator.populate(&rows);
            
            // CRITICAL BUG FIX: If readCSV read 0 rows, more_data should be false
            // But if it's still true, we have an infinite loop risk
            // Check if rows is still empty after readCSV
            if (rows.empty() && had_more_data && more_data) {
                // This shouldn't happen, but if it does, we're stuck
                // Set more_data to false to prevent infinite loop
                more_data = false;
                return false;
            }
            
            return this->iterator.hasNext();
        }
        else {
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
    if (rows.empty()) {
        throw std::runtime_error("No data available to peek");
    }
    // Access first row directly (iterator is at position 0 after reset)
    return rows[0].id;
}

void CsvReader::rewindIterator() {
    // Rewind iterator to beginning without resetting the whole reader
    iterator.reset();
}

void CsvReader::reset() {
    // Guard: Reset state
    num_reads = 0;
    headers.clear();
    iterator.reset();
    more_data = true;
    
    // Clear iterator's pointer before clearing rows
    iterator.populate(nullptr);
    rows.clear();
    rows.shrink_to_fit();  // Release memory to prevent accumulation
    
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_int_distribution<> distr(0, start_read);
    int start_line = distr(gen);
    
    // Guard: Verify start_line is within valid range
    if (start_line < 0 || start_line > start_read) {
        throw std::runtime_error("Invalid start_line: " + std::to_string(start_line));
    }
    
    if (this->filestream.is_open()) {
        this->filestream.close();
    }
    this->filestream.open(filename, std::ios::in);
    if (!this->filestream.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    this->readCSV(start_line);
    this->iterator.populate(&rows);
    
    // Guard: Verify reset completed successfully
    if (num_reads < 0) {
        throw std::runtime_error("num_reads is negative after reset");
    }
}

void CsvReader::readCSV(int start_line) {
    if (!filestream.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    std::string line;
    bool batch_read = false;

    try {
        if (headers.empty()) {
            filestream.clear();
            filestream.seekg(0, std::ios::beg);
            if (!std::getline(filestream, line)) {
                throw std::runtime_error("Failed to read header line");
            }

            std::istringstream headerStream(line);
            std::string header;
            std::getline(headerStream, header, ','); // Skip exchange
            std::getline(headerStream, header, ','); // Skip symbol
            std::getline(headerStream, header, ','); // Skip timestamp (will be used as ID)
            std::getline(headerStream, header, ','); // Skip local_timestamp

            while (std::getline(headerStream, header, ',')) {
                headers.push_back(header);
            }

            // Skip to start_line, but handle case where file is shorter than expected
            for(int linenum = 0; linenum < start_line; ++linenum) {
                if (!std::getline(filestream, line)) {
                    // File is shorter than expected - set more_data to false and return
                    more_data = false;
                    rows.clear();
                    return;  // Exit early - no data available
                }
            }
        }

        rows.clear();
        int num_lines = 0;
        
        // Guard: Ensure file stream is in good state before reading
        if (!filestream.good() && !filestream.eof()) {
            more_data = false;
            return;  // File stream is in bad state
        }
        
        while (std::getline(filestream, line)) {
            ++num_reads;
            std::istringstream lineStream(line);
            std::string cell;

            // Skip exchange (1st column)
            if (!std::getline(lineStream, cell, ',')) {
                continue;  // Skip malformed lines
            }

            // Skip symbol (2nd column)
            if (!std::getline(lineStream, cell, ',')) {
                continue;  // Skip malformed lines
            }

            // Use timestamp (3rd column) as ID
            if (!std::getline(lineStream, cell, ',')) {
                continue;  // Skip malformed lines
            }
            long long id = std::stoll(cell);
            
            // Skip local_timestamp (4th column) in parseLineToDoubles
            std::vector<double> values = parseLineToDoubles(line);

            if (values.size() != headers.size()) {
                LOG(WARNING) << "Skipping malformed line: " << line;
                continue;  // Skip malformed lines
            }

            std::unordered_map<std::string, double> data;
            for (size_t i = 0; i < values.size(); ++i) {
                data[headers[i]] = values[i];
            }

            rows.emplace_back(id, data);

            if (++num_lines >= 2500) {
                batch_read = true;
                break;
            }
        }

        // Guard: Verify data availability
        // If we read 0 rows, we've reached end of file
        if (!batch_read) {
            more_data = false;  // End of file reached
        }
        
        // Guard: If rows is empty after reading, no more data available
        if (rows.empty() && !batch_read) {
            more_data = false;
        }
    }
    catch (const std::exception& e) {
        more_data = false;  // Exception means no more data
        throw std::runtime_error("Error reading CSV: " + std::string(e.what()));
    }
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
        try {
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
        } catch (const std::exception& e) {
            // Re-throw to be caught by outer handler
            throw;
        }
        col_index++;
    }
    return results;
}

