#include <sstream>
#include <random>
#include <glog/logging.h>
#include "csv_reader.h"

using namespace RLTrader;

CsvReader::CsvReader(const std::string& fname, int start_read_lines, int max_read_lines):filename(fname), // NOLINT(*-pass-by-value)
                                                                                         more_data(true), start_read(start_read_lines),
                                                                                         max_read(max_read_lines), num_reads(0), rows{} {
}

bool CsvReader::hasNext() {
    bool retval = this->iterator.hasNext();
    if (!retval) {
        if (more_data) {
            // Continue reading sequentially from current file position
            // (start_line parameter is ignored when headers already exist)
            this->readCSV(0);
            this->iterator.populate(&rows);
            return this->iterator.hasNext();
        }
        else {
            return more_data;
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

void CsvReader::reset() {
    num_reads = 0;
    headers.clear();
    iterator.reset();
    more_data = true;
    
    // Clear iterator's pointer before clearing rows
    iterator.populate(nullptr);
    rows = std::vector<DataRow>();
    
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_int_distribution<> distr(0, start_read);
    int start_line = distr(gen);
    
    if (this->filestream.is_open()) {
        this->filestream.close();
    }
    this->filestream.open(filename, std::ios::in);
    this->readCSV(start_line);
    this->iterator.populate(&rows);
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
            std::getline(headerStream, header, ','); // Skip the first header (ID)

            while (std::getline(headerStream, header, ',')) {
                headers.push_back(header);
            }

            for(int linenum = 0; linenum < start_line; ++linenum) {
                if (!std::getline(filestream, line)) {
                    throw std::runtime_error("Failed to skip to start line");
                }
            }
        }

        rows.clear();
        int num_lines = 0;
        while (std::getline(filestream, line)) {
            ++num_reads;
            std::istringstream lineStream(line);
            std::string cell;

            if (!std::getline(lineStream, cell, ',')) {
                continue;  // Skip malformed lines
            }

            long long id = std::stoll(cell);
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

            if (num_reads > max_read) {
                break;
            }
        }

        if (!batch_read) {
            more_data = false;
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Error reading CSV: " + std::string(e.what()));
    }
}

std::vector<double> CsvReader::parseLineToDoubles(const std::string& line) {
    std::istringstream stream(line);
    std::string cell;
    std::vector<double> results;
    std::getline(stream, cell, ','); // Skip the first token (ID)

    while (std::getline(stream, cell, ',')) {
        results.push_back(std::stod(cell));
    }
    return results;
}

