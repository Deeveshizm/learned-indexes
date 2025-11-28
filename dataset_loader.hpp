#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <ctime>
#include <cstring>

namespace dataset_loader {

// Parse NASA log timestamp to Unix epoch
// Format: [01/Jul/1995:00:00:01 -0400]
time_t parse_nasa_timestamp(const std::string& ts_str) {
    if (ts_str.length() < 20) return 0;
    
    struct tm tm_info = {};
    
    // Extract day, month, year, time
    int day = std::stoi(ts_str.substr(0, 2));
    std::string month_str = ts_str.substr(3, 3);
    int year = std::stoi(ts_str.substr(7, 4));
    
    int hour = std::stoi(ts_str.substr(12, 2));
    int min = std::stoi(ts_str.substr(15, 2));
    int sec = std::stoi(ts_str.substr(18, 2));
    
    // Convert month string to number
    const char* months[] = {"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"};
    int month = 0;
    for (int i = 0; i < 12; i++) {
        if (month_str == months[i]) {
            month = i;
            break;
        }
    }
    
    tm_info.tm_year = year - 1900;
    tm_info.tm_mon = month;
    tm_info.tm_mday = day;
    tm_info.tm_hour = hour;
    tm_info.tm_min = min;
    tm_info.tm_sec = sec;
    tm_info.tm_isdst = -1;
    
    return mktime(&tm_info);
}

// Load NASA web logs with proper timestamp parsing
std::vector<std::pair<double, size_t>> load_nasa_logs(const std::string& filepath, size_t max_records = 0) {
    std::vector<std::pair<double, size_t>> data;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return data;
    }
    
    std::cout << "Loading NASA web log data from " << filepath << "...\n";
    
    std::string line;
    size_t line_num = 0;
    
    while (std::getline(file, line) && (max_records == 0 || data.size() < max_records)) {
        line_num++;
        
        // Extract timestamp between [ and ]
        size_t start = line.find('[');
        size_t end = line.find(']');
        
        if (start != std::string::npos && end != std::string::npos) {
            std::string timestamp_str = line.substr(start + 1, end - start - 1);
            time_t timestamp = parse_nasa_timestamp(timestamp_str);
            
            if (timestamp > 0) {
                data.emplace_back(static_cast<double>(timestamp), data.size());
            }
        }
        
        if (line_num % 100000 == 0) {
            std::cout << "  Processed " << line_num / 1000 << "K records\n";
        }
    }
    
    file.close();
    
    std::cout << "Loaded " << data.size() << " records\n";
    std::cout << "Sorting...\n";
    
    std::sort(data.begin(), data.end());
    
    std::cout << "Reassigning positions...\n";
    for (size_t i = 0; i < data.size(); ++i) {
        data[i].second = i;
    }
    
    return data;
}

// Load OpenStreetMap longitude data from CSV
// Format: id,lon,lat (osmium export output)
std::vector<std::pair<double, size_t>> load_osm_longitudes(const std::string& filepath, size_t max_records = 0) {
    std::vector<std::pair<double, size_t>> data;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return data;
    }
    
    std::cout << "Loading OSM data from " << filepath << "...\n";
    
    std::string line;
    size_t line_num = 0;
    
    // Skip header (id,lon,lat)
    std::getline(file, line);
    
    while (std::getline(file, line) && (max_records == 0 || data.size() < max_records)) {
        line_num++;
        
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }
        
        // Longitude is in column 1 (format: id,lon,lat)
        if (tokens.size() >= 3) {
            try {
                double longitude = std::stod(tokens[1]);
                data.emplace_back(longitude, data.size());
            } catch (...) {
                continue;
            }
        }
        
        if (line_num % 1000000 == 0) {
            std::cout << "  Loaded " << line_num / 1000000 << "M records\n";
        }
    }
    
    file.close();
    
    std::cout << "Loaded " << data.size() << " records\n";
    std::cout << "Sorting...\n";
    
    std::sort(data.begin(), data.end());
    
    std::cout << "Reassigning positions...\n";
    for (size_t i = 0; i < data.size(); ++i) {
        data[i].second = i;
    }
    
    return data;
}

// Load from generic CSV
std::vector<std::pair<double, size_t>> load_csv_column(const std::string& filepath, 
                                                         size_t column_index,
                                                         bool has_header = true,
                                                         char delimiter = ',',
                                                         size_t max_records = 0) {
    std::vector<std::pair<double, size_t>> data;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return data;
    }
    
    std::cout << "Loading CSV data from " << filepath << " (column " << column_index << ")...\n";
    
    std::string line;
    size_t line_num = 0;
    
    if (has_header) {
        std::getline(file, line);
    }
    
    while (std::getline(file, line) && (max_records == 0 || data.size() < max_records)) {
        line_num++;
        
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (std::getline(ss, token, delimiter)) {
            tokens.push_back(token);
        }
        
        if (column_index < tokens.size()) {
            try {
                double value = std::stod(tokens[column_index]);
                data.emplace_back(value, data.size());
            } catch (...) {
                continue;
            }
        }
        
        if (line_num % 1000000 == 0) {
            std::cout << "  Loaded " << line_num / 1000000 << "M records\n";
        }
    }
    
    file.close();
    
    std::cout << "Loaded " << data.size() << " records\n";
    std::cout << "Sorting...\n";
    
    std::sort(data.begin(), data.end());
    
    std::cout << "Reassigning positions...\n";
    for (size_t i = 0; i < data.size(); ++i) {
        data[i].second = i;
    }
    
    return data;
}

// Generate lognormal distribution
std::vector<std::pair<double, size_t>> generate_lognormal(size_t n) {
    std::vector<std::pair<double, size_t>> data;
    data.reserve(n);
    
    std::mt19937 rng(42);
    std::lognormal_distribution<double> dist(0.0, 2.0);
    
    std::cout << "Generating " << n << " lognormal samples...\n";
    for (size_t i = 0; i < n; ++i) {
        double key = dist(rng) * 1e9;
        data.emplace_back(key, i);
        
        if ((i + 1) % 10000000 == 0) {
            std::cout << "  Generated " << (i + 1) / 1000000 << "M samples\n";
        }
    }
    
    std::cout << "Sorting...\n";
    std::sort(data.begin(), data.end());
    
    for (size_t i = 0; i < n; ++i) {
        data[i].second = i;
    }
    
    return data;
}

} // namespace dataset_loader

