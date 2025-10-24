#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <unordered_set>
#include <type_traits>

using namespace std;

// structure to hold a record
struct Record {
    uint64_t key;
    uint64_t value;

    Record(uint64_t k = 0, uint64_t v = 0) : key(k), value(v) {}
    bool operator<(const Record &other) const {
        return key < other.key;
    }
};

// generate synthetic data with a distribution
class DataGenerator {
public:
    static vector<Record> genSequentual(size_t n, uint64_t start = 0) {
        vector<Record> data;
        for (size_t i = 0; i < n; ++i) {
            data.emplace_back(start + i, start + i);
        }
        return data;
    }
        
    static vector<Record> genLogNormal(size_t n) {
        vector<Record> data;
        random_device rd;
        mt19937 gen(rd());
        lognormal_distribution<double> dist(0.0, 2.0);
    
        unordered_set<uint64_t> seen;
        while (data.size() < n) {
            uint64_t key = static_cast<uint64_t>(dist(gen) * 1e7);
            if (seen.insert(key).second) {
                data.emplace_back(key, key);
            }
        }
            
        sort(data.begin(), data.end());
        return data;
    }
        
    static vector<Record> genUniform(size_t n, uint64_t max) {
        vector<Record> data;
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<uint64_t> dist(0, max);
    
        unordered_set<uint64_t> seen;
        while (data.size() < n) {
            uint64_t key = dist(gen);
            if (seen.insert(key).second) {
                data.emplace_back(key, key);
            }
        }
            
        sort(data.begin(), data.end());
        return data;
    }
};

// simple b+ tree (page based index)
class SimpleBPlusTree {
private:
    vector<Record> data_;
    vector<pair<uint64_t, size_t>> index_;
    size_t page_size_;
public:
    SimpleBPlusTree(size_t page_size = 128) : page_size_(page_size) {}

    void build (const vector<Record> &data) {
        data_ = data;
        index_.clear();

        // create index entry
        for (size_t i = 0; i < data_.size(); i += page_size_) {
            index_.emplace_back(data_[i].key, i);
        }
    }

    bool lookup(uint64_t key, uint64_t &value) {
        if (data_.empty()) return false;

        // binary search in index
        size_t left  = 0, right = index_.size();
        while (left < right) {
            size_t mid = left + (right - left) / 2;
            if (index_[mid].first <= key) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        size_t page_idx = (left > 0) ? left - 1 : 0;
        size_t start = index_[page_idx].second;
        size_t end = (page_idx + 1 < index_.size()) ? index_[page_idx + 1].second : data_.size();

        // binary search in page
        auto it  = lower_bound(
                    data_.begin() + start,
                    data_.begin() + end,
                    Record(key, 0)
                   );
        if (it != data_.begin() + end && it->key == key) {
            value = it->value;
            return true;
        }
        return false;
    }

    size_t getMemoryUsage() const {
        return sizeof(Record) * data_.size() + sizeof(pair<uint64_t, size_t>) * index_.size();
    }

    string getName() const {
        return "B+ Tree";
    }
    
};

// simple leaned Index
class SimpleLearnedIndex {
private:
    vector<Record> data_;
    double slope_;
    double intercept_;
    int min_error_;
    int max_error_;

public:
    void build (const vector<Record> &data) {
        data_ = data;


        // train linear regression model on cdf
        // position = slope * key + intercept
        size_t n = data_.size();

        // calculate slope and intercepts uding least squares
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;

        for (size_t i = 0; i < n; i++) {
            double x = static_cast<double>(data_[i].key);
            double y = static_cast<double>(i); // position

            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        slope_ = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        intercept_ = (sum_y - slope_ * sum_x) / n;

        // calculate error bounds
        min_error_ = 0;
        max_error_ = 0;

        for (size_t i = 0; i < n; i++) {
            int predicted = predict(data_[i].key);
            int error = static_cast<int>(i) - predicted;
            min_error_ = min(min_error_, error);
            max_error_ = max(max_error_, error);
        }

    }

    int predict(uint64_t key) const {
        double pred = slope_ * key + intercept_;
        return max(0, min(static_cast<int>(data_.size() - 1), static_cast<int>(pred)));
    }

    bool lookup(uint64_t key, uint64_t &value) {
        if (data_.empty()) return false;

        int predicted_pos = predict(key);
        int start = max(0, predicted_pos + min_error_);
        int end = min(static_cast<int>(data_.size() - 1), predicted_pos + max_error_);

        // binary search in the error bound range
        auto it = lower_bound(
                    data_.begin() + start,
                    data_.begin() + end + 1,
                    Record(key, 0)
                  );
        if (it != data_.begin() + end + 1 && it->key == key) {
            value = it->value;
            return true;
        }
        return false;
    }

    size_t getMemoryUsage() const {
        return sizeof(Record) * data_.size() + sizeof(slope_) + sizeof(intercept_) + sizeof(min_error_) + sizeof(max_error_);
    }
    string getName() const {
        return "Learned Index";
    }

    void printModelInfo() const {
        cout << "  Model: y = " << slope_ << " * x + " << intercept_ << endl;
        cout << "  Error bounds: [" << min_error_ << ", " << max_error_ << "]" << endl;
        cout << "  Search range: " << (max_error_ - min_error_) << " positions" << endl;
    }
};

// benchmark
class Timer {
    chrono::high_resolution_clock::time_point start_;
public:
    Timer() : start_(chrono::high_resolution_clock::now()) {}
    
    void reset() { start_ = chrono::high_resolution_clock::now(); }
    
    long long elapsed_ns() const {
        auto end = chrono::high_resolution_clock::now();
        return chrono::duration_cast<chrono::nanoseconds>(end - start_).count();
    }
};

template<typename IndexType>
void benchmarkIndex(IndexType& index, const vector<Record>& data, 
                   const vector<uint64_t>& query_keys, const string& dataset_name) {
    
    cout << "\n" << string(60, '=') << endl;
    cout << index.getName() << " on " << dataset_name << endl;
    cout << string(60, '=') << endl;
    
    // Build
    Timer timer;
    index.build(data);
    cout << "Build time: " << (timer.elapsed_ns() / 1e6) << " ms" << endl;
    
    // Memory
    double memory_mb = index.getMemoryUsage() / (1024.0 * 1024.0);
    cout << "Memory usage: " << fixed << setprecision(2) << memory_mb << " MB" << endl;
    
    // Print model info for learned index
    if constexpr (is_same_v<IndexType, SimpleLearnedIndex>) {
        index.printModelInfo();
    }
    
    // Lookup benchmark
    const size_t warmup = 1000;
    const size_t iterations = 10000;
    
    // Warmup
    for (size_t i = 0; i < warmup; i++) {
        uint64_t val;
        index.lookup(query_keys[i % query_keys.size()], val);
    }
    
    // Measure
    timer.reset();
    size_t found = 0;
    for (size_t i = 0; i < iterations; i++) {
        uint64_t val;
        if (index.lookup(query_keys[i % query_keys.size()], val)) {
            found++;
        }
    }
    long long total_ns = timer.elapsed_ns();
    double avg_ns = static_cast<double>(total_ns) / iterations;
    
    cout << "Lookup time: " << fixed << setprecision(1) << avg_ns << " ns/op" << endl;
    cout << "Throughput: " << fixed << setprecision(2) 
         << (1e9 / avg_ns / 1e6) << " M ops/sec" << endl;
    cout << "Found: " << found << "/" << iterations << endl;
}


// Main
int main() {
    // test configurations
    vector<pair<string, size_t>> configs = {
        {"Tiny", 10000},
        {"Small", 100000},
        {"Medium", 1000000}
    };

    string config_name = configs[1].first;  // taking 100K for demo
    size_t num_records = configs[1].second;

    cout << "Configuration: " << config_name << " (" << num_records << " records)" << endl;

    // Generate Datasets
    auto seq_data = DataGenerator::genSequentual(num_records);
    auto lognormal_data = DataGenerator::genLogNormal(num_records);

    cout << "✓ Sequential dataset: " << seq_data.size() << " records" << endl;
    cout << "✓ LogNormal dataset: " << lognormal_data.size() << " records" << endl;

    // Generate query keys
    random_device rd;
    mt19937_64 gen(rd());
    
    auto generateQueries = [&](const vector<Record>& data) {
        vector<uint64_t> queries;
        uniform_int_distribution<size_t> dist(0, data.size() - 1);
        for (int i = 0; i < 10000; i++) {
            queries.push_back(data[dist(gen)].key);
        }
        return queries;
    };
    
    auto seq_queries = generateQueries(seq_data);
    auto log_queries = generateQueries(lognormal_data);
    
    // Test 1 Sequential data
    SimpleBPlusTree btree_seq(128);
    SimpleLearnedIndex learned_seq;
    
    benchmarkIndex(btree_seq, seq_data, seq_queries, "Sequential Data");
    benchmarkIndex(learned_seq, seq_data, seq_queries, "Sequential Data");

    // Test 2 LogNormal data
    SimpleBPlusTree btree_lognormal(128);
    SimpleLearnedIndex learned_lognormal;
    benchmarkIndex(btree_lognormal, lognormal_data, log_queries, "LogNormal Data");
    benchmarkIndex(learned_lognormal, lognormal_data, log_queries, "LogNormal Data");

    //cout << "\n" << string(60, '=') << endl;
    //cout << "SUMMARY" << endl;
    //cout << string(60, '=') << endl;
    //cout << "\nKey Findings:" << endl;
    //cout << "1. Learned index uses ~same memory as B-tree" << endl;
    //cout << "2. Sequential data: Learned index significantly faster" << endl;
    //cout << "3. LogNormal data: Learned index competitive" << endl;
    //cout << "4. Model learns the data distribution (CDF)" << endl;
    //cout << "\nConclusion: ML models can replace traditional index structures!" << endl;
    //cout << "\n";
    
    return 0;
    
}
