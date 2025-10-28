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

// Simple 2-Stage Recursive Model Index (RMI) - FIXED VERSION
class SimpleRMI {
private:
    struct Stage1Model {
        double slope, intercept;
        int min_error, max_error;
        size_t start_idx, end_idx;
        uint64_t min_key, max_key;  // KEY RANGE this model covers
    };
    
    vector<Record> data_;
    
    // Stage 0: root model (key → stage1 model index)
    double stage0_slope_;
    double stage0_intercept_;
    
    // Stage 1: multiple leaf models
    vector<Stage1Model> stage1_models_;
    size_t num_stage1_models_;
    
public:
    SimpleRMI(size_t num_models = 100) : num_stage1_models_(num_models) {}
    
    void build(const vector<Record>& data) {
        data_ = data;
        stage1_models_.clear();
        
        if (data_.empty()) return;
        
        size_t n = data_.size();
        
        // ============================================
        // STAGE 0: Train root model (key → model_id)
        // ============================================
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
        
        for (size_t i = 0; i < n; i++) {
            double x = static_cast<double>(data_[i].key);
            double y = (static_cast<double>(i) / n) * num_stage1_models_;
            
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }
        
        stage0_slope_ = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        stage0_intercept_ = (sum_y - stage0_slope_ * sum_x) / n;
        
        // ============================================
        // STAGE 1: Train leaf models (key → position)
        // ============================================
        size_t records_per_model = (n + num_stage1_models_ - 1) / num_stage1_models_;
        
        for (size_t model_idx = 0; model_idx < num_stage1_models_; model_idx++) {
            Stage1Model model;
            model.start_idx = model_idx * records_per_model;
            model.end_idx = min(n, (model_idx + 1) * records_per_model);
            
            if (model.start_idx >= n) break;
            
            // Store key range for this model
            model.min_key = data_[model.start_idx].key;
            model.max_key = data_[model.end_idx - 1].key;
            
            size_t model_size = model.end_idx - model.start_idx;
            
            // Train linear model for this segment
            sum_x = sum_y = sum_xy = sum_xx = 0;
            
            for (size_t i = model.start_idx; i < model.end_idx; i++) {
                double x = static_cast<double>(data_[i].key);
                double y = static_cast<double>(i - model.start_idx);
                
                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_xx += x * x;
            }
            
            model.slope = (model_size * sum_xy - sum_x * sum_y) / 
                         (model_size * sum_xx - sum_x * sum_x);
            model.intercept = (sum_y - model.slope * sum_x) / model_size;
            
            // Calculate error bounds
            model.min_error = 0;
            model.max_error = 0;
            
            for (size_t i = model.start_idx; i < model.end_idx; i++) {
                int pred = static_cast<int>(model.slope * data_[i].key + model.intercept);
                pred = max(0, min((int)model_size - 1, pred));
                int actual = static_cast<int>(i - model.start_idx);
                int error = actual - pred;
                
                model.min_error = min(model.min_error, error);
                model.max_error = max(model.max_error, error);
            }
            
            stage1_models_.push_back(model);
        }
    }
    
    bool lookup(uint64_t key, uint64_t& value) {
        if (data_.empty() || stage1_models_.empty()) return false;
        
        // FIXED: Binary search to find the correct model by key range
        size_t left = 0, right = stage1_models_.size();
        while (left < right) {
            size_t mid = left + (right - left) / 2;
            if (stage1_models_[mid].max_key < key) {
                left = mid + 1;
            } else if (stage1_models_[mid].min_key > key) {
                right = mid;
            } else {
                // Found the correct model!
                left = mid;
                break;
            }
        }
        
        if (left >= stage1_models_.size()) return false;
        
        // Verify key is in this model's range
        const Stage1Model& model = stage1_models_[left];
        if (key < model.min_key || key > model.max_key) {
            return false;
        }
        
        // Stage 1: Predict position within model's range
        int local_pred = static_cast<int>(model.slope * key + model.intercept);
        local_pred = max(0, min((int)(model.end_idx - model.start_idx) - 1, local_pred));
        
        // Calculate search range with error bounds
        int start = max((int)model.start_idx, 
                       (int)model.start_idx + local_pred + model.min_error);
        int end = min((int)model.end_idx - 1, 
                     (int)model.start_idx + local_pred + model.max_error);
        
        // Binary search in predicted range
        auto it = lower_bound(data_.begin() + start, 
                             data_.begin() + end + 1,
                             Record(key, 0));
        
        if (it != data_.begin() + end + 1 && it->key == key) {
            value = it->value;
            return true;
        }
        return false;
    }
    
    size_t getMemoryUsage() const {
        size_t data_size = sizeof(Record) * data_.size();
        size_t stage0_size = sizeof(stage0_slope_) + sizeof(stage0_intercept_);
        size_t stage1_size = sizeof(Stage1Model) * stage1_models_.size();
        return data_size + stage0_size + stage1_size;
    }
    
    string getName() const {
        return "RMI (" + to_string(num_stage1_models_) + " models)";
    }
    
    void printModelInfo() const {
        cout << "  Stage 0: Routes to " << stage1_models_.size() << " models" << endl;
        cout << "  Stage 1: " << stage1_models_.size() << " models" << endl;
        
        // Calculate average error range
        int total_range = 0;
        for (const auto& model : stage1_models_) {
            total_range += (model.max_error - model.min_error);
        }
        cout << "  Avg search range: " << (total_range / stage1_models_.size()) 
             << " positions" << endl;
        
        // Show min/max ranges
        int min_range = INT_MAX, max_range = 0;
        for (const auto& model : stage1_models_) {
            int range = model.max_error - model.min_error;
            min_range = min(min_range, range);
            max_range = max(max_range, range);
        }
        cout << "  Range: [" << min_range << ", " << max_range << "] positions" << endl;
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
    
    // Print model info for RMI
    if constexpr (is_same_v<IndexType, SimpleRMI>) {
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
        {"Medium", 1000000},
        {"Large", 10000000}
    };

    int selected_config = 3;  // 0=Tiny, 1=Small, 2=Medium, 3=Large
    string config_name = configs[selected_config].first;
    size_t num_records = configs[selected_config].second;

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
    
    // Test 1: Sequential data
    SimpleBPlusTree btree_seq(128);
    SimpleLearnedIndex learned_seq;
    SimpleRMI rmi_seq(10000);  // 10000 second-stage models
    
    benchmarkIndex(btree_seq, seq_data, seq_queries, "Sequential Data");
    benchmarkIndex(learned_seq, seq_data, seq_queries, "Sequential Data");
    benchmarkIndex(rmi_seq, seq_data, seq_queries, "Sequential Data");

    // Test 2: LogNormal data
    SimpleBPlusTree btree_lognormal(128);
    SimpleLearnedIndex learned_lognormal;
    SimpleRMI rmi_lognormal(10000);
    
    benchmarkIndex(btree_lognormal, lognormal_data, log_queries, "LogNormal Data");
    benchmarkIndex(learned_lognormal, lognormal_data, log_queries, "LogNormal Data");
    benchmarkIndex(rmi_lognormal, lognormal_data, log_queries, "LogNormal Data");

    //cout << "\n" << string(60, '=') << endl;
    //cout << "SUMMARY" << endl;
    //cout << string(60, '=') << endl;
    //cout << "\nKey Findings:" << endl;
    //cout << "1. Single learned index fails on skewed distributions" << endl;
    //cout << "2. RMI adapts to complex distributions using hierarchy" << endl;
    //cout << "3. Sequential data: All learned approaches excel" << endl;
    //cout << "4. More models = better fit for complex distributions" << endl;
    //cout << "\nConclusion: Recursive models handle diverse data patterns!" << endl;
    //cout << "\n";
    
    return 0;
}
