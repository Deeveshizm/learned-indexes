#include "learned_index.hpp"
#include "btree.hpp"
#include "dataset_loader.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <random>
#include <map>
#include <string>

struct BenchmarkResult {
    std::string name;
    double build_time_ms;
    double avg_lookup_ns;
    double size_mb;
    double avg_error;
};

// Benchmark B-Tree
template<size_t PAGE_SIZE>
BenchmarkResult benchmark_btree(std::vector<std::pair<double, size_t>>& data,
                                const std::vector<double>& queries,
                                const std::string& name) {
    BTree<double, size_t, PAGE_SIZE> btree;
    
    auto start = std::chrono::high_resolution_clock::now();
    btree.build(data);
    auto end = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    for (const auto& q : queries) {
        volatile size_t pos = btree.lower_bound(q);
        (void)pos;
    }
    end = std::chrono::high_resolution_clock::now();
    auto lookup_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    return {name,
            static_cast<double>(build_time),
            static_cast<double>(lookup_time) / queries.size(),
            btree.get_size_bytes() / (1024.0 * 1024.0),
            0.0};
}

// Benchmark Learned Index
BenchmarkResult benchmark_learned(std::vector<std::pair<double, size_t>>& data,
                                   const std::vector<double>& queries,
                                   const RecursiveModelIndex::Config& cfg,
                                   const std::string& name) {
    RecursiveModelIndex index(cfg);
    
    auto start = std::chrono::high_resolution_clock::now();
    index.build(data);
    auto end = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Measure lookup time and error
    start = std::chrono::high_resolution_clock::now();
    double total_error = 0.0;
    for (const auto& q : queries) {
        size_t pred = index.lookup(q);
        (void)pred;
        
        // Calculate error
        auto it = std::lower_bound(data.begin(), data.end(), 
                                   std::make_pair(q, size_t(0)));
        if (it != data.end()) {
            size_t true_pos = std::distance(data.begin(), it);
            total_error += std::abs(static_cast<long>(pred) - static_cast<long>(true_pos));
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto lookup_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    return {name,
            static_cast<double>(build_time),
            static_cast<double>(lookup_time) / queries.size(),
            index.get_total_size() / (1024.0 * 1024.0),
            total_error / queries.size()};
}

void print_results(const std::string& dataset_name, 
                   size_t dataset_size,
                   const std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "DATASET: " << dataset_name << " (" << dataset_size << " records)\n";
    std::cout << std::string(100, '=') << "\n";
    std::cout << std::left << std::setw(35) << "Configuration"
              << std::right << std::setw(15) << "Build (ms)"
              << std::setw(15) << "Lookup (ns)"
              << std::setw(15) << "Size (MB)"
              << std::setw(15) << "Avg Error" << "\n";
    std::cout << std::string(100, '-') << "\n";
    
    for (const auto& r : results) {
        std::cout << std::left << std::setw(35) << r.name
                  << std::right << std::setw(15) << std::fixed << std::setprecision(2) << r.build_time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << r.avg_lookup_ns
                  << std::setw(15) << std::fixed << std::setprecision(2) << r.size_mb
                  << std::setw(15) << std::fixed << std::setprecision(1) << r.avg_error << "\n";
    }
    std::cout << std::string(100, '=') << "\n";
}

void save_results_json(const std::map<std::string, std::vector<BenchmarkResult>>& all_results,
                       const std::string& filename = "benchmark_results.json") {
    std::ofstream file(filename);
    file << "{\n";
    
    bool first_dataset = true;
    for (const auto& [dataset_name, results] : all_results) {
        if (!first_dataset) file << ",\n";
        first_dataset = false;
        
        file << "  \"" << dataset_name << "\": [\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            file << "    {\n";
            file << "      \"name\": \"" << r.name << "\",\n";
            file << "      \"build_time_ms\": " << r.build_time_ms << ",\n";
            file << "      \"avg_lookup_ns\": " << r.avg_lookup_ns << ",\n";
            file << "      \"size_mb\": " << r.size_mb << ",\n";
            file << "      \"avg_error\": " << r.avg_error << "\n";
            file << "    }" << (i < results.size() - 1 ? "," : "") << "\n";
        }
        file << "  ]";
    }
    
    file << "\n}\n";
    file.close();
    std::cout << "\n✓ Results saved to " << filename << "\n";
}

int main() {
    std::cout << "Learned Index vs B-Tree Benchmark (Quick Test)\n";
    std::cout << "================================================\n\n";
    
    const size_t NUM_QUERIES = 10000;
    std::mt19937 rng(42);
    
    std::map<std::string, std::vector<BenchmarkResult>> all_results;
    
    // Test 1: Lognormal (1M)
    {
        std::cout << "\n>>> Loading Lognormal (1M)...\n";
        auto data = dataset_loader::generate_lognormal(1000000);
        
        std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
        std::vector<double> queries;
        for (size_t i = 0; i < NUM_QUERIES; ++i) {
            queries.push_back(data[dist(rng)].first);
        }
        
        std::vector<BenchmarkResult> results;
        
        std::cout << "Testing B-Trees...\n";
        auto copy = data;
        results.push_back(benchmark_btree<128>(copy, queries, "B-Tree (page=128)"));
        
        copy = data;
        results.push_back(benchmark_btree<256>(copy, queries, "B-Tree (page=256)"));
        
        std::cout << "Testing Learned Indexes...\n";
        copy = data;
        results.push_back(benchmark_learned(copy, queries, 
            {{1, 1000}, 8, 0, 128, false}, "Learned (1K, linear)"));
        
        copy = data;
        results.push_back(benchmark_learned(copy, queries,
            {{1, 1000}, 8, 1, 128, false}, "Learned (1K, 1-layer)"));
        
        copy = data;
        results.push_back(benchmark_learned(copy, queries,
            {{1, 10000}, 8, 0, 128, false}, "Learned (10K, linear)"));
        
        print_results("Lognormal", data.size(), results);
        all_results["Lognormal (1M)"] = results;
    }
    
    // Test 2: NASA Logs
    {
        std::cout << "\n>>> Loading NASA Logs...\n";
        auto data = dataset_loader::load_nasa_logs("data/NASA_access_log_Jul95");
        
        if (data.size() > 0) {
            std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
            std::vector<double> queries;
            for (size_t i = 0; i < NUM_QUERIES; ++i) {
                queries.push_back(data[dist(rng)].first);
            }
            
            std::vector<BenchmarkResult> results;
            
            std::cout << "Testing B-Trees...\n";
            auto copy = data;
            results.push_back(benchmark_btree<128>(copy, queries, "B-Tree (page=128)"));
            
            copy = data;
            results.push_back(benchmark_btree<256>(copy, queries, "B-Tree (page=256)"));
            
            std::cout << "Testing Learned Indexes...\n";
            copy = data;
            results.push_back(benchmark_learned(copy, queries, 
                {{1, 1000}, 8, 0, 128, false}, "Learned (1K, linear)"));
            
            copy = data;
            results.push_back(benchmark_learned(copy, queries,
                {{1, 1000}, 8, 1, 128, false}, "Learned (1K, 1-layer)"));
            
            copy = data;
            results.push_back(benchmark_learned(copy, queries,
                {{1, 10000}, 8, 0, 128, false}, "Learned (10K, linear)"));
            
            print_results("NASA Web Logs", data.size(), results);
            all_results["NASA Web Logs (1.9M)"] = results;
        }
    }
    
    // Test 3: Florida OSM (1M subset)
    {
        std::cout << "\n>>> Loading Florida OSM (1M subset)...\n";
        auto data = dataset_loader::load_osm_longitudes("data/florida_nodes.csv", 1000000);
        
        if (data.size() > 0) {
            std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
            std::vector<double> queries;
            for (size_t i = 0; i < NUM_QUERIES; ++i) {
                queries.push_back(data[dist(rng)].first);
            }
            
            std::vector<BenchmarkResult> results;
            
            std::cout << "Testing B-Trees...\n";
            auto copy = data;
            results.push_back(benchmark_btree<128>(copy, queries, "B-Tree (page=128)"));
            
            copy = data;
            results.push_back(benchmark_btree<256>(copy, queries, "B-Tree (page=256)"));
            
            std::cout << "Testing Learned Indexes...\n";
            copy = data;
            results.push_back(benchmark_learned(copy, queries, 
                {{1, 1000}, 8, 0, 128, false}, "Learned (1K, linear)"));
            
            copy = data;
            results.push_back(benchmark_learned(copy, queries,
                {{1, 1000}, 8, 1, 128, false}, "Learned (1K, 1-layer)"));
            
            copy = data;
            results.push_back(benchmark_learned(copy, queries,
                {{1, 10000}, 8, 0, 128, false}, "Learned (10K, linear)"));
            
            print_results("Florida OSM (subset)", data.size(), results);
            all_results["Florida OSM (1M)"] = results;
        }
    }
    
    // Save results to JSON
    save_results_json(all_results);
    
    std::cout << "\n✓ Benchmark complete!\n";
    std::cout << "\nTo generate plots, run:\n";
    std::cout << "  python3 plot_results.py\n";
    std::cout << "  (or: make plot)\n\n";
    
    return 0;
}
