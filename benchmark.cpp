// Your original working benchmark with error tracking added
// Same structure, just added error metrics

#include "learned_index.hpp"
#include "btree.hpp"
#include "dataset_loader.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <map>
#include <algorithm>
#include <fstream>
#include <cmath>

struct BenchmarkResult {
    std::string name;
    double build_time_ms;
    double avg_lookup_ns;
    double size_mb;
    double error_percentage;     // Error as percentage of dataset size
};

// Benchmark B-Tree
template<size_t PAGE_SIZE>
BenchmarkResult benchmark_btree(std::vector<std::pair<double, size_t>>& data,
                                const std::vector<double>& queries,
                                const std::string& name) {
    BTree<double, size_t, PAGE_SIZE> btree;
    
    std::cout << "  Building " << name << "...\n";
    auto start = std::chrono::high_resolution_clock::now();
    btree.build(data);
    auto end = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "  Benchmarking lookups...\n";
    start = std::chrono::high_resolution_clock::now();
    for (const auto& q : queries) {
        btree.lookup(q);
    }
    end = std::chrono::high_resolution_clock::now();
    auto lookup_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    return {name,
            static_cast<double>(build_time),
            static_cast<double>(lookup_time) / queries.size(),
            btree.get_size_bytes() / (1024.0 * 1024.0),
            0.0};  // B-Tree has perfect accuracy (0% error)
}

// Benchmark Learned Index
BenchmarkResult benchmark_learned(std::vector<std::pair<double, size_t>>& data,
                                   const std::vector<double>& queries,
                                   const RecursiveModelIndex::Config& cfg,
                                   const std::string& name) {
    RecursiveModelIndex index(cfg);
    
    std::cout << "  Building " << name << "...\n";
    auto start = std::chrono::high_resolution_clock::now();
    index.build(data);
    auto end = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "  Benchmarking lookups...\n";
    
    start = std::chrono::high_resolution_clock::now();
    double total_error = 0.0;
    
    for (const auto& q : queries) {
        size_t pred = index.lookup(q);
        
        // Calculate error
        auto it = std::lower_bound(data.begin(), data.end(), 
                                   std::make_pair(q, size_t(0)));
        if (it != data.end()) {
            size_t true_pos = std::distance(data.begin(), it);
            double error = std::abs(static_cast<long>(pred) - static_cast<long>(true_pos));
            total_error += error;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto lookup_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    // Calculate error as percentage of dataset size
    double avg_error = total_error / queries.size();
    double error_percentage = (avg_error / data.size()) * 100.0;
    
    return {name,
            static_cast<double>(build_time),
            static_cast<double>(lookup_time) / queries.size(),
            index.get_total_size() / (1024.0 * 1024.0),
            error_percentage};
}

void print_results(const std::string& dataset_name, 
                   size_t dataset_size,
                   size_t num_queries,
                   const std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(105, '=') << "\n";
    std::cout << "DATASET: " << dataset_name << " (" << dataset_size << " records, " 
              << num_queries << " queries)\n";
    std::cout << std::string(105, '=') << "\n";
    std::cout << std::left << std::setw(45) << "Configuration"
              << std::right << std::setw(15) << "Build (ms)"
              << std::setw(15) << "Lookup (ns)"
              << std::setw(15) << "Error %"
              << std::setw(15) << "Speedup" << "\n";
    std::cout << std::string(105, '-') << "\n";
    
    // Find best B-Tree for speedup calculation
    double best_btree = 1e9;
    for (const auto& r : results) {
        if (r.name.find("B-Tree") != std::string::npos) {
            best_btree = std::min(best_btree, r.avg_lookup_ns);
        }
    }
    
    for (const auto& r : results) {
        double speedup = best_btree / r.avg_lookup_ns;
        
        std::cout << std::left << std::setw(45) << r.name
                  << std::right << std::setw(15) << std::fixed << std::setprecision(0) << r.build_time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << r.avg_lookup_ns
                  << std::setw(15) << std::fixed << std::setprecision(4) << r.error_percentage << "%"
                  << std::setw(15) << std::fixed << std::setprecision(1) << speedup << "×" << "\n";
    }
    std::cout << std::string(105, '=') << "\n";
}

void save_to_json(const std::string& filename, 
                  const std::map<std::string, std::vector<BenchmarkResult>>& all_results) {
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
            file << "      \"error_percentage\": " << r.error_percentage << "\n";
            file << "    }" << (i < results.size() - 1 ? "," : "") << "\n";
        }
        file << "  ]";
    }
    file << "\n}\n";
    file.close();
}

int main() {
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║    COMPLETE LEARNED INDEX BENCHMARK - ALL APPROACHES          ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Testing Three Approaches:\n";
    std::cout << "  1. B-Tree (Baseline)\n";
    std::cout << "  2. Pure Linear Models\n";
    std::cout << "  3. HYBRID (NN top + Linear bottom) - Paper's recommendation\n\n";
    
    // CHANGED: Reduced for faster testing
    std::cout << "Dataset sizes:\n";
    std::cout << "  • Lognormal: 10M records\n";
    std::cout << "  • NASA: Full dataset\n";
    std::cout << "  • Florida OSM: 10M records\n\n";
    
    const size_t NUM_QUERIES = 10000;
    const size_t DATASET_SIZE = 1000000;
    std::mt19937 rng(42);
    
    std::map<std::string, std::vector<BenchmarkResult>> all_results;
    
    // Test on all three datasets
    std::vector<std::pair<std::string, std::vector<std::pair<double, size_t>>>> datasets;
    
    // Dataset 1: Lognormal (10M) - CHANGED
    std::cout << ">>> Loading Lognormal (1M)...\n";
    datasets.push_back({"Lognormal (1M)", dataset_loader::generate_lognormal(DATASET_SIZE)});
    
    // Dataset 2: NASA Logs (full)
    std::cout << ">>> Loading NASA Logs...\n";
    auto nasa_data = dataset_loader::load_nasa_logs("data/NASA_access_log_Jul95");
    if (nasa_data.size() > 0) {
        datasets.push_back({"NASA Web Logs", nasa_data});
    }
    
    // Dataset 3: Florida OSM (10M) - CHANGED
    std::cout << ">>> Loading Florida OSM (1M)...\n";
    auto osm_data = dataset_loader::load_osm_longitudes("data/florida_nodes.csv", DATASET_SIZE);
    if (osm_data.size() > 0) {
        datasets.push_back({"Florida OSM (1M)", osm_data});
    }
    
    // Run benchmarks on each dataset
    for (auto& [dataset_name, data] : datasets) {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "BENCHMARKING: " << dataset_name << "\n";
        std::cout << std::string(80, '=') << "\n";
        
        // Generate queries
        std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
        std::vector<double> queries;
        queries.reserve(NUM_QUERIES);
        for (size_t i = 0; i < NUM_QUERIES; ++i) {
            queries.push_back(data[dist(rng)].first);
        }
        
        std::vector<BenchmarkResult> results;
        
        std::cout << "\n[1/4] B-Tree Baselines:\n";
        std::cout << std::string(50, '-') << "\n";
        results.push_back(benchmark_btree<128>(data, queries, "B-Tree (page=128)"));
        results.push_back(benchmark_btree<256>(data, queries, "B-Tree (page=256)"));
        
        std::cout << "\n[2/3] Pure Linear Models:\n";
        std::cout << std::string(50, '-') << "\n";
        
        RecursiveModelIndex::Config cfg_linear_1;
        cfg_linear_1.stage_sizes = {1};
        cfg_linear_1.num_hidden_layers = 0;
        results.push_back(benchmark_learned(data, queries, cfg_linear_1, "Linear [1]"));
        
        RecursiveModelIndex::Config cfg_linear_1k;
        cfg_linear_1k.stage_sizes = {1, 1000};
        cfg_linear_1k.num_hidden_layers = 0;
        results.push_back(benchmark_learned(data, queries, cfg_linear_1k, "Linear [1,1K]"));
        
        RecursiveModelIndex::Config cfg_linear_10k;
        cfg_linear_10k.stage_sizes = {1, 10000};
        cfg_linear_10k.num_hidden_layers = 0;
        results.push_back(benchmark_learned(data, queries, cfg_linear_10k, "Linear [1,10K]"));
        
        std::cout << "\n[3/3] HYBRID Approach (NN top + Linear bottom):\n";
        std::cout << std::string(50, '-') << "\n";
        
        RecursiveModelIndex::Config cfg_hybrid_1k;
        cfg_hybrid_1k.stage_sizes = {1, 1000};
        cfg_hybrid_1k.num_hidden_layers = 1;
        cfg_hybrid_1k.hidden_size = 8;
        results.push_back(benchmark_learned(data, queries, cfg_hybrid_1k, "HYBRID: 1-layer NN + 1K Linear"));
        
        RecursiveModelIndex::Config cfg_hybrid_10k;
        cfg_hybrid_10k.stage_sizes = {1, 10000};
        cfg_hybrid_10k.num_hidden_layers = 1;
        cfg_hybrid_10k.hidden_size = 8;
        results.push_back(benchmark_learned(data, queries, cfg_hybrid_10k, "HYBRID: 1-layer NN + 10K Linear"));
        
        RecursiveModelIndex::Config cfg_hybrid_2layer_10k;
        cfg_hybrid_2layer_10k.stage_sizes = {1, 10000};
        cfg_hybrid_2layer_10k.num_hidden_layers = 2;
        cfg_hybrid_2layer_10k.hidden_size = 16;
        results.push_back(benchmark_learned(data, queries, cfg_hybrid_2layer_10k, "HYBRID: 2-layer NN + 10K Linear"));
        
        print_results(dataset_name, data.size(), NUM_QUERIES, results);
        all_results[dataset_name] = results;
    }
    
    save_to_json("benchmark_results_complete.json", all_results);
    std::cout << "\n✓ Results saved to benchmark_results_complete.json\n";
    
    return 0;
}
