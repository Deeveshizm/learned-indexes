#pragma once
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

// base model interface
class Model {
public:
    virtual ~Model() = default;
    virtual double predict(double key) const = 0;
    virtual void train(const std::vector<std::pair<double, size_t>>& data) = 0;
    virtual size_t get_model_size() const = 0;
};

// Linear Model
class LinearModel : public Model {
private:
    double slope;
    double intercept;

public:
    double predict(double key) const override {
        return slope * key + intercept;
    }
    void train(const std::vector<std::pair<double, size_t>>& data) override;
    
    size_t get_model_size() const override {
        return sizeof(slope) + sizeof(intercept);
    }
};

// Neural Network with configurable layers
class NeuralNetModel : public Model {
private:
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> biases;
    size_t hidden_size;
    size_t num_layers;

    double relu(double x) const { return std::max(0.0, x); }

    double x_min_, x_max_, x_range_, y_max_;
    bool use_log_;  // Flag for log transformation

public:
    NeuralNetModel(size_t hidden_size, size_t num_layers);
    double predict(double key) const override;
    void train(const std::vector<std::pair<double, size_t>>& data) override;
    size_t get_model_size() const override;
};


// Stage in RMI
struct Stage {
    std::vector<std::unique_ptr<Model>> models;
    std::vector<double> min_errors;
    std::vector<double> max_errors;
};

// main RMI architecture
class RecursiveModelIndex {
private:
    std::vector<Stage> stages;
    std::vector<double> sorted_keys;
    std::vector<size_t> sorted_positions;
    size_t total_records;

    
public:
    // Configuration
    struct Config {
        std::vector<size_t> stage_sizes;        // number of models per stage
        size_t hidden_size = 8;
        size_t num_hidden_layers = 1;
        double error_threshold = 128;           // for hybrid indexes
        bool use_hybrid = false;
    };
    Config config;

    // Constructor
    RecursiveModelIndex(const Config& cfg) :config(cfg) {}

    // Build the index
    void build(std::vector<std::pair<double, size_t>>& data);

    // lookup operations
    size_t lookup(double key) const;
    size_t lower_bound(double key) const;
    size_t upper_bound(double key) const;

    //statistics
    size_t get_total_size() const;
    double get_average_error() const;
    void print_statistics() const;
};
