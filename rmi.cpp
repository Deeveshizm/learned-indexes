#include "learned_index.hpp"
#include <iostream>
#include <limits>

void RecursiveModelIndex::build(std::vector<std::pair<double, size_t>>& data) {
    if (data.empty()) return;
    
    std::sort(data.begin(), data.end());
    total_records = data.size();
    
    sorted_keys.reserve(total_records);
    sorted_positions.reserve(total_records);
    for (size_t i = 0; i < total_records; ++i) {
        sorted_keys.push_back(data[i].first);
        sorted_positions.push_back(i);
    }
    
    stages.resize(config.stage_sizes.size());
    
    // Allocate stage data structures
    std::vector<std::vector<std::vector<std::pair<double, size_t>>>> stage_data(
        config.stage_sizes.size()
    );
    
    // Initialize first stage with all data
    stage_data[0].resize(config.stage_sizes[0]);
    stage_data[0][0] = data;
    
    // Build each stage
    for (size_t stage_idx = 0; stage_idx < config.stage_sizes.size(); ++stage_idx) {
        size_t num_models = config.stage_sizes[stage_idx];
        stages[stage_idx].models.resize(num_models);
        stages[stage_idx].min_errors.resize(num_models, 0);
        stages[stage_idx].max_errors.resize(num_models, 0);
        
        // Prepare next stage data structure
        if (stage_idx + 1 < config.stage_sizes.size()) {
            stage_data[stage_idx + 1].resize(config.stage_sizes[stage_idx + 1]);
        }
        
        // Train each model in this stage
        for (size_t model_idx = 0; model_idx < num_models; ++model_idx) {
            auto& model_data = stage_data[stage_idx][model_idx];
            
            if (model_data.empty()) {
                // Create a dummy linear model for empty data
                stages[stage_idx].models[model_idx] = std::make_unique<LinearModel>();
                continue;
            }
            
            // Create appropriate model
            if (stage_idx == 0 && config.num_hidden_layers > 0) {
                stages[stage_idx].models[model_idx] = 
                    std::make_unique<NeuralNetModel>(
                        config.hidden_size, 
                        config.num_hidden_layers
                    );
            } else {
                stages[stage_idx].models[model_idx] = 
                    std::make_unique<LinearModel>();
            }
            
            // Train the model
            stages[stage_idx].models[model_idx]->train(model_data);
            
            // Route data to next stage if not final stage
            if (stage_idx + 1 < config.stage_sizes.size()) {
                size_t next_stage_size = config.stage_sizes[stage_idx + 1];
                double min_err = std::numeric_limits<double>::max();
                double max_err = std::numeric_limits<double>::lowest();
                
                for (const auto& [key, actual_pos] : model_data) {
                    double pred = stages[stage_idx].models[model_idx]->predict(key);
                    double error = pred - static_cast<double>(actual_pos);
                    min_err = std::min(min_err, error);
                    max_err = std::max(max_err, error);
                    
                    // Calculate which next-stage model to route to
                    // Clamp prediction to valid range
                    pred = std::max(0.0, std::min(pred, static_cast<double>(total_records - 1)));
                    
                    // Map prediction to model index
                    size_t next_model_idx = static_cast<size_t>(
                        (pred / total_records) * next_stage_size
                    );
                    
                    // Ensure we don't go out of bounds
                    next_model_idx = std::min(next_model_idx, next_stage_size - 1);
                    
                    stage_data[stage_idx + 1][next_model_idx].push_back({key, actual_pos});
                }
                
                stages[stage_idx].min_errors[model_idx] = min_err;
                stages[stage_idx].max_errors[model_idx] = max_err;
            } else {
                // Last stage - calculate final errors
                double min_err = std::numeric_limits<double>::max();
                double max_err = std::numeric_limits<double>::lowest();
                
                for (const auto& [key, actual_pos] : model_data) {
                    double pred = stages[stage_idx].models[model_idx]->predict(key);
                    double error = pred - static_cast<double>(actual_pos);
                    min_err = std::min(min_err, error);
                    max_err = std::max(max_err, error);
                }
                
                stages[stage_idx].min_errors[model_idx] = min_err;
                stages[stage_idx].max_errors[model_idx] = max_err;
            }
        }
    }
}

size_t RecursiveModelIndex::lookup(double key) const {
    if (stages.empty() || total_records == 0) {
        return 0;
    }
    
    // Traverse RMI stages
    size_t model_idx = 0;
    double prediction = 0;
    
    for (size_t stage_idx = 0; stage_idx < stages.size(); ++stage_idx) {
        const auto& model = stages[stage_idx].models[model_idx];
        if (!model) continue;
        
        prediction = model->predict(key);
        
        if (stage_idx + 1 < stages.size()) {
            // Not the last stage - use prediction to select next model
            size_t next_stage_size = stages[stage_idx + 1].models.size();
            model_idx = static_cast<size_t>(
                std::max(0.0, std::min(prediction / total_records * next_stage_size, 
                                      static_cast<double>(next_stage_size - 1)))
            );
        }
    }
    
    // Final prediction - perform local search
    size_t pos_estimate = static_cast<size_t>(
        std::max(0.0, std::min(prediction, static_cast<double>(total_records - 1)))
    );
    
    // Get error bounds for last stage
    double min_err = stages.back().min_errors[model_idx];
    double max_err = stages.back().max_errors[model_idx];
    
    size_t search_start = static_cast<size_t>(
        std::max(0.0, static_cast<double>(pos_estimate) + min_err)
    );
    size_t search_end = static_cast<size_t>(
        std::min(static_cast<double>(total_records), 
                static_cast<double>(pos_estimate) + max_err + 1)
    );
    
    // Model-biased binary search
    auto it = std::lower_bound(
        sorted_keys.begin() + search_start,
        sorted_keys.begin() + search_end,
        key
    );
    
    if (it == sorted_keys.end()) {
        return total_records;
    }
    
    return it - sorted_keys.begin();
}

size_t RecursiveModelIndex::lower_bound(double key) const {
    return lookup(key);
}

size_t RecursiveModelIndex::upper_bound(double key) const {
    size_t pos = lookup(key);
    // Find first position with key > search key
    while (pos < total_records && sorted_keys[pos] <= key) {
        ++pos;
    }
    return pos;
}

// ============= Statistics Functions =============

size_t RecursiveModelIndex::get_total_size() const {
    size_t total = 0;
    
    // Model sizes
    for (const auto& stage : stages) {
        for (const auto& model : stage.models) {
            if (model) {
                total += model->get_model_size();
            }
        }
        // Error bounds
        total += stage.min_errors.size() * sizeof(double);
        total += stage.max_errors.size() * sizeof(double);
    }
    
    // Data storage
    total += sorted_keys.capacity() * sizeof(double);
    total += sorted_positions.capacity() * sizeof(size_t);
    
    return total;
}

double RecursiveModelIndex::get_average_error() const {
    if (total_records == 0) return 0.0;
    
    double total_error = 0.0;
    size_t count = 0;
    
    // Sample some keys to estimate average error
    size_t sample_size = std::min(total_records, size_t(10000));
    size_t step = total_records / sample_size;
    
    for (size_t i = 0; i < total_records; i += step) {
        double key = sorted_keys[i];
        size_t predicted_pos = lookup(key);
        total_error += std::abs(static_cast<double>(predicted_pos) - static_cast<double>(i));
        ++count;
    }
    
    return count > 0 ? total_error / count : 0.0;
}

void RecursiveModelIndex::print_statistics() const {
    std::cout << "  Number of stages: " << stages.size() << "\n";
    std::cout << "  Total records: " << total_records << "\n";
    std::cout << "  Average prediction error: " << get_average_error() << " positions\n";
    
    for (size_t i = 0; i < stages.size(); ++i) {
        std::cout << "  Stage " << i << ": " << stages[i].models.size() << " models\n";
    }
}
