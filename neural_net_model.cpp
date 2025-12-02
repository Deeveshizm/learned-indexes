#include "learned_index.hpp"
#include <cmath>
#include <algorithm>
#include <random>

NeuralNetModel::NeuralNetModel(size_t hidden_size, size_t num_layers) 
    : hidden_size(hidden_size), num_layers(num_layers),
      x_min_(0.0), x_max_(1.0), x_range_(1.0), y_max_(1.0), use_log_(false) {  // Initialize use_log_
    
    weights.resize(num_layers);
    biases.resize(num_layers);
    
    for (size_t l = 0; l < num_layers; ++l) {
        size_t input_size = (l == 0) ? 1 : hidden_size;
        size_t output_size = (l == num_layers - 1) ? 1 : hidden_size;
        
        weights[l].resize(input_size * output_size);
        biases[l].resize(output_size);
    }
}

void NeuralNetModel::train(const std::vector<std::pair<double, size_t>>& data) {
    if (data.empty()) return;
    
    size_t n = data.size();
    
    std::vector<std::pair<double, size_t>> transformed_data;
    transformed_data.reserve(n);
    
    double min_key = data.front().first;
    double max_key = data.back().first;
    use_log_ = (max_key / (min_key + 1.0)) > 100.0;  // Use log if >100x range
    
    if (use_log_) {
        // Apply log transformation
        for (const auto& [key, pos] : data) {
            double log_key = std::log(key + 1.0);  // +1 to handle zero
            transformed_data.emplace_back(log_key, pos);
        }
    } else {
        transformed_data = data;
    }
    
    // STORE normalization parameters for inference
    x_min_ = transformed_data.front().first;
    x_max_ = transformed_data.back().first;
    x_range_ = x_max_ - x_min_;
    if (x_range_ == 0) x_range_ = 1.0;
    
    y_max_ = static_cast<double>(n - 1);
    
    // Initialize weights with He initialization
    std::mt19937 rng(42);
    std::normal_distribution<double> weight_dist(0.0, std::sqrt(2.0 / hidden_size));
    
    for (auto& w : weights) {
        for (auto& val : w) {
            val = weight_dist(rng);
        }
    }
    
    for (auto& b : biases) {
        std::fill(b.begin(), b.end(), 0.0);
    }
    
    // Training hyperparameters - OPTIMIZED FOR SKEWED DISTRIBUTIONS
    const size_t num_epochs = 100;          // More epochs for better convergence
    const double learning_rate = 0.05;      // Much higher learning rate
    const size_t batch_size = 128;          // Smaller batches for more updates
    
    // Mini-batch SGD
    std::vector<size_t> indices(n);
    for (size_t i = 0; i < n; ++i) indices[i] = i;
    
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);
        
        for (size_t batch_start = 0; batch_start < n; batch_start += batch_size) {
            size_t batch_end = std::min(batch_start + batch_size, n);
            size_t current_batch_size = batch_end - batch_start;
            
            // Accumulate gradients over batch
            std::vector<std::vector<double>> weight_grads(weights.size());
            std::vector<std::vector<double>> bias_grads(biases.size());
            
            for (size_t l = 0; l < weights.size(); ++l) {
                weight_grads[l].resize(weights[l].size(), 0.0);
                bias_grads[l].resize(biases[l].size(), 0.0);
            }
            
            for (size_t b = batch_start; b < batch_end; ++b) {
                size_t idx = indices[b];
                double x = (transformed_data[idx].first - x_min_) / x_range_;
                double y_true = transformed_data[idx].second / y_max_;
                
                // Forward pass
                std::vector<std::vector<double>> layer_outputs(num_layers + 1);
                layer_outputs[0] = {x};
                
                for (size_t l = 0; l < num_layers; ++l) {
                    size_t input_size = (l == 0) ? 1 : hidden_size;
                    size_t output_size = (l == num_layers - 1) ? 1 : hidden_size;
                    
                    layer_outputs[l + 1].resize(output_size, 0.0);
                    
                    for (size_t j = 0; j < output_size; ++j) {
                        double sum = biases[l][j];
                        for (size_t i = 0; i < input_size; ++i) {
                            sum += layer_outputs[l][i] * weights[l][i * output_size + j];
                        }
                        
                        if (l < num_layers - 1) {
                            layer_outputs[l + 1][j] = std::max(0.0, sum);  // ReLU
                        } else {
                            layer_outputs[l + 1][j] = sum;  // Linear output
                        }
                    }
                }
                
                double y_pred = layer_outputs.back()[0];
                double error = y_pred - y_true;
                
                // Backward pass
                std::vector<std::vector<double>> deltas(num_layers + 1);
                deltas.back() = {2.0 * error};
                
                for (int l = static_cast<int>(num_layers) - 1; l >= 0; --l) {
                    size_t input_size = (l == 0) ? 1 : hidden_size;
                    size_t output_size = (l == static_cast<int>(num_layers) - 1) ? 1 : hidden_size;
                    
                    if (l < static_cast<int>(num_layers) - 1) {
                        deltas[l + 1].resize(hidden_size, 0.0);
                        for (size_t i = 0; i < hidden_size; ++i) {
                            double sum = 0.0;
                            size_t next_size = (l == static_cast<int>(num_layers) - 2) ? 1 : hidden_size;
                            for (size_t j = 0; j < next_size; ++j) {
                                sum += deltas[l + 2][j] * weights[l + 1][i * next_size + j];
                            }
                            deltas[l + 1][i] = (layer_outputs[l + 1][i] > 0) ? sum : 0.0;
                        }
                    }
                    
                    // Accumulate gradients
                    for (size_t j = 0; j < output_size; ++j) {
                        bias_grads[l][j] += deltas[l + 1][j];
                        for (size_t i = 0; i < input_size; ++i) {
                            weight_grads[l][i * output_size + j] += 
                                layer_outputs[l][i] * deltas[l + 1][j];
                        }
                    }
                }
            }
            
            // Update weights
            for (size_t l = 0; l < weights.size(); ++l) {
                for (size_t i = 0; i < weights[l].size(); ++i) {
                    weights[l][i] -= learning_rate * weight_grads[l][i] / current_batch_size;
                }
                for (size_t i = 0; i < biases[l].size(); ++i) {
                    biases[l][i] -= learning_rate * bias_grads[l][i] / current_batch_size;
                }
            }
        }
    }
}

double NeuralNetModel::predict(double key) const {
    // Apply log transformation if it was used during training
    double transformed_key = use_log_ ? std::log(key + 1.0) : key;
    
    // FIX: Use stored normalization parameters
    double x = (transformed_key - x_min_) / x_range_;
    
    std::vector<double> activations = {x};
    
    for (size_t l = 0; l < num_layers; ++l) {
        size_t input_size = (l == 0) ? 1 : hidden_size;
        size_t output_size = (l == num_layers - 1) ? 1 : hidden_size;
        
        std::vector<double> next_activations(output_size, 0.0);
        
        for (size_t j = 0; j < output_size; ++j) {
            double sum = biases[l][j];
            for (size_t i = 0; i < input_size; ++i) {
                sum += activations[i] * weights[l][i * output_size + j];
            }
            
            if (l < num_layers - 1) {
                next_activations[j] = std::max(0.0, sum);
            } else {
                next_activations[j] = sum;
            }
        }
        
        activations = next_activations;
    }
    
    return activations[0] * y_max_;
}

size_t NeuralNetModel::get_model_size() const {
    size_t total = 0;
    for (const auto& w : weights) total += w.size() * sizeof(double);
    for (const auto& b : biases) total += b.size() * sizeof(double);
    // Add normalization parameters
    total += 4 * sizeof(double);  // x_min_, x_max_, x_range_, y_max_
    total += sizeof(bool);         // use_log_
    return total;
}
