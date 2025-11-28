#include "learned_index.hpp"

void LinearModel::train(const std::vector<std::pair<double, size_t>>& data) {
    if (data.empty()) return;

    //compute statistics for linear regression
    size_t n = data.size();
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;

    for (const auto& [key, pos] : data) {
        sum_x += key;
        sum_y += pos;
        sum_xy += key * pos;
        sum_x2 += key * key;
    }

    double mean_x = sum_x / n;
    double mean_y = sum_y / n;

    //calculate slope and intercept
    double denominator = sum_x2 - n * mean_x * mean_x;
    if (std::abs(denominator) < 1e-10) {
        slope = 0;
        intercept = mean_y;
    } else {
        slope = (sum_xy - n * mean_x * mean_y) / denominator;
        intercept = mean_y - slope * mean_x;
    }
}
