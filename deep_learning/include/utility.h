#pragma once

#ifndef UTILITY_H
#define UTILITY_H

#include <torch/torch.h>
#include <filesystem>

namespace fs = std::filesystem;

torch::Tensor compute_rotation_error(const torch::Tensor& R_pred, const torch::Tensor& R_gt) {
    auto R_diff = torch::matmul(R_pred.transpose(1, 2), R_gt);
    auto tr = torch::diagonal(R_diff, 0, -1, -2).sum(-1);
    auto cos_theta = (tr - 1) / 2.0;
    cos_theta = torch::clamp(cos_theta, -0.999999, 0.999999);
    return torch::acos(cos_theta).mean();
}

void initialize_folders(const std::string& exp_name) {
    std::string checkpoint_dir = "checkpoints/models";

    if (!fs::exists(checkpoint_dir)) {
        if (fs::create_directories(checkpoint_dir)) {
            std::cout << "Created directory: " << checkpoint_dir << std::endl;
        }
    }
}

#endif