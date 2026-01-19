#pragma once

#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <iostream>
#include <vector>
#include <string>
#include <highfive/H5File.hpp>
#include <torch/torch.h>

struct ModelNet40Example 
{
    torch::Tensor data_src;
    torch::Tensor data_tgt;
    torch::Tensor rot_ab;
    torch::Tensor trans_ab;
    torch::Tensor rot_ba;
    torch::Tensor trans_ba;
};

class ModelNet40Dataset : public torch::data::Dataset<ModelNet40Dataset, ModelNet40Example> {
public:
    ModelNet40Dataset(std::string root, std::string partition = "train", 
                      int num_points = 1024, bool gaussian_noise = false, float factor = 4.0);

    ModelNet40Example get(size_t index) override;

    torch::optional<size_t> size() const override;

private:
    void load_data(const std::string& root);
    
    torch::Tensor jitter_pointcloud(torch::Tensor& pc);

    torch::Tensor all_data;
    torch::Tensor all_labels;
    
    std::string partition_;
    int num_points_;
    bool gaussian_noise_;
    float factor_;
};

#endif