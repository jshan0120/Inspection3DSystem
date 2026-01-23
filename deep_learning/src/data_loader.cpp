#include "data_loader.h"

#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;

ModelNet40Dataset::ModelNet40Dataset(std::string root, std::string partition, 
                                     int num_points, bool gaussian_noise, float factor)
    : partition_(partition), num_points_(num_points), 
      gaussian_noise_(gaussian_noise), factor_(factor) 
{
    load_data(root);
}

void ModelNet40Dataset::load_data(const std::string& root) 
{
    std::vector<torch::Tensor> data_list;

    for (const auto& entry : fs::directory_iterator(root)) {
        std::string filename = entry.path().filename().string();
        if (filename.find("ply_data_" + partition_) != std::string::npos && filename.find(".h5") != std::string::npos) {
            
            HighFive::File file(entry.path().string(), HighFive::File::ReadOnly);
            auto dataset = file.getDataSet("data");
            
            std::vector<size_t> dims = dataset.getSpace().getDimensions(); 
            
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            torch::Tensor tensor_data = torch::empty({(long)dims[0], (long)dims[1], (long)dims[2]}, options);
            
            dataset.read_raw(tensor_data.data_ptr<float>());
            
            data_list.push_back(tensor_data.clone());
        }
    }

    if (data_list.empty()) {
        throw std::runtime_error("No HDF5 files found in: " + root);
    }
    all_data = torch::cat(data_list, 0); 
}

ModelNet40Example ModelNet40Dataset::get(size_t index) 
{
    torch::NoGradGuard no_grad;

    if (partition_ != "train") {
        torch::manual_seed(index); 
    }

    auto f32_options = torch::TensorOptions().dtype(torch::kFloat32);

    torch::Tensor pc = all_data[index].slice(0, 0, num_points_); 
    if (gaussian_noise_) pc = jitter_pointcloud(pc);

    float anglex = (torch::rand({1}).item<float>() * M_PI) / factor_;
    float angley = (torch::rand({1}).item<float>() * M_PI) / factor_;
    float anglez = (torch::rand({1}).item<float>() * M_PI) / factor_;

    auto cosx = std::cos(anglex), sinx = std::sin(anglex);
    auto cosy = std::cos(angley), siny = std::sin(angley);
    auto cosz = std::cos(anglez), sinz = std::sin(anglez);

    auto Rx = torch::tensor({{1.0f, 0.0f, 0.0f}, {0.0f, (float)cosx, (float)-sinx}, {0.0f, (float)sinx, (float)cosx}}, f32_options);
    auto Ry = torch::tensor({{(float)cosy, 0.0f, (float)siny}, {0.0f, 1.0f, 0.0f}, {(float)-siny, 0.0f, (float)cosy}}, f32_options);
    auto Rz = torch::tensor({{(float)cosz, (float)-sinz, 0.0f}, {(float)sinz, (float)cosz, 0.0f}, {0.0f, 0.0f, 1.0f}}, f32_options);

    torch::Tensor rot_ab = Rx.mm(Ry).mm(Rz);
    torch::Tensor rot_ba = rot_ab.t();

    torch::Tensor trans_ab = (torch::rand({3}, f32_options) - 0.5);
    torch::Tensor trans_ba = -rot_ba.mv(trans_ab);

    torch::Tensor src = pc.t().contiguous();
    torch::Tensor tgt = (rot_ab.mm(src) + trans_ab.view({3, 1})).contiguous();

    auto perm_src = torch::randperm(num_points_, torch::kLong);
    auto perm_tgt = torch::randperm(num_points_, torch::kLong);

    src = src.index_select(1, perm_src).to(torch::kFloat32).contiguous();
    tgt = tgt.index_select(1, perm_tgt).to(torch::kFloat32).contiguous();

    ModelNet40Example example;
    example.data_src = src;
    example.data_tgt = tgt;
    example.rot_ab = rot_ab.to(torch::kFloat32).contiguous();
    example.trans_ab = trans_ab.to(torch::kFloat32).contiguous();
    example.rot_ba = rot_ba.to(torch::kFloat32).contiguous();
    example.trans_ba = trans_ba.to(torch::kFloat32).contiguous();

    return example;
}

torch::Tensor ModelNet40Dataset::jitter_pointcloud(torch::Tensor& pc) {
    return pc + torch::clamp(0.01 * torch::randn_like(pc), -0.05, 0.05);
}

torch::optional<size_t> ModelNet40Dataset::size() const {
    return all_data.size(0);
}