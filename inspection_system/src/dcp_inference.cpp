#include "dcp_inference.h"
#include <iostream>
#include <cmath>

DCPInference::DCPInference(const std::string& model_path) {
    device = torch::kCUDA;

    model = DCP(512, 6, 4, 1024, 0.1);

    try {
        model->to(device);

        torch::serialize::InputArchive archive;
        archive.load_from(model_path);
        model->load(archive);
        model->to(device);
        model->eval();
        std::cout << "Model loaded successfully from " << model_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
    }
}

torch::Tensor DCPInference::preprocess(const Eigen::MatrixXf& pc) {
    Eigen::Vector3f centroid = pc.colwise().mean();
    Eigen::MatrixXf centered = pc.rowwise() - centroid.transpose();

    float max_norm = 0.0f;
    for (int i = 0; i < centered.rows(); ++i) {
        max_norm = std::max(max_norm, centered.row(i).norm());
    }
    Eigen::MatrixXf normalized = centered / max_norm;

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tensor = torch::from_blob(normalized.data(), {normalized.rows(), 3}, options).clone();
    
    tensor = tensor.transpose(1, 0).unsqueeze(0); 
    return tensor.to(device);
}

RegistrationResult DCPInference::predict(const Eigen::MatrixXf& src_pc, const Eigen::MatrixXf& tgt_pc) {
    torch::NoGradGuard no_grad;

    torch::Tensor src_tensor = preprocess(src_pc);
    torch::Tensor tgt_tensor = preprocess(tgt_pc);

    auto [R_pred, t_pred] = model->forward(src_tensor, tgt_tensor);
    
    at::Tensor R_tensor = R_pred.detach().cpu().squeeze(0);
    at::Tensor t_tensor = t_pred.detach().cpu().squeeze(0);

    RegistrationResult result;
    auto R_final = R_tensor.contiguous();
    auto t_final = t_tensor.contiguous();

    std::memcpy(result.R.data(), R_final.data_ptr<float>(), 9 * sizeof(float));
    std::memcpy(result.t.data(), t_final.data_ptr<float>(), 3 * sizeof(float));

    result.R.transposeInPlace();

    return result;
}