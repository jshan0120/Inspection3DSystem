#include "dcp_inference.h"
#include <iostream>
#include <cmath>

DCPInference::DCPInference(const std::string& model_path) {
    device = torch::kCUDA;

    model = DCP(512, 6, 4, 1024, 0.1);

    try {
        model->to(device);

        torch::load(model, model_path);

        // torch::serialize::InputArchive archive;
        // archive.load_from(model_path, device);
        // model->load(archive); 

        // torch::jit::script::Module container = torch::jit::load(model_path, device);
        // auto params = container.named_parameters();
        // auto buffers = container.named_buffers();

        model->eval();

        std::cout << "Model loaded successfully from " << model_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        std::cerr << e.what() << std::endl;
    }
}

Eigen::MatrixXf DCPInference::preprocess_eigen(const Eigen::MatrixXf& pc) {
    Eigen::Vector3f centroid = pc.colwise().mean();
    Eigen::MatrixXf centered = pc.rowwise() - centroid.transpose();

    float max_norm = 0.0f;
    for (int i = 0; i < centered.rows(); ++i) {
        max_norm = std::max(max_norm, centered.row(i).norm());
    }
    
    if (max_norm > 1e-6f) {
        return centered / max_norm;
    }
    return centered;
}

torch::Tensor DCPInference::preprocess(const Eigen::MatrixXf& pc) {
    Eigen::MatrixXf normalized = preprocess_eigen(pc);

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tensor = torch::from_blob(const_cast<float*>(normalized.data()), 
                                            {normalized.rows(), 3}, options).clone();
    
    tensor = tensor.transpose(1, 0).unsqueeze(0); 
    return tensor.to(device);
}

RegistrationResult DCPInference::predict(const Eigen::MatrixXf& src_pc, const Eigen::MatrixXf& tgt_pc) {
    torch::NoGradGuard no_grad;

    Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> src_row = src_pc;
    Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> tgt_row = tgt_pc;

    auto options = torch::TensorOptions().dtype(torch::kFloat32); 

    torch::Tensor src_tensor = torch::from_blob(src_row.data(), {src_pc.rows(), 3}, options).to(device);
    torch::Tensor tgt_tensor = torch::from_blob(tgt_row.data(), {tgt_pc.rows(), 3}, options).to(device);

    src_tensor = src_tensor.transpose(1, 0).unsqueeze(0);
    tgt_tensor = tgt_tensor.transpose(1, 0).unsqueeze(0);

    auto [R_pred_t, t_pred_t, R_ba_t, t_ba_t] = model->forward(src_tensor, tgt_tensor);
    
    at::Tensor R_cpu = R_pred_t.detach().cpu().squeeze(0);
    at::Tensor t_cpu = t_pred_t.detach().cpu().squeeze(0);
    
    RegistrationResult result;
    result.R = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(R_cpu.data_ptr<float>());
    result.t = Eigen::Map<Eigen::Vector3f>(t_cpu.data_ptr<float>());

    return result;
}