#ifndef DCP_INFERENCE_H
#define DCP_INFERENCE_H

#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <Eigen/Dense>

#include "dcp.h"

struct RegistrationResult {
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
};

class DCPInference {
public:
    DCPInference(const std::string& model_path);

    Eigen::MatrixXf preprocess_eigen(const Eigen::MatrixXf& pc);
    
    RegistrationResult predict(const Eigen::MatrixXf& src_pc, const Eigen::MatrixXf& tgt_pc);

private:
    DCP model{nullptr};
    
    torch::Tensor preprocess(const Eigen::MatrixXf& pc);
    
    torch::Device device{torch::kCUDA};
};

#endif