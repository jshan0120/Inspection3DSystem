#include <iostream>
#include <random>
#include "dcp_inference.h"

float compute_rotation_error(const Eigen::Matrix3f& R_pred, const Eigen::Matrix3f& R_gt) {
    Eigen::Matrix3f R = R_pred * R_gt.transpose();
    float tr = R.trace();
    float cos_theta = (tr - 1.0f) / 2.0f;
    cos_theta = std::clamp(cos_theta, -1.0f, 1.0f);
    return std::acos(cos_theta) * 180.0f / M_PI;
}

float compute_translation_error(const Eigen::Vector3f& t_pred, const Eigen::Vector3f& t_gt) {
    return (t_pred - t_gt).norm();
}

int main() {
    std::string model_path = "checkpoints/models/model_best.pt";
    DCPInference dcp(model_path);

    torch::manual_seed(42);
    if (torch::cuda::is_available()) {
        torch::cuda::manual_seed_all(42);
    }

    // std::random_device rd;
    // std::mt19937 gen(rd());
    unsigned int seed = 42;
    std::mt19937 gen(seed);

    std::uniform_real_distribution<float> rot_dist(-M_PI/4.0f, M_PI/4.0f);
    std::uniform_real_distribution<float> trans_dist(-0.5f, 0.5f);

    Eigen::AngleAxisf roll(rot_dist(gen),  Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitch(rot_dist(gen), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yaw(rot_dist(gen),   Eigen::Vector3f::UnitZ());
    Eigen::Matrix3f R_gt = (yaw * pitch * roll).toRotationMatrix();

    Eigen::Vector3f t_gt(trans_dist(gen), trans_dist(gen), trans_dist(gen));

    Eigen::MatrixXf raw_pc = Eigen::MatrixXf::Random(1024, 3);
    Eigen::MatrixXf src_pc = dcp.preprocess_eigen(raw_pc);
    Eigen::MatrixXf tgt_pc = (src_pc * R_gt.transpose()).rowwise() + t_gt.transpose();

    RegistrationResult res = dcp.predict(src_pc, tgt_pc);

    float rot_err = compute_rotation_error(res.R, R_gt);
    float trans_err = compute_translation_error(res.t, t_gt);

    std::cout << "\n========================================" << std::endl;
    std::cout << " [Ground Truth]" << std::endl;
    std::cout << " R_gt:\n" << R_gt << std::endl;
    std::cout << " t_gt: " << t_gt.transpose() << std::endl;
    
    std::cout << "\n [Model Prediction]" << std::endl;
    std::cout << " R_pred:\n" << res.R << std::endl;
    std::cout << " t_pred: " << res.t.transpose() << std::endl;

    std::cout << "\n [Registration Errors]" << std::endl;
    std::cout << " >> Rotation Error (deg): " << rot_err << std::endl;
    std::cout << " >> Translation Error:     " << trans_err << std::endl;
    std::cout << "========================================\n" << std::endl;

    return 0;
}