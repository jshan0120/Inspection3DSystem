#include <iostream>
#include <random>
#include "H5Cpp.h"
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

Eigen::MatrixXf load_modelnet_h5(const std::string& file_path, int batch_idx, int num_points = 1024) {
    try {
        H5::H5File file(file_path, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet("data");
        H5::DataSpace dataspace = dataset.getSpace();

        hsize_t dims_out[3];
        dataspace.getSimpleExtentDims(dims_out);
        
        int B = dims_out[0];
        int N = dims_out[1];
        int C = dims_out[2];

        std::vector<float> data_buffer(B * N * C);
        dataset.read(data_buffer.data(), H5::PredType::NATIVE_FLOAT);

        Eigen::MatrixXf pc(num_points, 3);
        int start_offset = batch_idx * N * C;

        double step = static_cast<double>(N) / num_points;

        for (int i = 0; i < num_points; ++i) {
            int sampled_idx = static_cast<int>(i * step);
            
            if (sampled_idx >= N) sampled_idx = N - 1;

            pc(i, 0) = data_buffer[start_offset + sampled_idx * C + 0];
            pc(i, 1) = data_buffer[start_offset + sampled_idx * C + 1];
            pc(i, 2) = data_buffer[start_offset + sampled_idx * C + 2];
        }

        return pc;
    } catch (H5::Exception& e) {
        std::cerr << "HDF5 Error: " << e.getDetailMsg() << std::endl;
        return Eigen::MatrixXf::Zero(num_points, 3);
    }
}

int main() {
    std::string model_path = "checkpoints/models/model_best.pt";
    std::string h5_path = "../data/ply_data_test0.h5";
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

    Eigen::MatrixXf raw_pc = load_modelnet_h5(h5_path, 0, 1024);
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