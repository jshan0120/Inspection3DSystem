#include <torch/script.h>
#include <torch/torch.h>
#include <torch/data.h>
#include <torch/csrc/jit/frontend/tracer.h>

#include "iostream.h"
#include "dcp.h"
#include "loss.h"
#include "data_loader.h"
#include "utility.h"

struct EpochResult {
    float loss;
    float rot_error_deg;
    float trans_error_mat;
};

template <typename Loader>
EpochResult train_one_epoch(DCP& net,
                                         Loader& loader,
                                         torch::optim::Optimizer& opt,
                                         bool use_cycle) {
    net->train();
    float total_loss = 0, total_rot_deg = 0, total_trans_err = 0;
    size_t total_samples = 0;

    for (auto& batch_vec : loader) {
        std::vector<torch::Tensor> src_list, tgt_list, rot_ab_list, trans_ab_list, rot_ba_list, trans_ba_list;
        for (auto& ex : batch_vec) {
            src_list.push_back(ex.data_src.to(torch::kFloat32));
            tgt_list.push_back(ex.data_tgt.to(torch::kFloat32));
            rot_ab_list.push_back(ex.rot_ab.to(torch::kFloat32));
            trans_ab_list.push_back(ex.trans_ab.to(torch::kFloat32));
            rot_ba_list.push_back(ex.rot_ba.to(torch::kFloat32));
            trans_ba_list.push_back(ex.trans_ba.to(torch::kFloat32));
        }

        auto src = torch::stack(src_list).to(torch::kCUDA);
        auto tgt = torch::stack(tgt_list).to(torch::kCUDA);
        auto rot_ab = torch::stack(rot_ab_list).to(torch::kCUDA);
        auto trans_ab = torch::stack(trans_ab_list).to(torch::kCUDA);

        opt.zero_grad();

        auto [rot_pred, trans_pred, rot_ba_pred, trans_ba_pred] = net->forward(src, tgt);

        auto batch_size = src.size(0);
        auto identity = torch::eye(3, src.options()).unsqueeze(0).expand({batch_size, 3, 3});
        auto loss = pose_loss(rot_pred, trans_pred, rot_ab, trans_ab, src);

        auto r_err = compute_rotation_error(rot_pred, rot_ab);
        auto t_err = torch::norm(trans_pred - trans_ab, 2, 1);

        if (use_cycle) {
            auto rot_ba_gt = torch::stack(rot_ba_list).to(torch::kCUDA);
            auto trans_ba_gt = torch::stack(trans_ba_list).to(torch::kCUDA);

            auto rot_loss_ba = torch::mse_loss(rot_ba_pred, rot_ba_gt);
            auto trans_loss_ba = torch::mse_loss(trans_ba_pred, trans_ba_gt);

            auto identity = torch::eye(3, src.options()).unsqueeze(0).expand({batch_size, 3, 3});
            auto cycle_rot_loss = torch::mse_loss(torch::matmul(rot_ba_pred, rot_pred), identity);

            loss = loss + 0.1 * (rot_loss_ba + trans_loss_ba + cycle_rot_loss);
        }

        loss.backward();
        opt.step();

        total_loss += loss.template item<float>() * batch_size;
        total_rot_deg += r_err.sum().item<float>();
        total_trans_err += t_err.sum().item<float>();
        total_samples += batch_size;
    }

    return { total_loss / total_samples, 
             total_rot_deg / total_samples, 
             total_trans_err / total_samples };
}

template <typename Loader>
EpochResult test_one_epoch(DCP& net,
                          Loader& loader,
                          bool use_cycle) {
    net->eval();
    torch::NoGradGuard no_grad;

    float total_loss = 0, total_rot_deg = 0, total_trans_err = 0;
    size_t total_samples = 0;

    for (auto& batch_vec : loader) {
        std::vector<torch::Tensor> src_list, tgt_list, rot_ab_list, trans_ab_list, rot_ba_list, trans_ba_list;
        for (auto& ex : batch_vec) {
            src_list.push_back(ex.data_src.to(torch::kFloat32));
            tgt_list.push_back(ex.data_tgt.to(torch::kFloat32));
            rot_ab_list.push_back(ex.rot_ab.to(torch::kFloat32));
            trans_ab_list.push_back(ex.trans_ab.to(torch::kFloat32));
            rot_ba_list.push_back(ex.rot_ba.to(torch::kFloat32));
            trans_ba_list.push_back(ex.trans_ba.to(torch::kFloat32));
        }

        auto src = torch::stack(src_list).to(torch::kCUDA);
        auto tgt = torch::stack(tgt_list).to(torch::kCUDA);
        auto rot_ab = torch::stack(rot_ab_list).to(torch::kCUDA);
        auto trans_ab = torch::stack(trans_ab_list).to(torch::kCUDA);

        auto batch_size = src.size(0);

        auto [rot_pred, trans_pred, rot_ba_pred, trans_ba_pred] = net->forward(src, tgt);

        auto identity = torch::eye(3, torch::device(torch::kCUDA).dtype(torch::kFloat32)).unsqueeze(0).repeat({(long)batch_size, 1, 1});
        auto loss = pose_loss(rot_pred, trans_pred, rot_ab, trans_ab, src);

        auto r_err = compute_rotation_error(rot_pred, rot_ab);
        auto t_err = torch::norm(trans_pred - trans_ab, 2, 1);

        if (use_cycle) {
            auto rot_ba_gt = torch::stack(rot_ba_list).to(torch::kCUDA);
            auto trans_ba_gt = torch::stack(trans_ba_list).to(torch::kCUDA);

            auto rot_loss_ba = torch::mse_loss(rot_ba_pred, rot_ba_gt);
            auto trans_loss_ba = torch::mse_loss(trans_ba_pred, trans_ba_gt);

            auto identity = torch::eye(3, src.options()).unsqueeze(0).expand({batch_size, 3, 3});
            auto cycle_rot_loss = torch::mse_loss(torch::matmul(rot_ba_pred, rot_pred), identity);

            loss = loss + 0.1 * (rot_loss_ba + trans_loss_ba + cycle_rot_loss);
        }

        total_loss += loss.template item<float>() * batch_size;
        total_rot_deg += r_err.sum().item<float>();
        total_trans_err += t_err.sum().item<float>();
        total_samples += batch_size;
    }

    return { total_loss / total_samples, 
             total_rot_deg / total_samples, 
             total_trans_err / total_samples };
}


int main() {
    IOStream text_logger("train_log.txt");
    text_logger.cprint("Initializing training...");

    std::string exp_name = "dcp_v2";
    initialize_folders(exp_name);

    int batch_size = 32;
    int embedding_dims = 512;
    int layers = 6;
    int n_heads = 4;
    int ff_dims = 1024;
    double dropout = 0.1;
    float lr = 0.0001;
    int epochs = 250;
    bool use_cycle = true;

    float best_test_loss = std::numeric_limits<float>::max();

    torch::Device device(torch::kCUDA);
    DCP model(embedding_dims, layers, n_heads, ff_dims, dropout);
    model->to(device);

    auto train_dataset = ModelNet40Dataset("../data", "train");
    auto train_loader = torch::data::make_data_loader(std::move(train_dataset), torch::data::DataLoaderOptions().batch_size(batch_size).workers(4).enforce_ordering(false));

    auto test_dataset = ModelNet40Dataset("../data", "test");
    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), batch_size);

    torch::optim::Adam optimizer(
        model->parameters(),
        torch::optim::AdamOptions(lr)
    );

    text_logger.cprint("Start training...");
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        if (epoch == 75 || epoch == 150 || epoch == 200) {
            for (auto& options : optimizer.param_groups()) {
                static_cast<torch::optim::AdamOptions&>(options.options()).lr(
                    static_cast<torch::optim::AdamOptions&>(options.options()).lr() * 0.1
                );
            }
            text_logger.cprint("Learning rate decreased.");
        }

        EpochResult train_res = train_one_epoch(model, *train_loader, optimizer, use_cycle);
        EpochResult test_res = test_one_epoch(model, *test_loader, use_cycle);

        std::string log_msg = "Epoch [" + std::to_string(epoch) + "/" + std::to_string(epochs) + "]\n" +
                          "  [Train] Loss: " + std::to_string(train_res.loss) + 
                          ", RotErr(deg): " + std::to_string(train_res.rot_error_deg) + 
                          ", TransErr: " + std::to_string(train_res.trans_error_mat) + "\n" +
                          "  [Test ] Loss: " + std::to_string(test_res.loss) + 
                          ", RotErr(deg): " + std::to_string(test_res.rot_error_deg) + 
                          ", TransErr: " + std::to_string(test_res.trans_error_mat);
        text_logger.cprint(log_msg);

        if (test_res.loss < best_test_loss) {
            best_test_loss = test_res.loss;
            model->eval();
            torch::NoGradGuard no_grad;

            auto s = torch::randn({1, 3, 1024}).to(device);
            auto t = torch::randn({1, 3, 1024}).to(device);
            
            std::vector<torch::jit::IValue> inputs = {s, t};

            try {
                torch::serialize::OutputArchive archive;
                model->save(archive);
                archive.save_to("checkpoints/models/model_best.pt");

                text_logger.cprint("  --> New Best Model Weights Saved!");
            } catch (const std::exception& e) {
                text_logger.cprint("Save failed: " + std::string(e.what()));
            }
        }

        if (epoch % 1 == 0) {
            model->eval();
            torch::NoGradGuard no_grad;

            auto s = torch::randn({1, 3, 1024}).to(device);
            auto t = torch::randn({1, 3, 1024}).to(device);
            
            std::vector<torch::jit::IValue> inputs = {s, t};

            try {
                torch::serialize::OutputArchive archive;
                model->save(archive);
                archive.save_to("checkpoints/models/model_epoch_" + std::to_string(epoch) + ".pt");
            } catch (const std::exception& e) {
                text_logger.cprint("Save failed: " + std::string(e.what()));
            }
        }
        
    }
}