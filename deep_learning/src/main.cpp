#include <torch/torch.h>
#include <torch/data.h>
#include <iostream>
#include "dcp.h"
#include "loss.h"
#include "data_loader.h"

template <typename Loader>
std::tuple<float, float> train_one_epoch(DCP& net,
                                         Loader& loader,
                                         torch::optim::Optimizer& opt,
                                         bool use_cycle) {
    net->train();
    float total_loss = 0;
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
        std::cout << "Input src shape: " << src.sizes() << std::endl;
        auto tgt = torch::stack(tgt_list).to(torch::kCUDA);
        auto rot_ab = torch::stack(rot_ab_list).to(torch::kCUDA);
        auto trans_ab = torch::stack(trans_ab_list).to(torch::kCUDA);

        opt.zero_grad();

        auto [rot_pred, trans_pred] = net->forward(src, tgt);

        // Identity loss
        auto batch_size = src.size(0);
        auto identity = torch::eye(3, torch::device(torch::kCUDA).dtype(torch::kFloat32)).unsqueeze(0).repeat({(long)batch_size,1,1});
        auto loss = torch::mse_loss(rot_pred.transpose(1,2).matmul(rot_ab), identity)
                    + torch::mse_loss(trans_pred, trans_ab);

        if(use_cycle) {
            // cycle consistency
            auto rot_ba_gt = torch::stack(rot_ba_list).to(torch::kCUDA);
            auto trans_ba_gt = torch::stack(trans_ba_list).to(torch::kCUDA);

            auto rot_loss = torch::mse_loss(rot_ba_gt.matmul(rot_pred), identity.clone());
            auto trans_loss = torch::mean((rot_ba_gt.transpose(1,2).matmul(trans_pred.view({(long)batch_size, 3, 1})).view({(long)batch_size, 3}) + trans_ba_gt).pow(2));
            loss = loss + 0.1 * (rot_loss + trans_loss);
        }

        loss.backward();
        opt.step();

        total_loss += loss.template item<float>() * batch_size;
        total_samples += batch_size;
    }

    return { total_loss / total_samples, static_cast<float>(total_samples) };
}

template <typename Loader>
std::tuple<float,float> test_one_epoch(DCP& net,
                                       Loader& loader,
                                       bool use_cycle) {
    net->eval();
    float total_loss = 0;
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

        auto [rot_pred, trans_pred] = net->forward(src, tgt);

        auto identity = torch::eye(3, torch::device(torch::kCUDA).dtype(torch::kFloat32)).unsqueeze(0).repeat({(long)batch_size,1,1});
        auto loss = torch::mse_loss(rot_pred.transpose(1,2).matmul(rot_ab), identity)
                    + torch::mse_loss(trans_pred, trans_ab);

        if(use_cycle) {
            // cycle consistency
            auto rot_ba_gt = torch::stack(rot_ba_list).to(torch::kCUDA);
            auto trans_ba_gt = torch::stack(trans_ba_list).to(torch::kCUDA);


            auto rot_loss = torch::mse_loss(rot_ba_gt.matmul(rot_pred), identity.clone());
            auto trans_loss = torch::mean((rot_ba_gt.transpose(1,2).matmul(trans_pred.view({(long)batch_size, 3, 1})).view({(long)batch_size, 3}) + trans_ba_gt).pow(2));
            loss = loss + 0.1 * (rot_loss + trans_loss);
        }

        total_loss += loss.template item<float>() * batch_size;
        total_samples += batch_size;
    }

    return { total_loss / total_samples, static_cast<float>(total_samples) };
}


int main() {
    int batch_size = 32;
    int embedding_dims = 512;
    int layers = 6;
    int n_heads = 4;
    int ff_dims = 1024;
    double dropout = 0.0;
    float lr = 0.001;
    int epochs = 250;
    bool use_cycle = true;

    torch::Device device(torch::kCUDA);
    DCP model(embedding_dims, layers, n_heads, ff_dims, dropout);
    model->to(device);

    auto train_dataset = ModelNet40Dataset("../data", "train");
    auto train_loader = torch::data::make_data_loader(std::move(train_dataset), batch_size);

    auto test_dataset = ModelNet40Dataset("../data", "test");
    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), batch_size);

    torch::optim::Adam optimizer(
        model->parameters(),
        torch::optim::AdamOptions(1e-4)
    );

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        // optimizer.zero_grad();

        // auto src = torch::rand({1,1024,3}).to(torch::kCUDA);
        // auto tgt = torch::rand({1,1024,3}).to(torch::kCUDA);

        // auto [R, t] = model->forward(src, tgt);

        // auto loss = torch::norm(R) + torch::norm(t);
        // loss.backward();
        // optimizer.step();

        auto [train_loss, _] = train_one_epoch(model, *train_loader, optimizer, use_cycle);
        auto [test_loss, __] = test_one_epoch(model, *test_loader, use_cycle);

        std::cout << "Epoch " << epoch << " loss " << train_loss << std::endl;
    }
}