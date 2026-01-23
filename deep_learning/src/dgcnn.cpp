#include "dgcnn.h"
#include "knn.h"

DGCNNImpl::DGCNNImpl(int embedding_dims)
{
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(6, 64, 1).bias(false)));
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 1).bias(false)));
    conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 1).bias(false)));
    conv4 = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 1).bias(false)));
    conv5 = register_module("conv5", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, embedding_dims, 1).bias(false)));

    bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(64));
    bn3 = register_module("bn3", torch::nn::BatchNorm2d(128));
    bn4 = register_module("bn4", torch::nn::BatchNorm2d(256));
    bn5 = register_module("bn5", torch::nn::BatchNorm2d(embedding_dims));
}

torch::Tensor DGCNNImpl::forward(torch::Tensor x)
{
    int64_t batch_size = x.size(0);
    int64_t num_points = x.size(2);

    torch::Tensor edge = get_graph_feature(x);

    torch::Tensor x1 = std::get<0>(torch::relu(bn1(conv1(edge))).max(-1, true));
    torch::Tensor x2 = std::get<0>(torch::relu(bn2(conv2(x1))).max(-1, true));
    torch::Tensor x3 = std::get<0>(torch::relu(bn3(conv3(x2))).max(-1, true));
    torch::Tensor x4 = std::get<0>(torch::relu(bn4(conv4(x3))).max(-1, true));

    torch::Tensor x_cat = torch::cat({x1, x2, x3, x4}, 1);

    torch::Tensor x_out = torch::relu(bn5(conv5(x_cat)));
    x_out = x_out.squeeze(-1);
    // x_out = x_out.transpose(1, 2); 

    return x_out;
}