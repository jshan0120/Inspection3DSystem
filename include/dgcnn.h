#pragma once

#ifndef DGCNN_H
#define DGCNN_H

#include <torch/torch.h>

struct DGCNNImpl : torch::nn::Module
{
    DGCNNImpl(int embedding_dims);

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr}, conv5{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr}, bn5{nullptr};

    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(DGCNN);

#endif