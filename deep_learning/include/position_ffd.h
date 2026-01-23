#pragma once

#ifndef POSITION_FFD_H
#define POSITION_FFD_H

#include <torch/torch.h>

struct PositionwiseFeedForwardImpl : torch::nn::Module
{
    torch::nn::Linear w_1{nullptr}, w_2{nullptr};
    torch::nn::Sequential norm{nullptr};
    double dropout;

    PositionwiseFeedForwardImpl(int64_t d_model, int64_t d_ff, double drop);

    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(PositionwiseFeedForward);

#endif