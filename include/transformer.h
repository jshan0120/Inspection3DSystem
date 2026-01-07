#pragma once

#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <torch/torch.h>
#include "encoder.h"

struct TransformerImpl : torch::nn::Module
{
    Encoder encoder{nullptr};
    Encoder decoder{nullptr};

    TransformerImpl(int64_t emb_dim, int64_t n_layers, int64_t n_heads, int64_t ff_dim, double dropout=0.0);

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor fx, torch::Tensor fy);
};
TORCH_MODULE(Transformer);

#endif