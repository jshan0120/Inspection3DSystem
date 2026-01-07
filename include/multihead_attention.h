#pragma once

#ifndef MULTIHEAD_ATTENTION_H
#define MULTIHEAD_ATTENTION_H

#include <torch/torch.h>
#include <vector>
#include <cmath>

struct MultiHeadAttentionImpl : torch::nn::Module {
    int64_t n_heads;
    int64_t d_model;
    int64_t d_k;
    torch::nn::Linear w_q{nullptr}, w_k{nullptr}, w_v{nullptr}, w_o{nullptr};

    MultiHeadAttentionImpl(int64_t heads, int64_t model_dim);

    torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);
};
TORCH_MODULE(MultiHeadAttention);

#endif