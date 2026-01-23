#pragma once

#ifndef DCP_H
#define DCP_H

#include <torch/torch.h>
#include "dgcnn.h"
#include "transformer.h"
#include "svd_head.h"

struct DCPImpl : torch::nn::Module
{
    DCPImpl(int64_t embedding_dims, int64_t n_layers, int64_t n_heads, int64_t ff_dim, double dropout);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor src, torch::Tensor tgt);

    DGCNN emb_nn{nullptr};
    Transformer pointer{nullptr};
    SVDHead head{nullptr};
    
    bool cycle = false;
};
TORCH_MODULE(DCP);

#endif