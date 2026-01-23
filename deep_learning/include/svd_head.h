#pragma once
#include <torch/torch.h>

struct SVDHeadImpl : torch::nn::Module {
    SVDHeadImpl(int64_t emb_dims);

    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor src_embedding, 
        torch::Tensor tgt_embedding, 
        torch::Tensor src, 
        torch::Tensor tgt);

    torch::Tensor reflect;
};

TORCH_MODULE(SVDHead);