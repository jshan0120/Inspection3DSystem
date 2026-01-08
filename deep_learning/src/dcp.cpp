#include "dcp.h"
#include "svd_head.h"

DCPImpl::DCPImpl(int64_t embedding_dims, int64_t n_layers, int64_t n_heads, int64_t ff_dim, double dropout)
{
    feature = register_module("feature", DGCNN(embedding_dims));
    transformer = register_module("transformer", Transformer(embedding_dims, n_layers, n_heads, ff_dim, dropout));
}

std::tuple<torch::Tensor, torch::Tensor> DCPImpl::forward(torch::Tensor src, torch::Tensor tgt)
{
    torch::Tensor fx = feature->forward(src);
    torch::Tensor fy = feature->forward(tgt);

    auto [fx_p, fy_p] = transformer->forward(fx, fy);

    fx = fx + fx_p;
    fy = fy + fy_p;

    return svd(fx, fy, src, tgt);
}