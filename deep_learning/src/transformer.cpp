#include "transformer.h"

TransformerImpl::TransformerImpl(int64_t emb_dim, int64_t n_layers, int64_t n_heads, int64_t ff_dim, double dropout)
{
    encoder = register_module("encoder", Encoder(emb_dim, n_layers, n_heads, ff_dim, dropout));
    encoder->to(torch::kCUDA);

    decoder = register_module("decoder", Encoder(emb_dim, n_layers, n_heads, ff_dim, dropout));
    decoder->to(torch::kCUDA);
}

std::tuple<torch::Tensor, torch::Tensor> TransformerImpl::forward(torch::Tensor fx, torch::Tensor fy)
{
    torch::Tensor tgt_embedding = encoder->forward(fx);
    torch::Tensor src_embedding = decoder->forward(fy);

    return {src_embedding, tgt_embedding};
}