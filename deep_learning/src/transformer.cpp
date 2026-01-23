#include "transformer.h"

TransformerImpl::TransformerImpl(int64_t emb_dim, int64_t n_layers, int64_t n_heads, int64_t ff_dim, double dropout)
{
    // torch::nn::Sequential model_holder;
    // register_module("model", model_holder);

    // encoder = Encoder(emb_dim, n_layers, n_heads, ff_dim, dropout);
    // decoder = Encoder(emb_dim, n_layers, n_heads, ff_dim, dropout);

    // model_holder->push_back("encoder", encoder);
    // model_holder->push_back("decoder", decoder);

    // this->to(torch::kCUDA);

    encoder = register_module("encoder", Encoder(emb_dim, n_layers, n_heads, ff_dim, dropout));
    encoder->to(torch::kCUDA);

    decoder = register_module("decoder", Encoder(emb_dim, n_layers, n_heads, ff_dim, dropout));
    decoder->to(torch::kCUDA);
}

std::tuple<torch::Tensor, torch::Tensor> TransformerImpl::forward(torch::Tensor fx, torch::Tensor fy)
{
    torch::Tensor src_embedding = decoder->forward(fx);
    torch::Tensor tgt_embedding = encoder->forward(fy);

    return {src_embedding, tgt_embedding};
}