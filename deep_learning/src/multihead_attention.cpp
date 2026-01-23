#include "multihead_attention.h"

MultiHeadAttentionImpl::MultiHeadAttentionImpl(int64_t heads, int64_t model_dim)
{
    n_heads = heads;
    d_model = model_dim;
    d_k = model_dim / n_heads;
    register_module("linears", linears);
    linears->push_back(torch::nn::Linear(model_dim, model_dim));
    linears->push_back(torch::nn::Linear(model_dim, model_dim));
    linears->push_back(torch::nn::Linear(model_dim, model_dim));
    linears->push_back(torch::nn::Linear(model_dim, model_dim));
}

torch::Tensor MultiHeadAttentionImpl::forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    int64_t B = q.size(0);

    // q = q.squeeze(-1);
    // k = k.squeeze(-1);
    // v = v.squeeze(-1);

    int64_t seq_len = q.size(1);

    auto Q = linears->ptr<torch::nn::LinearImpl>(0)->forward(q)
                .view({B, seq_len, n_heads, d_k}).transpose(1, 2).contiguous();
    auto K = linears->ptr<torch::nn::LinearImpl>(1)->forward(k)
                .view({B, seq_len, n_heads, d_k}).transpose(1, 2).contiguous();
    auto V = linears->ptr<torch::nn::LinearImpl>(2)->forward(v)
                .view({B, seq_len, n_heads, d_k}).transpose(1, 2).contiguous();

    auto scores = torch::matmul(Q, K.transpose(-2,-1)) / std::sqrt((double)d_k);
    auto attn = torch::softmax(scores, -1);
    auto out = torch::matmul(attn, V);

    out = out.transpose(1,2).contiguous().view({B, -1, d_model});
    return linears->ptr<torch::nn::LinearImpl>(3)->forward(out);
}