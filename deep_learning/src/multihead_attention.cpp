#include "multihead_attention.h"

MultiHeadAttentionImpl::MultiHeadAttentionImpl(int64_t heads, int64_t model_dim)
{
    n_heads = heads;
    d_model = model_dim;
    d_k = model_dim / n_heads;
    w_q = register_module("w_q", torch::nn::Linear(model_dim, model_dim));
    w_k = register_module("w_k", torch::nn::Linear(model_dim, model_dim));
    w_v = register_module("w_v", torch::nn::Linear(model_dim, model_dim));
    w_o = register_module("w_o", torch::nn::Linear(model_dim, model_dim));
}

torch::Tensor MultiHeadAttentionImpl::forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    int64_t B = q.size(0);

    q = q.squeeze(-1);
    k = k.squeeze(-1);
    v = v.squeeze(-1);

    int64_t seq_len = q.size(1);

    auto Q = w_q->forward(q).view({B, seq_len, n_heads, d_k}).transpose(1,2);
    auto K = w_k->forward(k).view({B, seq_len, n_heads, d_k}).transpose(1,2);
    auto V = w_v->forward(v).view({B, seq_len, n_heads, d_k}).transpose(1,2);

    auto scores = torch::matmul(Q, K.transpose(-2,-1)) / std::sqrt((double)d_k);
    auto attn = torch::softmax(scores, -1);
    auto out = torch::matmul(attn, V);

    out = out.transpose(1,2).contiguous().view({B, -1, d_model});
    return w_o->forward(out);
}