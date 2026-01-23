#include "dcp.h"

DCPImpl::DCPImpl(int64_t embedding_dims, int64_t n_layers, int64_t n_heads, int64_t ff_dim, double dropout)
{
    emb_nn = register_module("emb_nn", DGCNN(embedding_dims));
    pointer = register_module("pointer", Transformer(embedding_dims, n_layers, n_heads, ff_dim, dropout));
    head = register_module("head", SVDHead(embedding_dims));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> DCPImpl::forward(torch::Tensor src, torch::Tensor tgt)
{
    torch::Tensor src_embedding = emb_nn->forward(src);
    torch::Tensor tgt_embedding = emb_nn->forward(tgt);

    auto [src_embedding_p, tgt_embedding_p] = pointer->forward(
        src_embedding.transpose(1, 2).contiguous(), 
        tgt_embedding.transpose(1, 2).contiguous()
    );

    src_embedding = src_embedding + src_embedding_p.transpose(1, 2).contiguous();
    tgt_embedding = tgt_embedding + tgt_embedding_p.transpose(1, 2).contiguous();

    auto [rotation_ab, translation_ab] = head->forward(src_embedding, tgt_embedding, src, tgt);

    torch::Tensor rotation_ba = rotation_ab.transpose(2, 1).contiguous();
    torch::Tensor translation_ba = -torch::bmm(rotation_ba, translation_ab.unsqueeze(2)).squeeze(2);

    return std::make_tuple(rotation_ab, translation_ab, rotation_ba, translation_ba);
}