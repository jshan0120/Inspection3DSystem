#pragma once

#ifndef ENCODER_H
#define ENCODER_H

#include <torch/torch.h>
#include "multihead_attention.h"
#include "position_ffd.h"

struct EncoderLayerImpl : torch::nn::Module
{
    MultiHeadAttention self_attn{nullptr};
    PositionwiseFeedForward ff{nullptr};

    EncoderLayerImpl(MultiHeadAttention attn_module, PositionwiseFeedForward ff_module): self_attn(std::move(attn_module)), ff(std::move(ff_module))
    {
        register_module("self_attn", self_attn);
        register_module("ff", ff);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = x + self_attn->forward(x, x, x);
        x = x + ff->forward(x);
        return x;
    }
};
TORCH_MODULE(EncoderLayer);

struct EncoderImpl : torch::nn::Module
{
    torch::nn::ModuleList layers;

    EncoderImpl(int64_t d_model, int64_t num_layers, int64_t n_heads, int64_t d_ff, double dropout)
    {
        for(int64_t i = 0; i < num_layers; i++)
        {
            MultiHeadAttention attn = MultiHeadAttention(n_heads, d_model);
            PositionwiseFeedForward ff_module = PositionwiseFeedForward(d_model, d_ff, dropout);
            layers->push_back(register_module("layer" + std::to_string(i), EncoderLayer(attn, ff_module)));
        }
    }

    torch::Tensor forward(torch::Tensor x)
    {
        for (auto& layer : *layers)
        {
            x = layer->as<EncoderLayer>()->forward(x);
        }
        return x;
    }
};
TORCH_MODULE(Encoder);

#endif