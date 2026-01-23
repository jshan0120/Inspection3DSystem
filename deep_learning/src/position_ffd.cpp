#include "position_ffd.h"

PositionwiseFeedForwardImpl::PositionwiseFeedForwardImpl(int64_t d_model, int64_t d_ff, double drop)
{
    w_1 = register_module("w_1", torch::nn::Linear(d_model, d_ff));
    w_2 = register_module("w_2", torch::nn::Linear(d_ff, d_model));
    norm = register_module("norm", torch::nn::Sequential());
}

torch::Tensor PositionwiseFeedForwardImpl::forward(torch::Tensor x)
{
    x = torch::relu(w_1->forward(x));
    return w_2->forward(x);
}