#include "position_ffd.h"

PositionwiseFeedForwardImpl::PositionwiseFeedForwardImpl(int64_t d_model, int64_t d_ff, double drop)
{
    fc1 = register_module("fc1", torch::nn::Linear(d_model, d_ff));
    fc2 = register_module("fc2", torch::nn::Linear(d_ff, d_model));
}

torch::Tensor PositionwiseFeedForwardImpl::forward(torch::Tensor x)
{
    x = torch::relu(fc1->forward(x));
    return fc2->forward(x);
}