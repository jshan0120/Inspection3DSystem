#pragma once

#ifndef SVD_H
#define SVD_H

#include <torch/torch.h>

std::tuple<torch::Tensor, torch::Tensor> svd(torch::Tensor fx, torch::Tensor fy, torch::Tensor src, torch::Tensor tgt);

#endif