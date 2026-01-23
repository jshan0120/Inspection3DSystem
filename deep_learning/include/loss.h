#pragma once

#ifndef LOSS_H
#define LOSS_H

#include <torch/torch.h>

torch::Tensor pose_loss(torch::Tensor R_pred, torch::Tensor t_pred, 
                        torch::Tensor R_gt, torch::Tensor t_gt, 
                        torch::Tensor src);

#endif