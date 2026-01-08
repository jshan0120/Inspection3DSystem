#include "loss.h"

torch::Tensor pose_loss( torch::Tensor R_pred, torch::Tensor t_pred, torch::Tensor R_gt, torch::Tensor t_gt)
{
    auto rot_loss = torch::norm(R_pred - R_gt);
    auto trans_loss = torch::norm(t_pred - t_gt);
    return rot_loss + trans_loss;
}