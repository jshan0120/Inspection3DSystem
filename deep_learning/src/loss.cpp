#include "loss.h"

torch::Tensor pose_loss(torch::Tensor R_pred, torch::Tensor t_pred, 
                        torch::Tensor R_gt, torch::Tensor t_gt, 
                        torch::Tensor src)
{
    // auto src_transformed_pred = torch::matmul(R_pred, src) + t_pred.unsqueeze(2);
    // auto src_transformed_gt = torch::matmul(R_gt, src) + t_gt.unsqueeze(2);

    auto identity = torch::eye(3, R_pred.options()).unsqueeze(0).repeat({R_pred.size(0), 1, 1});
    auto mat_diff = torch::matmul(R_pred.transpose(1, 2), R_gt);
    auto rot_loss = torch::mse_loss(mat_diff, identity);

    auto trans_loss = torch::mse_loss(t_pred, t_gt);

    // auto loss = torch::mse_loss(src_transformed_pred, src_transformed_gt);

    return rot_loss + trans_loss;
}