#include "svd_head.h"

std::tuple<torch::Tensor, torch::Tensor> svd(torch::Tensor fx, torch::Tensor fy, torch::Tensor src, torch::Tensor tgt)
{
    torch::Tensor correspondence = torch::matmul(fx, fy.transpose(1, 2));

    correspondence = correspondence / std::sqrt((double)fx.size(-1));

    torch::Tensor w = torch::softmax(correspondence, -1);

    torch::Tensor src_mean = torch::mean(src, 1, true);
    torch::Tensor tgt_mean = torch::mean(tgt, 1, true);

    torch::Tensor X = src - src_mean;
    torch::Tensor Y = tgt - tgt_mean;

    torch::Tensor H = torch::matmul(
        X.transpose(1,2),
        torch::matmul(w, Y)
    );

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> svd = torch::linalg_svd(H);
    torch::Tensor U = std::get<0>(svd);
    torch::Tensor V = std::get<2>(svd);

    torch::Tensor R = torch::matmul(V, U.transpose(1, 2));

    torch::Tensor detR = torch::det(R);
    torch::Tensor mask = detR.lt(0).to(R.dtype());

    torch::Tensor V_fix = V.clone();
    V_fix.index_put_(
        {torch::indexing::Slice(), torch::indexing::Slice(), 2},
        V_fix.index(
            {torch::indexing::Slice(), torch::indexing::Slice(), 2}
        ) * (1 - 2 * mask).unsqueeze(1)
    );

    R = torch::matmul(V_fix, U.transpose(1, 2));

    torch::Tensor t =
        tgt_mean.squeeze(1)
        - torch::matmul(
              R,
              src_mean.squeeze(1).unsqueeze(-1)
          ).squeeze(-1);

    return std::make_pair(R, t);
}