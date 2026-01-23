#include "svd_head.h"
#include <cmath>

SVDHeadImpl::SVDHeadImpl(int64_t emb_dims) {
    torch::NoGradGuard no_grad;
    reflect = register_parameter("reflect", torch::eye(3));
    reflect[2][2] = -1.0;
    reflect.set_requires_grad(false);
}

std::tuple<torch::Tensor, torch::Tensor> SVDHeadImpl::forward(
    torch::Tensor src_embedding, torch::Tensor tgt_embedding, 
    torch::Tensor src, torch::Tensor tgt) 
{
    int64_t batch_size = src.size(0);
    int64_t d_k = src_embedding.size(1);

    torch::Tensor scores = torch::matmul(src_embedding.transpose(2, 1), tgt_embedding);
    scores = scores / std::sqrt((double)d_k);
    scores = torch::softmax(scores, 2);

    torch::Tensor src_corr = torch::matmul(tgt, scores.transpose(1, 2));

    torch::Tensor src_mean = src.mean(2, true);
    torch::Tensor src_corr_mean = src_corr.mean(2, true);

    torch::Tensor src_centered = src - src_mean;
    torch::Tensor src_corr_centered = src_corr - src_corr_mean;

    torch::Tensor H = torch::matmul(src_centered, src_corr_centered.transpose(2, 1));

    std::vector<torch::Tensor> R_list;
    for (int i = 0; i < batch_size; ++i) {
        auto [u, s, vh] = torch::linalg_svd(H[i]);
        torch::Tensor v = vh.transpose(-2, -1).conj();
        
        torch::Tensor r = torch::matmul(v, u.transpose(-2, -1));

        if (torch::det(r).item<float>() < 0) {
            r = torch::matmul(torch::matmul(v, reflect.to(v.device())), u.transpose(-2, -1));
        }
        R_list.push_back(r);
    }
    
    torch::Tensor R = torch::stack(R_list, 0);
    torch::Tensor t = torch::matmul(-R, src_mean) + src_corr_mean;

    return std::make_tuple(R, t.squeeze(2));
}