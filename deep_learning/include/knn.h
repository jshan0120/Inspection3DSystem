#pragma once

#ifndef KNN_H
#define KNN_H

#include <torch/torch.h>

torch::Tensor knn(torch::Tensor x, int64_t k)
{
    int64_t B = x.size(0);
    int64_t N = x.size(2);

    torch::Tensor xx = torch::sum(x * x, 1, true);
    torch::Tensor yy = xx.transpose(1, 2);
    torch::Tensor xy = torch::bmm(x.transpose(1, 2), x);
    torch::Tensor dist = xx + yy - 2 * xy;

    torch::Tensor neg_dist = -dist;
    torch::Tensor idx = std::get<1>(torch::topk(neg_dist, k, -1, true, true));
    return idx;
}

torch::Tensor index_points(torch::Tensor points, torch::Tensor idx)
{
    int64_t B = points.size(0);
    int64_t C = points.size(1);
    int64_t N = points.size(2);
    int64_t k = idx.size(2);

    torch::Tensor idx_exp = idx.unsqueeze(1).expand({B, C, N, k});

    torch::Tensor points_exp = points.unsqueeze(-1).expand({B, C, N, N});

    torch::Tensor neighbors = points_exp.gather(3, idx_exp);
    return neighbors;
}

torch::Tensor get_graph_feature(torch::Tensor x, int64_t k = 20)
{
    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t N = x.size(2);

    if (k > N) k = N;

    torch::Tensor idx = knn(x, k);
    torch::Tensor neighbors = index_points(x, idx);

    torch::Tensor x_expanded = x.unsqueeze(-1).expand({B, C, N, k});

    torch::Tensor edge_feature = torch::cat({neighbors - x_expanded, x_expanded}, 1);
    return edge_feature;
}

#endif