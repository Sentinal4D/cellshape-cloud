"""
Vendored and adapted from:
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/3/23 5:39 PM
@License: cellshape_cloud/vendor/models/LICENSE_AnTao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_functions import (
    get_graph_feature,
    local_maxpool,
    knn,
    local_cov,
)
from cellshape_cloud.helper_modules import Flatten


class DGCNNEncoder(nn.Module):
    def __init__(self, num_features, k=20):
        super(DGCNNEncoder, self).__init__()
        self.k = k
        self.num_features = num_features

        self.conv1 = nn.Sequential(
            nn.Conv2d(3 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.clustering = None
        self.lin_features_len = 512
        if (self.num_features < self.lin_features_len) or (
            self.num_features > self.lin_features_len
        ):
            self.flatten = Flatten()
            self.embedding = nn.Linear(
                self.lin_features_len, self.num_features, bias=False
            )

    def forward(self, x):
        # print(x.shape)
        # print(x)
        x = x.transpose(2, 1)

        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x0 = self.conv5(x)
        x = x0.max(dim=-1, keepdim=False)[0]
        feat = x.unsqueeze(1)

        if (self.num_features < self.lin_features_len) or (
            self.num_features > self.lin_features_len
        ):
            x = self.flatten(feat)
            features = self.embedding(x)
        else:
            features = torch.reshape(torch.squeeze(feat), (batch_size, 512))

        return features


class FoldNetEncoder(nn.Module):
    def __init__(self, num_features, k):
        super(FoldNetEncoder, self).__init__()
        if k is None:
            self.k = 16
        else:
            self.k = k
        self.n = 2048
        self.num_features = num_features
        self.mlp1 = nn.Sequential(
            nn.Conv1d(12, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(64, 64)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.linear2 = nn.Linear(128, 128)
        self.conv2 = nn.Conv1d(128, 1024, 1)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
        )
        self.clustering = None
        self.lin_features_len = 512
        if (self.num_features < self.lin_features_len) or (
            self.num_features > self.lin_features_len
        ):
            self.embedding = nn.Linear(
                self.lin_features_len, self.num_features, bias=False
            )

    def graph_layer(self, x, idx):
        x = local_maxpool(x, idx)
        x = self.linear1(x)
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        x = local_maxpool(x, idx)
        x = self.linear2(x)
        x = x.transpose(2, 1)
        x = self.conv2(x)
        return x

    def forward(self, pts):
        pts = pts.transpose(2, 1)
        batch_size = pts.size(0)
        idx = knn(pts, k=self.k)
        x = local_cov(pts, idx)
        x = self.mlp1(x)
        x = self.graph_layer(x, idx)
        x = torch.max(x, 2, keepdim=True)[0]
        x = self.mlp2(x)
        feat = x.transpose(2, 1)
        if (self.num_features < self.lin_features_len) or (
            self.num_features > self.lin_features_len
        ):
            x = self.flatten(feat)
            features = self.embedding(x)
        else:
            features = torch.reshape(torch.squeeze(feat), (batch_size, 512))

        return features


class DGCNN(nn.Module):
    def __init__(self, num_features=2, k=20, emb_dims=512):
        super(DGCNN, self).__init__()
        self.num_features = num_features
        self.k = k
        self.emb_dims = emb_dims
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)
        self.dropout = 0.2

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.linear1 = nn.Linear(self.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.dropout)
        self.linear3 = nn.Linear(256, num_features)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


if __name__ == "__main__":
    model = DGCNNEncoder(128)
    inp = torch.rand((1, 2048, 3))
    out = model(inp)
    print(out.shape)
