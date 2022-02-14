"""
Vendored and adapted from:
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/3/23 5:39 PM
@License: cellshape_cloud/vendored/models/LICENSE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from chamferdist import ChamferDistance

from cellshape_cloud.vendored.graph_functions import local_maxpool, knn, local_cov


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

        self.lin_features_len = 512
        if self.num_features < self.lin_features_len:
            self.embedding = nn.Linear(
                self.lin_features_len, self.num_features, bias=False
            )
            self.deembedding = nn.Linear(
                self.num_features, self.lin_features_len, bias=False
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
        idx = knn(pts, k=self.k)
        x = local_cov(pts, idx)
        x = self.mlp1(x)
        x = self.graph_layer(x, idx)
        x = torch.max(x, 2, keepdim=True)[0]
        x = self.mlp2(x)
        feat = x.transpose(2, 1)
        return feat


class FoldNetDecoder(nn.Module):
    def __init__(self, num_features):
        super(FoldNetDecoder, self).__init__()
        self.m = 2025  # 45 * 45.
        self.meshgrid = [[-3, 3, 45], [-3, 3, 45]]
        self.num_features = num_features
        self.folding1 = nn.Sequential(
            nn.Conv1d(512 + 2, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(512 + 3, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1),
        )

        self.lin_features_len = 512
        if self.num_features < self.lin_features_len:
            self.embedding = nn.Linear(
                self.lin_features_len, self.num_features, bias=False
            )
            self.deembedding = nn.Linear(
                self.num_features, self.lin_features_len, bias=False
            )

    def build_grid(self, batch_size):
        x = np.linspace(*self.meshgrid[0])
        y = np.linspace(*self.meshgrid[1])
        points = np.array(list(itertools.product(x, y)))
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points)
        return points.float()

    def forward(self, x):

        if self.num_features < self.lin_features_len:
            x = self.deembedding(x)
            x = x.unsqueeze(1)

        else:
            x = x.unsqueeze(1)
        x = x.transpose(1, 2).repeat(1, 1, self.m)
        points = self.build_grid(x.shape[0]).transpose(1, 2)
        if x.get_device() != -1:
            points = points.cuda(x.get_device())
        cat1 = torch.cat((x, points), dim=1)

        folding_result1 = self.folding1(cat1)
        cat2 = torch.cat((x, folding_result1), dim=1)
        folding_result2 = self.folding2(cat2)
        return folding_result2.transpose(1, 2), folding_result1.transpose(1, 2)


class FoldingNet(nn.Module):
    def __init__(self, num_features, k):
        super(FoldingNet, self).__init__()
        self.num_features = num_features
        self.k = k
        self.encoder = FoldNetEncoder(num_features, k)
        self.decoder = FoldNetDecoder(num_features)
        self.loss = ChamferDistance()

    def forward(self, input):
        feature, embedding, clustering_out = self.encoder(input)
        output, fold1 = self.decoder(embedding)
        return output, feature, embedding, clustering_out, fold1

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(
            self.decoder.parameters()
        )

    def get_loss(self, input, output):
        return self.loss(input, output, bidirectional=True)
