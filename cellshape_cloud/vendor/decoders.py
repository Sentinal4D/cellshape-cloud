"""
some code (class FoldNetDecoder) vendored and adapted from:
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/3/23 5:39 PM
@License: cellshape_cloud/vendor/models/LICENSE_AnTao
#################################
classes FoldingModule and FoldingNetBasicDecoder were
vendored and adapted from:
https://github.com/InterDigitalInc/TearingNet/blob/master/models/foldingnet.py
Owner: https://www.interdigital.com/
"""
import torch
from torch import nn
import numpy as np
import itertools


class FoldingModule(nn.Module):
    def __init__(self):
        super(FoldingModule, self).__init__()

        self.folding1 = nn.Sequential(
            nn.Linear(512 + 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

        self.folding2 = nn.Sequential(
            nn.Linear(512 + 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

    def forward(self, x, grid):
        cw_exp = x.expand(-1, grid.shape[1], -1)

        cat1 = torch.cat((cw_exp, grid), dim=2)
        folding_result1 = self.folding1(cat1)
        cat2 = torch.cat((cw_exp, folding_result1), dim=2)
        folding_result2 = self.folding2(cat2)
        return folding_result2


class FoldingNetBasicDecoder(nn.Module):
    def __init__(
        self,
        num_features,
        shape="plane",
        sphere_path="./sphere.npy",
        gaussian_path="./gaussian.npy",
        std=0.3,
    ):
        super(FoldingNetBasicDecoder, self).__init__()

        # initialise deembedding
        self.lin_features_len = 512
        self.num_features = num_features
        if self.num_features < self.lin_features_len:
            self.deembedding = nn.Linear(
                self.num_features, self.lin_features_len, bias=False
            )

        if shape == "plane":
            # make grid
            range_x = torch.linspace(-std, std, 45)
            range_y = torch.linspace(-std, std, 45)
            x_coor, y_coor = torch.meshgrid(range_x, range_y, indexing="ij")
            self.grid = (
                torch.stack([x_coor, y_coor], axis=-1).float().reshape(-1, 2)
            )
        elif shape == "sphere":
            self.grid = torch.tensor(np.load(sphere_path))
        elif self.shape == "gaussian":
            self.grid = torch.tensor(np.load(gaussian_path))

        # initialise folding module
        self.folding = FoldingModule()

    def forward(self, x):
        if self.num_features < self.lin_features_len:
            x = self.deembedding(x)
            x = x.unsqueeze(1)

        else:
            x = x.unsqueeze(1)
        grid = self.grid.cuda().unsqueeze(0).expand(x.shape[0], -1, -1)
        outputs = self.folding(x, grid)
        return outputs


class FoldNetDecoder(nn.Module):
    def __init__(
        self,
        num_features,
        shape="plane",
        sphere_path="./sphere.npy",
        gaussian_path="./gaussian.npy",
        std=0.3,
    ):
        super(FoldNetDecoder, self).__init__()
        self.m = 2025  # 45 * 45.
        self.std = std
        self.meshgrid = [[-std, std, 45], [-std, std, 45]]
        self.shape = shape
        if shape == "sphere":
            self.sphere = np.load(sphere_path)
        if shape == "gaussian":
            self.gaussian = np.load(gaussian_path)
        self.num_features = num_features
        if self.shape == "plane":
            self.folding1 = nn.Sequential(
                nn.Conv1d(512 + 2, 512, 1),
                nn.ReLU(),
                nn.Conv1d(512, 512, 1),
                nn.ReLU(),
                nn.Conv1d(512, 3, 1),
            )
        else:
            self.folding1 = nn.Sequential(
                nn.Conv1d(512 + 3, 512, 1),
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
            self.deembedding = nn.Linear(
                self.num_features, self.lin_features_len, bias=False
            )

    def build_grid(self, batch_size):
        if self.shape == "plane":
            x = np.linspace(*self.meshgrid[0])
            y = np.linspace(*self.meshgrid[1])
            points = np.array(list(itertools.product(x, y)))
        elif self.shape == "sphere":
            points = self.sphere
        elif self.shape == "gaussian":
            points = self.gaussian

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
        output = folding_result2.transpose(1, 2)
        return output
