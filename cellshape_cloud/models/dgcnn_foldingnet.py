import torch
import torch.nn as nn

from ..helpers.graph_functions import get_graph_feature
from ..helpers.helper_modules import Flatten
from ..losses.chamfer_loss import ChamferLoss
from .foldingnet import FoldNetDecoder


class DGCNNEncoder(nn.Module):
    def __init__(self, num_features):
        super(DGCNNEncoder, self).__init__()
        self.k = 20
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

        self.lin_features_len = 512
        if self.num_features < self.lin_features_len:
            self.flatten = Flatten()
            self.embedding = nn.Linear(
                self.lin_features_len, self.num_features, bias=False
            )
            self.deembedding = nn.Linear(
                self.num_features, self.lin_features_len, bias=False
            )

    def forward(self, x):
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

        if (
            self.num_features < self.lin_features_len
            or self.num_features > self.lin_features_len
        ):
            x = self.flatten(feat)
            embedding = self.embedding(x)
        else:
            embedding = torch.reshape(torch.squeeze(feat), (batch_size, 512))

        return feat, embedding


class DGCNNFoldingNet(nn.Module):
    def __init__(self, num_features):
        super(DGCNNFoldingNet, self).__init__()
        self.num_features = num_features
        self.encoder = DGCNNEncoder(num_features)
        self.decoder = FoldNetDecoder(num_features)
        self.loss = ChamferLoss()

    def forward(self, input):
        feature, embedding = self.encoder(input)
        output, fold1 = self.decoder(embedding)
        return output, feature, embedding, fold1

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(
            self.decoder.parameters()
        )

    def get_loss(self, input, output):
        return self.loss(input, output)
