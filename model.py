import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import itertools
from tearingnet import *


class Flatten(nn.Module):

    def forward(self, input):
        '''
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0) # of batches,
        will flatten to be 1 * nb_elements.
        '''
        batch_size = input.size(0)
        out = input.view(batch_size, -1)
        return out  # (batch_size, *size)


def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    else:
        idx_base = torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    return idx


def local_cov(pts, idx):
    batch_size = pts.size(0)
    num_points = pts.size(2)
    pts = pts.view(batch_size, -1, num_points)  # (batch_size, 3, num_points)

    _, num_dims, _ = pts.size()

    x = pts.transpose(2, 1).contiguous()  # (batch_size, num_points, 3)
    x = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size*num_points*2, 3)
    x = x.view(batch_size, num_points, -1, num_dims)  # (batch_size, num_points, k, 3)

    x = torch.matmul(x[:, :, 0].unsqueeze(3), x[:, :, 1].unsqueeze(
        2))  # (batch_size, num_points, 3, 1) * (batch_size, num_points, 1, 3) -> (batch_size, num_points, 3, 3)
    # x = torch.matmul(x[:,:,1:].transpose(3, 2), x[:,:,1:])
    x = x.view(batch_size, num_points, 9).transpose(2, 1)  # (batch_size, 9, num_points)

    x = torch.cat((pts, x), dim=1)  # (batch_size, 12, num_points)

    return x


def local_maxpool(x, idx):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    x = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    x = x.view(batch_size, num_points, -1, num_dims)  # (batch_size, num_points, k, num_dims)
    x, _ = torch.max(x, dim=2)  # (batch_size, num_points, num_dims)

    return x


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)  # (batch_size, num_dims, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)  # (batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (batch_size, num_points, k, num_dims)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1,
                                                         2)  # (batch_size, num_points, k, 2*num_dims) -> (batch_size, 2*num_dims, num_points, k)

    return feature  # (batch_size, 2*num_dims, num_points, k)


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x)
        diag_ind_y = torch.arange(0, num_points_y)
        if x.get_device() != -1:
            diag_ind_x = diag_ind_x.cuda(x.get_device())
            diag_ind_y = diag_ind_y.cuda(x.get_device())
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)
        return loss_1 + loss_2


# Clustering layer definition (see DCEC article for equations)
class ClusterlingLayer(nn.Module):
    def __init__(self, in_features=10, out_features=10, alpha=1.0):
        super(ClusterlingLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        # represent weight as module parameter
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.unsqueeze(1) - self.weight
        x = torch.mul(x, x)
        x = torch.sum(x, dim=2)
        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha + 1.0) / 2.0)
        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, alpha={}'.format(
            self.in_features, self.out_features, self.alpha
        )

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)


class DGCNN_Cls_Encoder(nn.Module):
    def __init__(self, num_clusters, num_features):
        super(DGCNN_Cls_Encoder, self).__init__()
        self.k = 20
        self.task = 'reconstruct'
        self.num_clusters = num_clusters
        self.num_features = num_features
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(512)

        self.conv1 = nn.Sequential(nn.Conv2d(3 * 2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.lin_features_len = 512
        if self.num_features < self.lin_features_len:
            self.flatten = Flatten()
            self.embedding = nn.Linear(self.lin_features_len, self.num_features, bias=False)
            self.deembedding = nn.Linear(self.num_features, self.lin_features_len, bias=False)

        self.clustering = ClusterlingLayer(self.num_features, self.num_clusters)

    def forward(self, x):
        x = x.transpose(2, 1)

        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 512, num_points)

        x0 = self.conv5(x)  # (batch_size, 512, num_points) -> (batch_size, feat_dims, num_points)
        x = x0.max(dim=-1, keepdim=False)[0]  # (batch_size, feat_dims, num_points) -> (batch_size, feat_dims)
        feat = x.unsqueeze(1)  # (batch_size, feat_dims) -> (batch_size, 1, feat_dims)
        # feat = x
        # print(feat.shape)
        if self.num_features < self.lin_features_len:
            x = self.flatten(feat)
            embedding = self.embedding(x)
        #         print(embedding.shape)
        #         print(embedding.shape)
        #         print(feat.shape)
        #         print(torch.squeeze(feat).shape)
        if self.num_features == self.lin_features_len:
            embedding = torch.reshape(torch.squeeze(feat), (batch_size, 512))

        #         clustering_input = torch.reshape(torch.squeeze(feat),(batch_size, 512))
        clustering_input = embedding
        clustering_out = self.clustering(clustering_input)
        #         print(clustering_out.shape)

        if self.task == 'classify':
            return feat, x0
        elif self.task == 'reconstruct':
            return feat, embedding, clustering_out  # (batch_size, 1, feat_dims)


class FoldNet_Decoder(nn.Module):
    def __init__(self, num_clusters, num_features, shape):
        super(FoldNet_Decoder, self).__init__()
        self.m = 2025  # 45 * 45.
        self.shape = shape
        self.meshgrid = [[-3, 3, 45], [-3, 3, 45]]
        self.sphere = np.load("sphere.npy")
        self.gaussian = np.load("gaussian100.npy")
        self.num_features = num_features
        if self.shape == 'plane':
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
            self.embedding = nn.Linear(self.lin_features_len, num_clusters, bias=False)
            self.deembedding = nn.Linear(self.num_features, self.lin_features_len, bias=False)

    def build_grid(self, batch_size):
        if self.shape == 'sphere':
            points = self.sphere
        elif self.shape == 'gaussian':
            points = self.gaussian
        else:
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
        x = x.transpose(1, 2).repeat(1, 1, self.m)  # (batch_size, feat_dims, num_points)
        points = self.build_grid(x.shape[0]).transpose(1,
                                                       2)  # (batch_size, 2, num_points) or (batch_size, 3, num_points)
        if x.get_device() != -1:
            points = points.cuda(x.get_device())
        cat1 = torch.cat((x, points),
                         dim=1)  # (batch_size, feat_dims+2, num_points) or (batch_size, feat_dims+3, num_points)
        #         print(cat1.size)
        folding_result1 = self.folding1(cat1)  # (batch_size, 3, num_points)
        cat2 = torch.cat((x, folding_result1), dim=1)  # (batch_size, 515, num_points)
        folding_result2 = self.folding2(cat2)  # (batch_size, 3, num_points)
        return folding_result2.transpose(1, 2), folding_result1.transpose(1, 2)  # (batch_size, num_points ,3)


class DGCNN_Cls_Classifier(nn.Module):
    def __init__(self):
        super(DGCNN_Cls_Classifier, self).__init__()

        output_channels = 10

        self.linear1 = nn.Linear(512 * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class ReconstructionNet(nn.Module):
    def __init__(self, decoder, num_clusters, num_features, shape):
        super(ReconstructionNet, self).__init__()
        self.num_cluster = num_clusters
        self.num_features = num_features
        self.encoder = DGCNN_Cls_Encoder(num_clusters, num_features)
        self.decoder_decision = decoder
        if decoder == 'tearing':
            self.decoder = TearingNetGraphModel()
        else:
            self.decoder = FoldNet_Decoder(num_clusters, num_features, shape)
        self.loss = ChamferLoss()

    def forward(self, input):
        feature, embedding, clustering_out = self.encoder(input)
        if self.decoder_decision == 'tearing':
            pc0, pc1, pc2, grid1, graph_wght = self.decoder(embedding)
            return pc2, feature, embedding, clustering_out, (pc0, pc1, grid1, graph_wght)
        else:
            output, fold1 = self.decoder(embedding)
            return output, feature, embedding, clustering_out, fold1

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def get_loss(self, input, output):
        # input shape  (batch_size, 2048, 3)
        # output shape (batch_size, 2025, 3)
        return self.loss(input, output)
