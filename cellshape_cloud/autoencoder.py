from torch import nn

from vendored.encoders import FoldNetEncoder, DGCNNEncoder
from vendored.decoders import FoldNetDecoder


class GraphAutoEncoder(nn.Module):
    def __init__(self, num_features, k=20, encoder="dgcnn"):
        super(GraphAutoEncoder, self).__init__()
        self.k = k
        self.num_features = num_features
        if encoder == "dgcnn":
            self.encoder = DGCNNEncoder(
                num_features=self.num_features, k=self.k
            )
        else:
            self.encoder = FoldNetEncoder(
                num_features=self.num_features, k=self.k
            )
        self.decoder = FoldNetDecoder

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output, features
