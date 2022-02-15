from torch import nn

from ._vendor.encoders import FoldNetEncoder, DGCNNEncoder
from ._vendor.decoders import FoldNetDecoder


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
        self.decoder = FoldNetDecoder(num_features=self.num_features)

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(x=features)
        return output, features
