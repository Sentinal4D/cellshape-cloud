from torch import nn

from .vendor.encoders import FoldNetEncoder, DGCNNEncoder
from .vendor.decoders import FoldNetDecoder, FoldingNetBasicDecoder


class CloudAutoEncoder(nn.Module):
    def __init__(
        self,
        num_features,
        k=20,
        encoder_type="dgcnn",
        decoder_type="foldingnet",
    ):
        super(CloudAutoEncoder, self).__init__()
        self.k = k
        self.num_features = num_features
        assert encoder_type.lower() in [
            "foldingnet",
            "dgcnn",
        ], "Please select an encoder type from either foldingnet or dgcnn."

        assert decoder_type.lower() in [
            "foldingnet",
            "foldingnetbasic",
        ], "Please select an decoder type from either foldingnet."

        self.encoder_type = encoder_type.lower()
        self.decoder_type = decoder_type.lower()
        if self.encoder_type == "dgcnn":
            self.encoder = DGCNNEncoder(
                num_features=self.num_features, k=self.k
            )
        else:
            self.encoder = FoldNetEncoder(
                num_features=self.num_features, k=self.k
            )

        if self.decoder_type == "foldingnet":
            self.decoder = FoldNetDecoder(num_features=self.num_features)
        else:
            self.decoder = FoldingNetBasicDecoder(
                num_features=self.num_features
            )

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(x=features)
        return output, features
