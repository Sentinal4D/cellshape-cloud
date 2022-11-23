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
        shape="plane",
        sphere_path="./sphere.npy",
        gaussian_path="./gaussian.npy",
        std=0.3,
    ):
        super(CloudAutoEncoder, self).__init__()
        self.k = k
        self.num_features = num_features
        self.shape = shape
        self.sphere_path = sphere_path
        self.gaussian_path = gaussian_path
        self.std = std

        assert encoder_type.lower() in [
            "foldingnet",
            "dgcnn",
            "dgcnn_orig",
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
        # elif self.encoder_type == "dgcnn_orig":
        #     self.encoder = DGCNN(num_features=self.num_features, k=self.k)
        else:
            self.encoder = FoldNetEncoder(
                num_features=self.num_features, k=self.k
            )

        if self.decoder_type == "foldingnet":
            self.decoder = FoldNetDecoder(
                num_features=self.num_features,
                shape=self.shape,
                sphere_path=self.sphere_path,
                gaussian_path=self.gaussian_path,
                std=self.std,
            )
        else:
            self.decoder = FoldingNetBasicDecoder(
                num_features=self.num_features,
                shape=self.shape,
                sphere_path=self.sphere_path,
                gaussian_path=self.gaussian_path,
                std=self.std,
            )

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(x=features)
        return output, features
