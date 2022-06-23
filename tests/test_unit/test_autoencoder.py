import torch
from cellshape_cloud import CloudAutoEncoder


def test_autoencoder():
    model = CloudAutoEncoder(
        num_features=128, k=20, encoder_type="dgcnn", decoder_type="foldingnet"
    )

    points = torch.randn(1, 2048, 3)

    recon, features = model(points)
    expected_recon_shape = (1, 2045, 3)
    expected_features_shape = (1, 128)
    assert recon.shape == expected_recon_shape
    assert features.shape == expected_features_shape
