import torch
from torch import nn

import pytorch_lightning as pl


class CloudClassifierPL(pl.LightningModule):
    def __init__(self, args, model, criterion=nn.CrossEntropyLoss()):
        super(CloudClassifierPL, self).__init__()

        self.save_hyperparameters(ignore=["criterion", "model"])
        self.args = args
        self.lr = args.learning_rate_autoencoder
        self.criterion = criterion
        self.encoder_type = args.encoder_type
        self.model = model

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                betas=(0.9, 0.999),
                weight_decay=1e-6,
            )
            return optimizer

        def encode(self, x):
            z = self.model(x)
            return z
