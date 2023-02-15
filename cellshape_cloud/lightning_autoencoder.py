import torch

import pytorch_lightning as pl

from .vendor.chamfer_distance import ChamferLoss


class CloudAutoEncoderPL(pl.LightningModule):
    def __init__(self, args, model, criterion=ChamferLoss()):
        super(CloudAutoEncoderPL, self).__init__()

        self.save_hyperparameters(ignore=["criterion", "model"])
        self.args = args
        self.lr = args.learning_rate_autoencoder
        self.criterion = criterion
        self.encoder_type = args.encoder_type
        self.decoder_type = args.decoder_type
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
        z = self.model.encoder(x)
        return z

    def decode(self, z):
        out = self.model.decoder(z)
        return out

    def load_model(self, path):
        checkpoint = torch.load(path, map_location="cuda:0")
        model_dict = (
            self.model.state_dict()
        )  # load parameters from pre-trained FoldingNet
        for k in checkpoint["model_state_dict"]:
            print(k)

        for k in model_dict:
            print(k)

        for k in checkpoint["model_state_dict"]:
            # print(k)
            if k in model_dict:
                model_dict[k] = checkpoint["model_state_dict"][k]
                print("    Found weight: " + k)
            elif k.replace("folding1", "folding") in model_dict:
                model_dict[k.replace("folding1", "folding")] = checkpoint[
                    "model_state_dict"
                ][k]
                print("    Found weight: " + k)
        print("Done loading autoencoder")

    def load_model_foldingnet(self, path):
        checkpoint = torch.load(path, map_location="cuda:0")
        model_dict = (
            self.model.state_dict()
        )  # load parameters from pre-trained FoldingNet
        for k in checkpoint["model_state_dict"]:
            print(k)

        for k in model_dict:
            print(k)

        for k in checkpoint["model_state_dict"]:
            # print(k)
            if k in model_dict:
                model_dict[k] = checkpoint["model_state_dict"][k]
                print("    Found weight: " + k)
            elif k.replace("folding.", "") in model_dict:
                model_dict[k.replace("folding.", "")] = checkpoint[
                    "model_state_dict"
                ][k]
                print("    Found weight: " + k)
        print("Done loading autoencoder")

        self.model.load_state_dict(model_dict)

    def load_lightning(self, path):
        checkpoint = torch.load(
            path, map_location=lambda storage, loc: storage
        )
        self.load_state_dict(checkpoint["state_dict"])

    def load_shapenet(self, path):
        checkpoint = torch.load(
            path, map_location=lambda storage, loc: storage
        )
        # "load encoder"
        model_dict = self.model.state_dict()
        for k in checkpoint:
            if k in model_dict:
                model_dict[k] = checkpoint[k]
                print("    Found weight: " + k)
            elif k.replace("encoder.", "model.encoder.") in model_dict:
                model_dict[
                    k.replace("encoder.", "model.encoder.")
                ] = checkpoint[k]
                print("    Found weight: " + k)
            elif k.replace("decoder.", "model.decoder.") in model_dict:
                model_dict[
                    k.replace("decoder.", "model.decoder.")
                ] = checkpoint[k]
                print("    Found weight: " + k)

        self.model.load_state_dict(model_dict)

    def training_step(self, batch, batch_idx):
        inputs = batch[0]
        outputs, features = self.model(inputs)

        loss = self.criterion(inputs, outputs)
        self.log("loss", loss)

        return loss
