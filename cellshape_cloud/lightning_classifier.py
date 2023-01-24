import torch
from torch import nn

import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC
from pointcloud_dataset import VesselMNIST3D
from torch.utils.data import DataLoader

from vendor.encoders import DGCNNEncoder


class VesselDataModule(pl.LightningDataModule):
    """Cassava DataModule for Lightning"""

    def __init__(
        self,
        transform=None,
        batch_size=32,
        points_dir="/home/mvries/Documents/Datasets/MedMNIST/vesselmnist3d/",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform
        self.points_dir = points_dir

    def setup(self, stage=None):
        self.vessel_train = VesselMNIST3D(
            self.points_dir, centre=True, scale=20.0, partition="train"
        )
        self.vessel_val = VesselMNIST3D(
            self.points_dir, centre=True, scale=20.0, partition="val"
        )
        self.vessel_test = VesselMNIST3D(
            self.points_dir, centre=True, scale=20.0, partition="test"
        )

    def train_dataloader(self):
        return DataLoader(self.vessel_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.vessel_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.vessel_test, batch_size=self.batch_size)


class CloudClassifierPL(pl.LightningModule):
    def __init__(self, criterion=nn.BCEWithLogitsLoss(), num_classes=2):
        super(CloudClassifierPL, self).__init__()

        self.save_hyperparameters(ignore=["criterion", "model"])
        self.lr = 0.00001
        self.criterion = criterion
        self.model = DGCNNEncoder(num_features=1)
        self.num_classes = num_classes
        self.accuracy_macro = Accuracy(
            task="binary", num_classes=2, average="macro"
        )
        self.accuracy_micro = Accuracy(
            task="binary", num_classes=2, average="micro"
        )
        self.accuracy_weighted = Accuracy(
            task="binary", num_classes=2, average="weighted"
        )
        self.AUC = AUROC(task="binary")

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

    def load_model(self, path):
        checkpoint = torch.load(path, map_location="cuda:0")
        model_dict = (
            self.model.state_dict()
        )  # load parameters from pre-trained FoldingNet

        for k in checkpoint["model_state_dict"]:
            if k in model_dict:
                model_dict[k] = checkpoint["model_state_dict"][k]
                print("    Found weight: " + k)
            elif k.replace("folding1", "folding") in model_dict:
                model_dict[k.replace("folding1", "folding")] = checkpoint[
                    "model_state_dict"
                ][k]
                print("    Found weight: " + k)
        print("Done loading encoder")

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
        inputs, labels = batch[0], batch[1]
        outputs = self.model(inputs)

        loss = self.criterion(outputs, torch.unsqueeze(labels, 1).float())
        preds = torch.sigmoid(outputs) > 0.5
        acc = self.accuracy(torch.squeeze(preds), labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log(
            "train_acc",
            acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        dic = {"loss": loss, "acc": acc}
        return dic

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch[0], batch[1]
        outputs = self.model(inputs)

        loss = self.criterion(outputs, torch.unsqueeze(labels, 1).float())
        preds = torch.argmax(outputs, dim=1)
        acc = self.accuracy(preds, labels)
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch[0], batch[1]
        outputs = self.model(inputs)

        loss = self.criterion(outputs, torch.unsqueeze(labels, 1).float())
        preds = torch.sigmoid(outputs) > 0.5
        print(preds)

        acc_macro = self.accuracy_macro(torch.squeeze(preds), labels)
        acc_micro = self.accuracy_micro(torch.squeeze(preds), labels)
        acc_weighted = self.accuracy_weighted(torch.squeeze(preds), labels)
        auc = self.AUC(torch.squeeze(torch.sigmoid(outputs)), labels)
        print(torch.squeeze(torch.sigmoid(outputs)))
        self.log("test_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log(
            "test_acc_macro",
            acc_macro,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "test_acc_micro",
            acc_micro,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "test_acc_weighted",
            acc_weighted,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        self.log("test_auc", auc, on_step=True, on_epoch=True, logger=True)


if __name__ == "__main__":
    import warnings
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    warnings.simplefilter("ignore", UserWarning)
    model = CloudClassifierPL().load_from_checkpoint(
        checkpoint_path="/home/mvries/Documents/GitHub/cellshape-cloud/"
        "lightning_logs/version_29/checkpoints/epoch=238-step=10038.ckpt",
        # hparams_file="/home/mvries/Documents/GitHub/cellshape-cloud/lightning_logs/version_29/hparams.yaml",
        map_location="cuda",
    )
    vessel_data = VesselDataModule()
    vessel_data.setup()
    trainer = pl.Trainer(
        gpus=1, callbacks=[EarlyStopping(monitor="val_loss", mode="min")]
    )
    trainer.fit(model, vessel_data)
    trainer.test(model=model, datamodule=vessel_data)
