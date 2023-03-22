import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC
from pointcloud_dataset import VesselMNIST3D, SingleCellDataset

from vendor.encoders import DGCNNEncoder


def make_weights_for_balanced_classes(images, classes, nclasses=2):
    n_images = len(images)

    count_per_class = [150, 1185]
    weight_per_class = [0.0] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = float(n_images) / float(count_per_class[i])
    weights = [0] * n_images
    for idx, image_class in enumerate(classes):
        weights[idx] = weight_per_class[image_class]
    return weights


class SingleCellDataModule(pl.LightningDataModule):
    def __init__(
        self,
        annotations_file="/home/mvries/Documents/Datasets/OPM/"
        "SingleCellFromNathan_17122021/all_data_removedwrong"
        "_ori_removedTwo_train_test.csv",
        points_dir="/home/mvries/Documents/Datasets/OPM/"
        "SingleCellFromNathan_17122021",
        img_size=400,
        transform=None,
        cell_component="cell",
        num_points=2048,
        batch_size=16,
    ):
        super().__init__()
        self.annot_df = annotations_file
        self.img_dir = points_dir
        self.img_size = img_size
        self.transform = transform
        self.cell_component = cell_component
        self.num_points = num_points
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dset = SingleCellDataset(
            annotations_file=self.annot_df,
            points_dir=self.img_dir,
            partition="train",
        )

        self.val_dset = SingleCellDataset(
            annotations_file=self.annot_df,
            points_dir=self.img_dir,
            partition="val",
        )

        self.test_dset = SingleCellDataset(
            annotations_file=self.annot_df,
            points_dir=self.img_dir,
            partition="test",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset, batch_size=self.batch_size, shuffle=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset, batch_size=self.batch_size, shuffle=True
        )


class VesselDataModule(pl.LightningDataModule):
    """Cassava DataModule for Lightning"""

    def __init__(
        self,
        transform=None,
        batch_size=16,
        points_dir="/mnt/nvme0n1/Datasets/MedMNIST/vesselmnist3d/",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform
        self.points_dir = points_dir

    def setup(self, stage=None):
        self.vessel_train = VesselMNIST3D(
            self.points_dir, centre=True, scale=20.0, partition="train"
        )

        weights = make_weights_for_balanced_classes(
            self.vessel_train.files, self.vessel_train.class_labels
        )
        weights = torch.DoubleTensor(weights)
        self.sampler = WeightedRandomSampler(weights, len(weights))

        self.vessel_val = VesselMNIST3D(
            self.points_dir, centre=True, scale=20.0, partition="val"
        )
        self.vessel_test = VesselMNIST3D(
            self.points_dir, centre=True, scale=20.0, partition="test"
        )

    def train_dataloader(self):
        return DataLoader(
            self.vessel_train, batch_size=self.batch_size, sampler=self.sampler
        )

    def val_dataloader(self):
        return DataLoader(self.vessel_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.vessel_test, batch_size=self.batch_size)


class CloudClassifierPL(pl.LightningModule):
    def __init__(self, criterion=nn.CrossEntropyLoss(), num_classes=10):
        super(CloudClassifierPL, self).__init__()

        self.save_hyperparameters(ignore=["criterion", "model"])
        self.lr = 0.00001
        self.criterion = criterion
        self.model = DGCNNEncoder(num_features=num_classes)
        self.num_classes = num_classes
        self.accuracy_macro = Accuracy(
            task="binary", num_classes=num_classes, average="macro"
        )
        self.accuracy_micro = Accuracy(
            task="binary", num_classes=num_classes, average="micro"
        )
        self.accuracy_weighted = Accuracy(
            task="binary", num_classes=num_classes, average="weighted"
        )
        self.AUC = AUROC(task="multiclass", num_classes=num_classes)

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
        labels = torch.squeeze(labels)
        loss = self.criterion(outputs, labels)
        preds = torch.sigmoid(outputs)
        acc = self.accuracy_weighted(torch.squeeze(preds), labels)
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
        labels = torch.squeeze(labels)
        loss = self.criterion(outputs, labels)
        preds = torch.sigmoid(outputs)
        acc = self.accuracy_weighted(preds, labels)
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch[0], batch[1]
        outputs = self.model(inputs)
        labels = torch.squeeze(labels)
        loss = self.criterion(torch.squeeze(outputs), labels)
        preds = torch.sigmoid(outputs)

        acc_macro = self.accuracy_macro(torch.squeeze(preds), labels)
        acc_micro = self.accuracy_micro(torch.squeeze(preds), labels)
        acc_weighted = self.accuracy_weighted(torch.squeeze(preds), labels)
        auc = self.AUC(torch.squeeze(torch.sigmoid(outputs)), labels)
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

    class MyEarlyStopping(EarlyStopping):
        def on_validation_end(self, trainer, pl_module):
            # override this to disable early stopping at the end of val loop
            pass

        def on_train_end(self, trainer, pl_module):
            # instead, do it at the end of training loop
            self._run_early_stopping_check(trainer)

    warnings.simplefilter("ignore", UserWarning)
    model = CloudClassifierPL(num_classes=5)
    # .load_from_checkpoint(
    # checkpoint_path="/home/mvries/Documents/GitHub/cellshape-cloud/"
    # "lightning_logs/version_52/checkpoints/epoch=132-step=11172.ckpt"
    # )
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")

    cell_data = SingleCellDataModule()
    cell_data.setup()
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, callbacks=[checkpoint_callback]
    )

    trainer.fit(model, cell_data)
    trainer.test(model=model, datamodule=cell_data)
