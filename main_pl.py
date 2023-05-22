import torch
from torch.utils.data import DataLoader
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import logging

from cellshape_cloud.lightning_autoencoder import CloudAutoEncoderPL
from cellshape_cloud.pointcloud_dataset import (
    PointCloudDataset,
    SingleCellDataset,
    GefGapDataset,
    ModelNet40,
    ShapeNetDataset,
    OPMDataset,
    VesselMNIST3D,
)
from cellshape_cloud.reports import get_experiment_name, get_model_name
from cellshape_cloud.cloud_autoencoder import CloudAutoEncoder
from lightning.pytorch.loggers import WandbLogger
from pathlib import Path

from torch.nn.parameter import Parameter


def load_my_state_dict(mod, state_dict):
    own_state = mod.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print("    Not found: " + name)
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = Parameter.data
        own_state[name].copy_(param)


# Using pytorch lightning to train on multiple GPUs


def train_vae_pl(args):

    find_lr = False

    model = CloudAutoEncoder(
        num_features=args.num_features,
        k=args.k,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        shape=args.shape,
        sphere_path=args.sphere_path,
        gaussian_path=args.gaussian_path,
        std=args.std,
    )
    autoencoder = CloudAutoEncoderPL(args=args, model=model)

    if args.is_pretrained_shapenet:
        # try:
        #     checkpoint = torch.load(
        #         args.pretrained_path,
        #         map_location=lambda storage, loc: storage
        #     )
        #     # "load encoder"
        #     model_dict = autoencoder.state_dict()
        #     for k in checkpoint:
        #         if k in model_dict:
        #             model_dict[k] = checkpoint[k]
        #             print("    Found weight: " + k)
        #         elif k.replace("encoder.", "model.encoder.") in model_dict:
        #             model_dict[
        #                 k.replace("encoder.", "model.encoder.")
        #             ] = checkpoint[k]
        #             print("    Found weight: " + k)
        #         elif k.replace("decoder.", "model.decoder.") in model_dict:
        #             model_dict[
        #                 k.replace("decoder.", "model.decoder.")
        #             ] = checkpoint[k]
        #             print("    Found weight: " + k)
        #
        #     autoencoder.load_state_dict(model_dict)
        #
        # except Exception as e:
        #     print(f"Cannot load model due to error {e}.")
        #     print("Training from scratch")
        try:
            file = list(Path(args.pretrained_path).glob("*.pkl"))[0]
            print(f"Loading model from {file}")
            checkpoint = torch.load(
                file, map_location=lambda storage, loc: storage
            )
            load_my_state_dict(model, checkpoint)
            autoencoder = CloudAutoEncoderPL(args=args, model=model)
        except Exception as e:
            print(f"Cannot load model due to error {e}.")
            print("Training from scratch")

    else:
        if args.is_pretrained_lightning:
            try:
                autoencoder.load_lightning(args.pretrained_path)

            except Exception as e:
                print(f"Cannot load model due to error {e}.")

        else:
            try:
                autoencoder.load_model_foldingnet(args.pretrained_path)
            except Exception as e:
                print(f"Can't load pretrained network due to error {e}.")

    if args.dataset_type == "SingleCell":
        dataset = SingleCellDataset(
            args.dataframe_path,
            args.cloud_dataset_path,
            num_points=args.num_points,
        )

    elif args.dataset_type == "GefGap":
        dataset = GefGapDataset(
            args.dataframe_path,
            args.cloud_dataset_path,
            norm_std=args.norm_std,
            cell_component=args.cell_component,
        )
    elif args.dataset_type == "ModelNet":
        dataset = ModelNet40(args.cloud_dataset_path)
    elif args.dataset_type == "ShapeNet":
        dataset = ShapeNetDataset(
            root=args.cloud_dataset_path,
            dataset_name="shapenetcorev2",
            random_rotate=True,
            random_jitter=True,
            random_translate=True,
        )
    elif args.dataset_type == "OPM":
        dataset = OPMDataset(
            args.dataframe_path,
            args.cloud_dataset_path,
            norm_std=args.norm_std,
            cell_component=args.cell_component,
            single_path=args.single_path,
            gef_path=args.gef_path,
        )
    elif args.dataset_type == "VesselMNIST":
        dataset = VesselMNIST3D(args.cloud_dataset_path)
    else:
        dataset = PointCloudDataset(args.cloud_dataset_path)

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    logging_info = get_experiment_name(
        model=autoencoder.model, output_dir=args.output_dir
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="loss", every_n_epochs=1, save_last=True
    )

    model_name = get_model_name(autoencoder.model) + f"_{args.shape}"

    if args.logger == "wandb":
        logger = WandbLogger(
            project=args.project_name,
            name=model_name,
            log_model=True,
            save_dir=args.output_dir + logging_info[3],
        )
    else:
        logger = pl.loggers.TensorBoardLogger(
            save_dir=args.log_dir,
            name=args.drug_label,
        )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        max_epochs=args.num_epochs_autoencoder,
        default_root_dir=args.output_dir + logging_info[3],
        callbacks=[checkpoint_callback],
        strategy="ddp_find_unused_parameters_false",
        logger=logger,
    )

    if find_lr:
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(
            autoencoder, dataloader, num_training=100
        )

        # Results can be found in
        logging.info(lr_finder.results)

        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.savefig(logging_info[3] + "/lr_finder_plot.png")

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        logging.info(new_lr)

        # update hparams of the model
        autoencoder.hparams.lr = new_lr

    trainer.fit(autoencoder, dataloader)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cellshape-cloud")
    parser.add_argument(
        "--cloud_convert",
        default=False,
        type=str2bool,
        help="Do you need to convert 3D images to point clouds?",
    )

    parser.add_argument(
        "--tif_dataset_path",
        default="/home/mvries/Documents/CellShape/"
        "UploadData/Dataset/TestConvert/TestTiff/",
        type=str,
        help="Please provide the path to the " "dataset of 3D tif images",
    )
    parser.add_argument(
        "--mesh_dataset_path",
        default="/home/mvries/Documents/CellShape/"
        "UploadData/Dataset/TestConvert/TestMesh/",
        type=str,
        help="Please provide the path to the " "dataset of 3D meshes.",
    )
    parser.add_argument(
        "--cloud_dataset_path",
        default="/mnt/nvme0n1/Datasets/SingleCellFromNathan_17122021/",
        type=str,
        help="Please provide the path to the " "dataset of the point clouds.",
    )
    parser.add_argument(
        "--dataset_type",
        default="SingleCell",
        type=str,
        choices=[
            "SingleCell",
            "GefGap",
            "Other",
            "ModelNet",
            "ShapeNet",
            "OPM",
            "VesselMNIST",
        ],
        help="Please provide the type of dataset. "
        "If using the one from our paper, then choose 'SingleCell', "
        "otherwise, choose 'Other'.",
    )
    parser.add_argument(
        "--dataframe_path",
        default="/mnt/nvme0n1/Datasets/SingleCellFromNathan_17122021/"
        "all_data_removed"
        "wrong_ori_removedTwo.csv",
        type=str,
        help="Please provide the path to the dataframe "
        "containing information on the dataset.",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/mvries/Documents/Testing_output_cloud/",
        type=str,
        help="Please provide the path for where to save output.",
    )
    parser.add_argument(
        "--num_epochs_autoencoder",
        default=250,
        type=int,
        help="Provide the number of epochs for the autoencoder training.",
    )
    parser.add_argument(
        "--num_features",
        default=128,
        type=int,
        help="Please provide the number of " "features to extract.",
    )
    parser.add_argument(
        "--k", default=16, type=int, help="Please provide the value for k."
    )
    parser.add_argument(
        "--encoder_type",
        default="foldingnet",
        type=str,
        help="Please provide the type of encoder.",
    )
    parser.add_argument(
        "--decoder_type",
        default="foldingnet",
        type=str,
        help="Please provide the type of decoder.",
    )
    parser.add_argument(
        "--learning_rate_autoencoder",
        default=0.0001,
        type=float,
        help="Please provide the learning rate "
        "for the autoencoder training.",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Please provide the batch size.",
    )
    parser.add_argument(
        "--proximal",
        default=2,
        type=int,
        help="Please provide the value of proximality "
        "[0 = distal, 1 = proximal, 2 = both].",
    )
    parser.add_argument(
        "--pretrained_path",
        default="/home/mvries/Downloads/shapenetcorev2_278.pkl",
        type=str,
        help="Please provide the path to a pretrained autoencoder.",
    )
    parser.add_argument(
        "--is_pretrained_lightning",
        default=False,
        type=str2bool,
        help="Is the pretrained model a lightning module?",
    )

    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="The number of gpus to use for training.",
    )
    parser.add_argument(
        "--norm_std",
        default=True,
        type=str2bool,
        help="Standardize by a factor of 20?",
    )
    parser.add_argument(
        "--num_points",
        default=2046,
        type=int,
        help="Enter the number of points in the point cloud",
    )
    parser.add_argument(
        "--cell_component",
        default="cell",
        type=str,
        help="Enter the number of points in the point cloud",
    )
    parser.add_argument(
        "--is_pretrained_shapenet",
        default=True,
        type=str2bool,
        help="Was trained on shapenet?",
    )
    parser.add_argument(
        "--shape",
        default="plane",
        choices=["sphere", "plane", "gaussian", "ModelNet"],
        type=str,
        help="What shape of points to concatenate features to?",
    )
    parser.add_argument(
        "--sphere_path",
        default="./cellshape_cloud/vendor/sphere.npy",
        type=str,
        help="Path to sphere.",
    )
    parser.add_argument(
        "--gaussian_path",
        default="./cellshape_cloud/vendor/gaussian.npy",
        type=str,
        help="Path to gaussian shape.",
    )
    parser.add_argument(
        "--std",
        default=3.0,
        type=float,
        help="Standard deviation of sampled points.",
    )
    parser.add_argument(
        "--single_path",
        default="./",
        type=str,
        help="Standard deviation of sampled points.",
    )
    parser.add_argument(
        "--gef_path",
        default="./",
        type=str,
        help="Standard deviation of sampled points.",
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["wandb", "tensorboard", "neptune"],
        help="Whether to use wandb for logging",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="StartFromShapeNet",
        help="Name of the project to log to",
    )

    arguments = parser.parse_args()
    train_vae_pl(arguments)
