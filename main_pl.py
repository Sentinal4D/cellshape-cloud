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
)
from cellshape_cloud.reports import get_experiment_name
from cellshape_cloud.cloud_autoencoder import CloudAutoEncoder


def train_vae_pl(args):

    find_lr = False

    model = CloudAutoEncoder(
        num_features=args.num_features,
        k=args.k,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
    )
    autoencoder = CloudAutoEncoderPL(args=args, model=model)

    try:
        autoencoder.load_model(args.pretrained_path)
    except Exception as e:
        print(f"Can't load pretrained network due to error {e}")

    if args.dataset_type == "SingleCell":
        dataset = SingleCellDataset(
            args.dataframe_path, args.cloud_dataset_path
        )

    elif args.dataset_type == "GefGap":
        dataset = GefGapDataset(args.dataframe_path, args.cloud_dataset_path)

    else:
        dataset = PointCloudDataset(args.cloud_dataset_path)

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir, save_top_k=1, monitor="loss"
    )

    logging_info = get_experiment_name(
        model=autoencoder.model, output_dir=args.output_dir
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        max_epochs=args.num_epochs_autoencoder,
        default_root_dir=args.output_dir + logging_info[3],
        callbacks=[checkpoint_callback],
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
        default="/home/mvries/Documents/Datasets/OPM/" "VickyCellshape/",
        type=str,
        help="Please provide the path to the " "dataset of the point clouds.",
    )
    parser.add_argument(
        "--dataset_type",
        default="GefGap",
        type=str,
        choices=["SingleCell", "GefGap", "Other"],
        help="Please provide the type of dataset. "
        "If using the one from our paper, then choose 'SingleCell', "
        "otherwise, choose 'Other'.",
    )
    parser.add_argument(
        "--dataframe_path",
        default="/home/mvries/Documents/Datasets/OPM/VickyCellshape/"
        "cn_allFeatures_withGeneNames_updated.csv",
        type=str,
        help="Please provide the path to the dataframe "
        "containing information on the dataset.",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/mvries/Documents/Testing_output/",
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
        "--k", default=20, type=int, help="Please provide the value for k."
    )
    parser.add_argument(
        "--encoder_type",
        default="dgcnn",
        type=str,
        help="Please provide the type of encoder.",
    )
    parser.add_argument(
        "--decoder_type",
        default="foldingnetbasic",
        type=str,
        help="Please provide the type of decoder.",
    )
    parser.add_argument(
        "--learning_rate_autoencoder",
        default=0.00001,
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
        default=0,
        type=int,
        help="Please provide the value of proximality "
        "[0 = distal, 1 = proximal, 2 = both].",
    )
    parser.add_argument(
        "--pretrained_path",
        default="/run/user/1128299809/gvfs/smb-share:server="
        "rds.icr.ac.uk,share=data"
        "/DBI/DUDBI/DYNCESYS/mvries/ResultsAlma/TearingNetNew/"
        "nets/dgcnn_foldingnet_128_009.pt",
        type=str,
        help="Please provide the path to a pretrained autoencoder.",
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="The number of gpus to use for training.",
    )

    args = parser.parse_args()
    train_vae_pl(args)
