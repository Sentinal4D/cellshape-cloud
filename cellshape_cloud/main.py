import torch
from torch.utils.data import DataLoader
import argparse
from datetime import datetime
import logging


import cellshape_cloud as cscloud
from cellshape_cloud.vendor.chamfer_distance import ChamferLoss
from cellshape_cloud.helpers.reports import get_experiment_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cellshape-cloud")
    parser.add_argument(
        "--cloud_convert",
        default=False,
        type=bool,
        help="Do you need to convert 3D images to point clouds?",
    )
    parser.add_argument(
        "--dataset_path",
        default="./datasets/",
        type=str,
        help="Please provide the path to the "
        "dataset of 3D images or point clouds",
    )
    parser.add_argument(
        "--dataframe_path",
        default="./dataframe/",
        type=str,
        help="Please provide the path to the dataframe "
        "containing information on the dataset.",
    )
    parser.add_argument(
        "--output_path",
        default="./",
        type=str,
        help="Please provide the path for where to save output.",
    )
    parser.add_argument(
        "--num_epochs",
        default=250,
        type=int,
        help="Provide the number of epochs for the " "autoencoder training.",
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
        "--learning_rate",
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
        default=0,
        type=int,
        help="Please provide the value of proximality "
        "[0 = distal, 1 = proximal, 2 = both].",
    )
    parser.add_argument(
        "--pretrained_path",
        default="/run/user/1128299809/gvfs/smb-share:server=rds.icr.ac.uk,"
        "share=data/DBI/DUDBI/DYNCESYS/mvries/ResultsAlma/TearingNetNew/"
        "nets/dgcnn_foldingnet_128_008.pt",
        type=str,
        help="Please provide the path to a pretrained autoencoder.",
    )

    args = parser.parse_args()
    if args.cloud_convert:
        print("Converting tif to point cloud using cellshape-helper")
    params = {
        "cloud_convert": args.cloud_convert,
        "pretrained_path": args.pretrained_path,
        "dataframe_path": args.dataframe_path,
        "dataset_path": args.dataset_path,
        "output_path": args.output_path,
        "num_epochs_autoencoder": args.num_epochs,
        "num_features": args.num_features,
        "k": args.k,
        "encoder_type": args.encoder_type,
        "decoder_type": args.decoder_type,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
    }

    model = cscloud.CloudAutoEncoder(
        num_features=args.num_features,
        k=args.k,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
    )

    dataset = cscloud.PointCloudDataset(args.dataset_path)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    criterion = ChamferLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate * 16 / args.batch_size,
        betas=(0.9, 0.999),
        weight_decay=1e-6,
    )

    name_logging, name_model, name_writer, name = get_experiment_name(
        model=model, output_dir=args.output_dir
    )
    logging_info = name_logging, name_model, name_writer, name

    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logging.basicConfig(filename=name_logging, level=logging.INFO)
    logging.info(f"Started training model {name} at {now}.")

    output = cscloud.train(
        model, dataloader, args.num_epochs, criterion, optimizer, logging_info
    )
