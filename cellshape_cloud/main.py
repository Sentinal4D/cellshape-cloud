import argparse

from train_autoencoder import train_autoencoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cellshape-cloud")
    parser.add_argument(
        "--cloud_convert",
        default=False,
        type=bool,
        help="Do you need to convert 3D images to point clouds?",
    )
    parser.add_argument(
        "--tif_dataset_path",
        default="./dataset_tif/",
        type=str,
        help="Please provide the path to the " "dataset of 3D tif images",
    )
    parser.add_argument(
        "--mesh_dataset_path",
        default="./dataset_mesh/",
        type=str,
        help="Please provide the path to the " "dataset of 3D meshes.",
    )
    parser.add_argument(
        "--cloud_dataset_path",
        default="./dataset_cloud/",
        type=str,
        help="Please provide the path to the " "dataset of the point clouds.",
    )
    parser.add_argument(
        "--dataframe_path",
        default="./dataframe/",
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
        "--num_epochs",
        default=1,
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
    # if args.cloud_convert:
    #     print("Converting tif to point cloud using cellshape-helper")

    output = train_autoencoder(args)
