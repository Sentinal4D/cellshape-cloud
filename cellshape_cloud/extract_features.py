import argparse

from cellshape_cloud.helpers.extract import extract

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracting features from a graph-based autoencoder"
    )
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
        "--cell_component",
        default="cell",
        type=str,
        choices=["cell", "nucleus"],
        help="Please provide the component of the cell "
        "to extract shape features from.",
    )
    parser.add_argument(
        "--pretrained_path",
        default="/run/user/1128299809/gvfs/smb-share:server=rds.icr.ac.uk,"
        "share=data/DBI/DUDBI/DYNCESYS/mvries/ResultsAlma/TearingNetNew/"
        "nets/dgcnn_foldingnet_128_001.pt",
        type=str,
        help="Please provide the path to a pretrained autoencoder.",
    )

    args = parser.parse_args()
    extract(args=args)
