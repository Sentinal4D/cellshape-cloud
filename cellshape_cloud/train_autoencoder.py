import torch
from torch.utils.data import DataLoader
from datetime import datetime
import logging


import cellshape_cloud as cscloud
from cellshape_cloud.vendor.chamfer_distance import ChamferLoss


def train_autoencoder(args):
    autoencoder = cscloud.CloudAutoEncoder(
        num_features=args.num_features,
        k=args.k,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
    )
    try:
        checkpoint = torch.load(args.pretrained_path)
    except FileNotFoundError:
        print(
            "This model doesn't exist. "
            "Please check the provided path and try again."
        )
        checkpoint = {"model_state_dict": None}

    try:
        autoencoder.load_state_dict(checkpoint["model_state_dict"])
        print(f"The loss of the loaded model is {checkpoint['loss']}")
    except RuntimeError:
        print("The model architecture given doesn't match the one provided.")
        print("Training from scratch")
    except AttributeError:
        print("Training from scratch")

    if args.dataset_type == "SingleCell":
        dataset = cscloud.SingleCellDataset(
            args.dataframe_path, args.cloud_dataset_path
        )
    else:
        dataset = cscloud.PointCloudDataset(args.input_dir)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    reconstruction_criterion = ChamferLoss()

    optimizer = torch.optim.Adam(
        autoencoder.parameters(),
        lr=args.learning_rate_autoencoder * 16 / args.batch_size,
        betas=(0.9, 0.999),
        weight_decay=1e-6,
    )
    logging_info = cscloud.get_experiment_name(
        model=autoencoder, output_dir=args.output_dir
    )
    name_logging, name_model, name_writer, name = logging_info
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logging.basicConfig(filename=name_logging, level=logging.INFO)
    logging.info(f"Started training model {name} at {now}.")
    print(f"Started training model {name} at {now}.")
    for arg, value in sorted(vars(args).items()):
        logging.info(f"Argument {arg}: {value}")
        print(f"Argument {arg}: {value}")

    autoencoder, name_logging, name_model, name_writer, name = cscloud.train(
        model=autoencoder,
        dataloader=dataloader,
        num_epochs=args.num_epochs_autoencoder,
        criterion=reconstruction_criterion,
        optimizer=optimizer,
        logging_info=logging_info,
    )
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logging.info(f"Finished training at {now}.")
    print(f"Finished training at {now}.")

    return autoencoder, name_logging, name_model, name_writer, name
