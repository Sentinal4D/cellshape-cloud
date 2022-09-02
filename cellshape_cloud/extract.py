import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from cellshape_cloud.pointcloud_dataset import SingleCellDataset
from cellshape_cloud.cloud_autoencoder import CloudAutoEncoder


def extract(args):
    model = CloudAutoEncoder(
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
        exit()

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"The loss of the loaded model is {checkpoint['loss']}")
    except RuntimeError:
        print("The model architecture given doesn't match the one provided.")

    dataset = SingleCellDataset(
        annotations_file=args.dataframe_path,
        points_dir=args.dataset_path,
        cell_component=args.cell_component,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    from tqdm import tqdm

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_feat = []
    all_labels = []
    all_serial_numbers = []
    for data in tqdm(dataloader):
        inputs = data[0]
        lab = data[1]
        ser_num = data[3]
        inputs = inputs.to(device)

        output, features = model(inputs)
        all_feat.append(torch.squeeze(features).detach().cpu().numpy())
        all_labels.append(lab[0])
        all_serial_numbers.append(ser_num[0])

    extracted_df = pd.DataFrame(np.asarray(all_feat))
    extracted_df["Treatment"] = np.asarray(all_labels)
    extracted_df["serialNumber"] = np.asaarray(all_serial_numbers)
    out_p = Path(args.output_path)
    pretrained = Path(args.pretrained_path)
    out_p.mkdir(parents=True, exist_ok=True)
    extracted_df.to_csv(out_p / (str(pretrained.name)[:-2] + "csv"))
