import torch
from torch.utils.data import Dataset
from pyntcloud import PyntCloud
from pathlib import Path
import pandas as pd
import os


class PointCloudDataset(Dataset):
    def __init__(self, points_dir, centre=True, scale=20.0):
        self.points_dir = points_dir
        self.centre = centre
        self.scale = scale
        self.p = Path(self.points_dir)
        self.files = list(self.p.glob("**/*.ply"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # read the image
        file = self.files[idx]
        point_cloud = PyntCloud.from_file(str(file))
        mean = 0
        point_cloud = torch.tensor(point_cloud.points.values)
        if self.centre:
            mean = torch.mean(point_cloud, 0)

        scale = torch.tensor([[self.scale, self.scale, self.scale]])
        point_cloud = (point_cloud - mean) / scale

        return point_cloud, 0, 0, 0


class SingleCellDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        points_dir,
        img_size=400,
        transform=None,
        cell_component="cell",
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = points_dir
        self.img_size = img_size
        self.transform = transform
        self.cell_component = cell_component

        self.new_df = self.annot_df[
            (self.annot_df.xDim <= self.img_size)
            & (self.annot_df.yDim <= self.img_size)
            & (self.annot_df.zDim <= self.img_size)
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        treatment = self.new_df.loc[idx, "Treatment"]
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])
        if self.cell_component == "cell":
            component_path = "stacked_pointcloud"
        else:
            component_path = "stacked_pointcloud_nucleus"

        img_path = os.path.join(
            self.img_dir,
            plate_num,
            component_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )
        image = PyntCloud.from_file(img_path + ".ply")
        image = torch.tensor(image.points.values)
        mean = torch.mean(image, 0)
        std = torch.tensor([[20.0, 20.0, 20.0]])
        image = (image - mean) / std

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, treatment, feats, serial_number


class GefGapDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=400,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        cell_component="cell",
    ):
        self.new_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        plate_num = self.new_df.loc[idx, "PlateNumber"]
        treatment = self.new_df.loc[idx, "GEF_GAP_GTPase"]
        plate = "Plate" + str(plate_num)
        if self.cell_component == "cell":
            component_path = "stacked_pointcloud"
        else:
            component_path = "stacked_pointcloud_nucleus"

        img_path = os.path.join(
            self.img_dir,
            plate,
            component_path,
            "Cells",
            self.new_df.loc[idx, "serialNumber"],
        )
        image = PyntCloud.from_file(img_path + ".ply")
        image = image.points.values

        image = torch.tensor(image)
        mean = torch.mean(image, 0)
        std = torch.tensor([[20.0, 20.0, 20.0]])
        image = (image - mean) / std

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, treatment, 0, serial_number
