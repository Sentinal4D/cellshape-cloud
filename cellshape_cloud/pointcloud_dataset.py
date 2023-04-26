import torch
from torch.utils.data import Dataset
from pyntcloud import PyntCloud
from pathlib import Path
import pandas as pd
import os
import json
import h5py
from glob import glob
import numpy as np
from sklearn.decomposition import PCA

from sklearn import preprocessing


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
        num_points=2048,
        partition="all",
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = points_dir
        self.img_size = img_size
        self.transform = transform
        self.cell_component = cell_component
        self.num_points = num_points
        self.partition = partition
        if self.partition != "all":

            self.new_df = self.annot_df[
                (self.annot_df.xDim <= self.img_size)
                & (self.annot_df.yDim <= self.img_size)
                & (self.annot_df.zDim <= self.img_size)
                & (self.annot_df.Splits == self.partition)
            ].reset_index(drop=True)
        else:
            self.new_df = self.annot_df[
                (self.annot_df.xDim <= self.img_size)
                & (self.annot_df.yDim <= self.img_size)
                & (self.annot_df.zDim <= self.img_size)
            ].reset_index(drop=True)

        from sklearn import preprocessing

        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.new_df["Treatment"].values)

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        treatment = self.new_df.loc[idx, "Treatment"]
        # class_id = self.new_df.loc[idx, "Class"]
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])
        if self.num_points == 4096:
            num_str = "_4096"
        elif self.num_points == 1024:
            num_str = "_1024"
        else:
            num_str = ""

        if self.cell_component == "cell":
            component_path = "stacked_pointcloud" + num_str
        else:
            component_path = "stacked_pointcloud_nucleus" + num_str

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
        enc_labels = torch.tensor(self.le.transform([treatment]))
        return image, enc_labels, treatment, serial_number


class GefGapDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=100,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        cell_component="cell",
        norm_std=False,
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component
        self.norm_std = norm_std

        self.new_df = self.annot_df[
            (self.annot_df.xDim_cell <= self.img_size)
            & (self.annot_df.yDim_cell <= self.img_size)
            & (self.annot_df.zDim_cell <= self.img_size)
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        plate_num = self.new_df.loc[idx, "PlateNumber"]
        treatment = self.new_df.loc[idx, "GEF_GAP_GTPase"]
        plate = "Plate" + str(plate_num)
        if self.cell_component == "cell":
            component_path = "stacked_pointcloud"
            img_path = os.path.join(
                self.img_dir,
                plate,
                component_path,
                self.new_df.loc[idx, "serialNumber"],
            )
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
        if self.norm_std:
            std = torch.tensor([[20.0, 20.0, 20.0]])
        else:
            std = torch.abs(image - mean).max() * 0.9999999

        image = (image - mean) / std

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, treatment, 0, serial_number


class ModelNet40(Dataset):
    def __init__(self, img_dir, train="train", transform=None):

        self.img_dir = Path(img_dir)
        self.train = train
        self.transform = transform
        self.files = list(self.img_dir.glob(f"**/{train}/*.ply"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # read the image
        file = self.files[idx]
        image = PyntCloud.from_file(str(file))
        image = torch.tensor(image.points.values)
        label = str(file.name)[:-9]
        image = (image - torch.mean(image, 0)) / (image.max())

        return image, label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype(
        "float32"
    )
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi * 2 * np.random.choice(24) / 24
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    pointcloud[:, [0, 2]] = pointcloud[:, [0, 2]].dot(
        rotation_matrix
    )  # random rotation (x,z)
    return pointcloud


# The following was adapted from
# https://github.com/antao97/UnsupervisedPointCloudReconstruction
class ShapeNetDataset(Dataset):
    def __init__(
        self,
        root,
        dataset_name="modelnet40",
        num_points=2048,
        split="train",
        load_name=False,
        random_rotate=False,
        random_jitter=False,
        random_translate=False,
    ):

        assert dataset_name.lower() in [
            "shapenetcorev2",
            "shapenetpart",
            "modelnet10",
            "modelnet40",
        ]
        assert num_points <= 2048

        if dataset_name in ["shapenetpart", "shapenetcorev2"]:
            assert split.lower() in ["train", "test", "val", "trainval", "all"]
        else:
            assert split.lower() in ["train", "test", "all"]

        self.root = os.path.join(root, dataset_name + "*hdf5_2048")
        self.dataset_name = dataset_name
        self.num_points = num_points
        self.split = split
        self.load_name = load_name
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate

        self.path_h5py_all = []
        self.path_json_all = []
        if self.split in ["train", "trainval", "all"]:
            self.get_path("train")
        if self.dataset_name in ["shapenetpart", "shapenetcorev2"]:
            if self.split in ["val", "trainval", "all"]:
                self.get_path("val")
        if self.split in ["test", "all"]:
            self.get_path("test")

        self.path_h5py_all.sort()
        data, label = self.load_h5py(self.path_h5py_all)
        if self.load_name:
            self.path_json_all.sort()
            self.name = self.load_json(self.path_json_all)  # load label name

        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0)

    def get_path(self, type):
        path_h5py = os.path.join(self.root, "*%s*.h5" % type)
        self.path_h5py_all += glob(path_h5py)
        if self.load_name:
            path_json = os.path.join(self.root, "%s*_id2name.json" % type)
            self.path_json_all += glob(path_json)
        return

    def load_h5py(self, path):
        all_data = []
        all_label = []
        for h5_name in path:
            f = h5py.File(h5_name, "r+")
            data = f["data"][:].astype("float32")
            label = f["label"][:].astype("int64")
            f.close()
            all_data.append(data)
            all_label.append(label)
        return all_data, all_label

    def load_json(self, path):
        all_data = []
        for json_name in path:
            j = open(json_name, "r+")
            data = json.load(j)
            all_data += data
        return all_data

    def __getitem__(self, item):
        point_set = self.data[item][: self.num_points]
        label = self.label[item]
        if self.load_name:
            name = self.name[item]  # get label name

        if self.random_rotate:
            point_set = rotate_pointcloud(point_set)
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set)
        if self.random_translate:
            point_set = translate_pointcloud(point_set)

        # convert numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        label = label.squeeze(0)

        if self.load_name:
            return point_set, label, name
        else:
            return point_set, label

    def __len__(self):
        return self.data.shape[0]


class OPMDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=100,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        cell_component="cell",
        norm_std=True,
        single_path="./",
        gef_path="./",
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component
        self.norm_std = norm_std
        self.single_path = single_path
        self.gef_path = gef_path

        self.new_df = self.annot_df[
            (self.annot_df.xDim <= self.img_size)
            & (self.annot_df.yDim <= self.img_size)
            & (self.annot_df.zDim <= self.img_size)
            & (self.annot_df.Proximal == 1)
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        plate_num = self.new_df.loc[idx, "PlateNumber"]
        treatment = self.new_df.loc[idx, "Treatment"]
        plate = "Plate" + str(plate_num)

        if "accelerator" in self.new_df.loc[idx, "serialNumber"]:
            dat_type_path = self.single_path
            if self.cell_component == "cell":
                component_path = "stacked_pointcloud"
            elif self.cell_component == "smooth":
                component_path = "stacked_pointcloud_smoothed"
            else:
                component_path = "stacked_pointcloud_nucleus"

            img_path = os.path.join(
                self.img_dir,
                self.single_path,
                plate,
                component_path,
                treatment,
                str(self.new_df.loc[idx, "serialNumber"]),
            )

        else:
            dat_type_path = self.gef_path
            if self.cell_component == "cell":
                component_path = "stacked_pointcloud"
                img_path = os.path.join(
                    self.img_dir,
                    dat_type_path,
                    plate,
                    component_path,
                    str(self.new_df.loc[idx, "serialNumber"]),
                )
            else:
                component_path = "stacked_pointcloud_nucleus"
                img_path = os.path.join(
                    self.img_dir,
                    dat_type_path,
                    plate,
                    component_path,
                    "Cells",
                    str(self.new_df.loc[idx, "serialNumber"]),
                )

        image = PyntCloud.from_file(img_path + ".ply")
        image = image.points.values

        image = torch.tensor(image)
        mean = torch.mean(image, 0)
        if self.norm_std:
            std = torch.tensor([[20.0, 20.0, 20.0]])
        else:
            std = torch.std(image, 0)

        image = (image - mean) / std
        pc = PCA(n_components=3)
        u = torch.tensor(pc.fit_transform(image.numpy()))

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, treatment, u, serial_number


class VesselMNIST3D(Dataset):
    def __init__(self, points_dir, centre=True, scale=20.0, partition="train"):
        self.points_dir = points_dir
        self.centre = centre
        self.scale = scale
        self.p = Path(self.points_dir)
        self.partition = partition
        self.path = self.p / partition
        self.files = list(self.path.glob("**/*.ply"))
        self.classes = [
            x.parents[0].name.replace("_pointcloud", "") for x in self.files
        ]

        self.le = preprocessing.LabelEncoder()
        self.class_labels = self.le.fit_transform(self.classes)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # read the image
        file = self.files[idx]
        label = self.class_labels[idx]
        class_name = self.classes[idx]
        point_cloud = PyntCloud.from_file(str(file))
        mean = 0
        point_cloud = torch.tensor(point_cloud.points.values)
        if self.centre:
            mean = torch.mean(point_cloud, 0)

        scale = torch.tensor([[self.scale, self.scale, self.scale]])
        point_cloud = (point_cloud - mean) / scale
        pc = PCA(n_components=3)
        u = torch.tensor(pc.fit_transform(point_cloud.numpy()))

        return (
            point_cloud,
            torch.tensor(label, dtype=torch.int64),
            u,
            class_name,
        )
