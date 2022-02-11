import torch
from torch.utils.data import Dataset
from pyntcloud import PyntCloud
from pathlib import Path


class PointCloudDataset(Dataset):
    def __init__(self, img_dir, normalise=True):
        self.img_dir = img_dir
        self.normalise = normalise
        self.p = Path(self.img_dir)
        self.files = list(self.p.glob("**/*.ply"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # read the image
        file = self.files[idx]
        point_cloud = PyntCloud.from_file(str(file))
        point_cloud = torch.tensor(point_cloud.points.values)

        return point_cloud
