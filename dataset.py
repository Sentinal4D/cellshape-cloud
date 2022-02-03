import torch
from torch.utils.data import Dataset
from pyntcloud import PyntCloud
import glob


class PointCloudDataset(Dataset):
    def __init__(self,
                 img_dir,
                 normalise=True):
        self.img_dir = img_dir
        self.normalise = normalise
        self.files = glob.glob('*.ply')

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        file = self.files[idx]
        point_cloud = PyntCloud.from_file(self.img_dir + file)
        point_cloud = torch.tensor(point_cloud.points.values)

        return point_cloud
