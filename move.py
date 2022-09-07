import os
import pandas as pd
from tqdm import tqdm


def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def move(dir, csv_dir):
    df = pd.read_csv(csv_dir)
    listed_dir = sorted(os.listdir(dir))
    for file in tqdm(listed_dir):
        treatment = df[df["serialNumber"] == file.lower()[:-4]][
            "Treatment"
        ].values[0]
        create_dir_if_not_exist(dir + treatment + "/")
        os.rename(dir + file, dir + treatment + "/" + file.lower())


def move_to_cells(dir):
    listed_dir = sorted(os.listdir(dir))
    for file in listed_dir:
        create_dir_if_not_exist(dir + "Cells" + "/")
        os.rename(dir + file, dir + "Cells" + "/" + file)


if __name__ == "__main__":
    move(
        "/data/scratch/DBI/DUDBI/DYNCESYS/mvries/"
        "SingleCellFromNathan_17122021/"
        "Plate3/stacked_pointcloud_4096/Cells/",
        "/data/scratch/DBI/DUDBI/DYNCESYS/mvries/"
        "SingleCellFromNathan_17122021/"
        "Plate3/bakal_20210318_03.csv",
    )
