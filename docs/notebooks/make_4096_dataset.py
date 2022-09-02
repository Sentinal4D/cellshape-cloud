from cellshape_helper.vendor.pytorch_geometric_files import (
    read_off,
    sample_points,
)
from pyntcloud import PyntCloud
import pandas as pd
from tqdm import tqdm

from pathlib import Path


def create_dir_if_not_exist(path):
    p = Path(path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)


def mesh_to_pc(mesh_directory, num_points, save_dir):
    p = Path(mesh_directory)
    for folder in p.iterdir():

        files = list(folder.glob("*.off"))
        for mesh_file in tqdm(files):
            mesh_file_path = Path(mesh_file)
            data = read_off(mesh_file)
            # changed to .numpy() to avoid issue with pyntcloud
            points = sample_points(data=data, num=num_points).numpy()
            save_to_points_path = save_dir + folder.name
            create_dir_if_not_exist(save_to_points_path)
            split_string = mesh_file_path.name.split(".")
            file_name = split_string[0]
            cloud = PyntCloud(
                pd.DataFrame(data=points, columns=["x", "y", "z"])
            )
            cloud.to_file(save_to_points_path + file_name + ".ply")


# Plate 1
PATH_TO_SAVE_MESH = (
    "/data/scratch/DBI/DUDBI/DYNCESYS/mvries"
    "/SingleCellFromNathan_17122021/Plate1/stacked_off/raw/Cells/"
)
PATH_TO_SAVE_PC = (
    "/data/scratch/DBI/DUDBI/DYNCESYS/mvries/"
    "SingleCellFromNathan_17122021/Plate1/"
    "stacked_pointcloud_4096/"
)
NUM_POINTS = 4096

mesh_to_pc(PATH_TO_SAVE_MESH, NUM_POINTS, PATH_TO_SAVE_PC)


# Plate 2
PATH_TO_SAVE_MESH = (
    "/data/scratch/DBI/DUDBI/DYNCESYS/mvries/"
    "SingleCellFromNathan_17122021/Plate2/stacked_off/raw/Cells/"
)
PATH_TO_SAVE_PC = (
    "/data/scratch/DBI/DUDBI/DYNCESYS/mvries/"
    "SingleCellFromNathan_17122021/Plate2/"
    "stacked_pointcloud_4096/"
)
NUM_POINTS = 4096

mesh_to_pc(PATH_TO_SAVE_MESH, NUM_POINTS, PATH_TO_SAVE_PC)

# Plate 3
PATH_TO_SAVE_MESH = (
    "/data/scratch/DBI/DUDBI/DYNCESYS/mvries/"
    "SingleCellFromNathan_17122021/Plate3/stacked_off/raw/Cells/"
)
PATH_TO_SAVE_PC = (
    "/data/scratch/DBI/DUDBI/DYNCESYS/mvries/"
    "SingleCellFromNathan_17122021/Plate3/"
    "stacked_pointcloud_4096/"
)
NUM_POINTS = 4096

mesh_to_pc(PATH_TO_SAVE_MESH, NUM_POINTS, PATH_TO_SAVE_PC)
