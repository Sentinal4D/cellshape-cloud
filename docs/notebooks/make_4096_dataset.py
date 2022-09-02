from pyntcloud import PyntCloud
import pandas as pd
from tqdm import tqdm

from pathlib import Path

"""
Functions adapted from PyTorch Geometric
"""
import torch


def parse_txt_array(src, sep=None, start=0, end=None, dtype=None):
    src = [[float(x) for x in line.split(sep)[start:end]] for line in src]
    src = torch.tensor(src, dtype=dtype).squeeze()
    return src


def parse_off(src):
    # Some files may contain a bug and do not have a carriage return after OFF.
    if src[0] == "OFF":
        # Change to 2 for pymesh
        src = src[2:]
    else:
        src[0] = src[0][3:]

    num_nodes, num_faces = [int(float(item)) for item in src[0].split()[:2]]

    pos = parse_txt_array(src[1 : 1 + num_nodes])
    face = src[1 + num_nodes : 1 + num_nodes + num_faces]
    face = face_to_tri(face)
    data = {"pos": pos, "face": face}

    return data


def face_to_tri(face):
    face = [[int(x) for x in line.strip().split()] for line in face]

    triangle = torch.tensor([line[1:] for line in face if line[0] == 3])
    triangle = triangle.to(torch.int64)

    rect = torch.tensor([line[1:] for line in face if line[0] == 4])
    rect = rect.to(torch.int64)

    if rect.numel() > 0:
        first, second = rect[:, [0, 1, 2]], rect[:, [0, 2, 3]]
        return torch.cat([triangle, first, second], dim=0).t().contiguous()
    else:
        return triangle.t().contiguous()


def read_off(path):
    r"""Reads an OFF (Object File Format) file, returning both the position of
    nodes and their connectivity in a :class:`torch_geometric.data.Data`
    object.
    Args:
        path (str): The path to the file.
    """
    with open(path, "r") as f:
        src = f.read().split("\n")[:-1]
    return parse_off(src)


def sample_points(data, num):
    pos, face = data["pos"], data["face"]
    assert pos.size(1) == 3 and face.size(0) == 3

    pos_max = pos.abs().max()
    pos = pos / pos_max

    area = (pos[face[1]] - pos[face[0]]).cross(pos[face[2]] - pos[face[0]])
    area = area.norm(p=2, dim=1).abs() / 2

    prob = area / area.sum()
    sample = torch.multinomial(prob, num, replacement=True)
    face = face[:, sample]

    frac = torch.rand(num, 2)
    mask = frac.sum(dim=-1) > 1
    frac[mask] = 1 - frac[mask]

    vec1 = pos[face[1]] - pos[face[0]]
    vec2 = pos[face[2]] - pos[face[0]]

    pos_sampled = pos[face[0]]
    pos_sampled += frac[:, :1] * vec1
    pos_sampled += frac[:, 1:] * vec2

    pos_sampled = pos_sampled * pos_max

    return pos_sampled


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
    "/data/scratch/DBI/DUDBI/DYNCESYS/mvries/"
    "SingleCellFromNathan_17122021/Plate1/stacked_off/raw/"
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
    "SingleCellFromNathan_17122021/Plate2/stacked_off/raw/"
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
    "SingleCellFromNathan_17122021/Plate3/stacked_off/raw/"
)
PATH_TO_SAVE_PC = (
    "/data/scratch/DBI/DUDBI/DYNCESYS/mvries/"
    "SingleCellFromNathan_17122021/Plate3/"
    "stacked_pointcloud_4096/"
)
NUM_POINTS = 4096

mesh_to_pc(PATH_TO_SAVE_MESH, NUM_POINTS, PATH_TO_SAVE_PC)
