# MIT License
#
# Copyright (c) 2024 Meher Malladi, Luca Lobefaro, Tiziano Guadagnino, Cyrill Stachniss
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import csv
import numpy as np
import pickle
from pathlib import Path
from scipy.spatial import KDTree
from typing_extensions import Annotated
from forest_inventory_pipeline.cloud import Cloud
from forest_inventory_pipeline.clip_z import clip_z
from forest_inventory_pipeline.primitives import Cylinder
from forest_inventory_pipeline.traits.dbh import fit_cylinders
from forest_inventory_pipeline.cluster import project_2d_then_cluster_3d
from forest_inventory_pipeline.preprocess import segment_ground, normalize_height

import typer


def get_normalized_cloud(cloud: Cloud):
    ground_cloud, non_ground_cloud = segment_ground(cloud)
    normalized_cloud = normalize_height(
        non_ground_cloud=non_ground_cloud, ground_cloud=ground_cloud
    )
    return normalized_cloud


def cluster_cloud(cloud: Cloud):
    k = 399
    beta = 0.517
    clustered_cloud = project_2d_then_cluster_3d(cloud, k=k, beta=beta)
    return clustered_cloud


def make_target_arrays(target_cylinder_data):
    target_tag_id_array = np.array([tree_id for tree_id in target_cylinder_data.keys()])
    target_xyz_array = np.array([data["xyz"] for data in target_cylinder_data.values()])
    target_dbh_array = np.array([data["dbh"] for data in target_cylinder_data.values()])
    return target_tag_id_array, target_xyz_array, target_dbh_array


def make_pred_arrays(pred_cylinders: list[Cylinder]):
    pred_xyz_array = np.array([cyl.center for cyl in pred_cylinders])
    pred_dbh_array = np.array([2 * cyl.radius for cyl in pred_cylinders])

    return pred_xyz_array, pred_dbh_array


def evaluate_pred_against_target_cylinders(pred_arrays, target_arrays):
    pred_xyz_array, pred_dbh_array = pred_arrays
    target_id_array, target_xyz_array, target_dbh_array = target_arrays

    # the search
    # in xy plane, for each target_tree, find the closest pred_tree within a radius bound
    kdtree = KDTree(pred_xyz_array[:, :2])
    dist, closest_pred_to_target = kdtree.query(
        target_xyz_array[:, :2], k=1, p=2, distance_upper_bound=2.0, workers=-1
    )

    valid_pred_to_target = closest_pred_to_target != pred_xyz_array.shape[0]
    # loop through range(0, target_id_max) and diff target dbh and corresponding closest instance pred dbh
    error = np.array(
        [
            target_dbh_array[idx] - pred_dbh_array[closest_pred_to_target[idx]]
            for idx in range(target_id_array.shape[0])
            if valid_pred_to_target[idx]
        ]
    )

    return error


def print_metrics(error):
    rmse = np.sqrt(np.mean(np.square(error)))
    print("===DBH Error===")
    print(f"RMSE: {rmse*100:.3f} cm")
    return rmse


def evaluate_cylinders(cylinders: list, target_dbh_dict: dict):
    pred_arrays = make_pred_arrays(cylinders)
    target_arrays = make_target_arrays(target_dbh_dict)
    error = evaluate_pred_against_target_cylinders(pred_arrays, target_arrays)
    rmse = print_metrics(error)
    return rmse, error


def evaluate_dbh(name, target_cloud_fp, inventory_pickle_fp):
    print("\\\\\\\\", name, "////")
    cloud = get_normalized_cloud(Cloud.load(target_cloud_fp))
    cloud = cloud.voxel_down_sample(0.15)
    with inventory_pickle_fp.open("rb") as dbh_pickle_file:
        target_dbh_dict = pickle.load(dbh_pickle_file)
    clipped_cloud = clip_z(cloud, min_z=1.0, max_z=4.0)
    clustered_cloud = cluster_cloud(clipped_cloud)
    cylinders = fit_cylinders(clustered_cloud, ransac_iterations=10000, voxel_size=0.15)
    rmse, error = evaluate_cylinders(cylinders, target_dbh_dict)
    print("////", name, "\\\\\\\\")
    return name, rmse, error


def append_results_to_csv(results, rmse_all, exp_num, filename):
    with open(filename, "a", newline="") as csvfile:
        fieldnames = ["exp_num"]
        fieldnames.extend([result[0] for result in results])
        fieldnames.append("rmse_all")
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Check if file is empty, if so, write header
        if csvfile.tell() == 0:
            writer.writeheader()
        row = {result[0]: result[1] for result in results}
        row["exp_num"] = exp_num
        row["rmse_all"] = rmse_all

        writer.writerow(row)


def eval_loop(data_dir: Path, inventory_dir: Path):
    ground_truth_data = sorted(data_dir.glob("*.ply"))
    inventory_data = list(sorted(inventory_dir.glob("*.pickle")))

    pair_names = [file.stem for file in ground_truth_data]

    results = []
    error_all = []
    for arg_pair in zip(pair_names, ground_truth_data, inventory_data):
        name, rmse, error = evaluate_dbh(*arg_pair)
        results.append((name, rmse))
        error_all.append(error)
    error_all = np.hstack(error_all)
    rmse_all = np.sqrt(np.mean(np.square(error_all)))
    print(results)
    print("rmse_all", rmse_all)
    return results, rmse_all


def main(
    data_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            help="The directory should contain .ply files of all clouds to be evaluated.",
        ),
    ],
    inventory_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            help="""The directory should contain .pickle files of DBH inventory of corresponding point clouds in data_dir.
The assignment happens by (sorted) name order. The pickles themselves should just contain a dict whose keys are some
tree id and another dict. This second dict should contain keys xyz and dbh. xyz is the location of the tree in point
cloud frame.""",
        ),
    ],
    repeat: bool = False,
):
    """
    This script can be used for evaluating the DBH estimation of trees. However it expects input in a specific format.
    """
    num_repeat = 50 if repeat else 1
    filename = "dbh_results.csv"
    for iter in range(num_repeat):
        results, rmse_all = eval_loop(data_dir, inventory_dir)
        append_results_to_csv(results, rmse_all, iter, filename)


if __name__ == "__main__":
    typer.run(main)
