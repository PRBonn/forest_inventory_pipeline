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
import numpy as np
import csv
from pathlib import Path
from typing_extensions import Annotated

from panoptic_eval_metrics import panoptic_quality_metrics
from forest_inventory_pipeline.cloud import Cloud
from forest_inventory_pipeline.clip_z import clip_z
from forest_inventory_pipeline.cluster import project_2d_then_cluster_3d
from forest_inventory_pipeline.utils import logger
from forest_inventory_pipeline.utils.decorators import stage
from forest_inventory_pipeline.utils.upsample import upsample
from forest_inventory_pipeline.preprocess import segment_ground, normalize_height


import typer


upsample = stage()(upsample)


@stage
def get_normalized_and_clipped_cloud(cloud: Cloud):
    ground_cloud, non_ground_cloud = segment_ground(cloud)
    normalized_cloud = normalize_height(
        non_ground_cloud=non_ground_cloud, ground_cloud=ground_cloud
    )
    clipped_cloud = clip_z(normalized_cloud, min_z=1.0)
    return clipped_cloud


@stage
def get_pred_cloud(cloud: Cloud):
    k = 358
    beta = 0.7568
    vds_cloud = cloud.voxel_down_sample(0.15)
    clustered_cloud = project_2d_then_cluster_3d(vds_cloud, k=k, beta=beta)
    upsampled_cloud = upsample(clustered_cloud, cloud, 0.15)
    return upsampled_cloud


@stage
def evaluate_pq(exp_name, target_cloud, pred_cloud):
    print(exp_name)
    results = panoptic_quality_metrics(pred_cloud, target_cloud, verbose=True)
    pq = results[0]
    print(exp_name, "PQ:", pq)
    return exp_name, pq


def append_results_to_csv(results, mean_pq, exp_num, filename):
    with open(filename, "a", newline="") as csvfile:
        fieldnames = ["exp_num"]
        fieldnames.extend([result[0] for result in results])
        fieldnames.append("mean_pq")
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Check if file is empty, if so, write header
        if csvfile.tell() == 0:
            writer.writeheader()

        row = {result[0]: result[1] for result in results}
        row["exp_num"] = exp_num
        row["mean_pq"] = mean_pq
        writer.writerow(row)


def eval_loop(data_dir: Path):
    ground_truth_data = sorted(data_dir.glob("*.ply"))
    pair_names = [file.stem for file in ground_truth_data]
    target_clouds = [get_normalized_and_clipped_cloud(Cloud.load(fp)) for fp in ground_truth_data]
    results = []
    for name, target_cloud in list(zip(pair_names, target_clouds)):
        logger.trace("Working on", name)
        pred_cloud = get_pred_cloud(target_cloud)
        result = evaluate_pq(name, target_cloud, pred_cloud)
        results.append(result)
    print("=" * 20)
    print(results)
    mean_pq = np.mean([res[1] for res in results])
    print("Mean PQ:", mean_pq)
    return results, mean_pq


def main(
    data_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            help="""The directory should contain .ply files of all clouds to be evaluated. The clouds should have associated
ground truth labels in an 'instance' attribute.""",
        ),
    ],
    repeat: bool = False,
):
    """
    This script can be used for evaluating the instance segmentation of trees. However it expects input in a specific format.
    """
    num_repeat = 50 if repeat else 1
    filename = "qs_pq_results.csv"
    for iter in range(num_repeat):
        results, mean_pq = eval_loop(data_dir)
        append_results_to_csv(results, mean_pq, iter, filename)


if __name__ == "__main__":
    typer.run(main)
