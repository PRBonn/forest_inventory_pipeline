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
import typer
from typing_extensions import Annotated
from typing import Optional
from pathlib import Path
import numpy as np
import laspy
from forest_inventory_pipeline.cloud import Cloud
from forest_inventory_pipeline.render import render
from forest_inventory_pipeline.utils import logger


def main(
    las_file: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False, help="The las file"),
    ],
    write_file: Annotated[
        Optional[Path],
        typer.Argument(file_okay=True, dir_okay=False, help="the output file to write"),
    ] = None,
    zero_min: bool = False,
    visualize: bool = False,
):
    las = laspy.read(str(las_file))
    logger.info(f"las read from {las_file}")

    xyz = las.xyz
    if zero_min:
        xyz = xyz - np.min(xyz, axis=0)
    points = xyz

    # for donager
    instance = las.treeID
    # semantic = las.classification

    cloud = Cloud.from_points(points)
    cloud.instance = np.nan_to_num(instance).astype(np.int32)
    if write_file is None:
        write_file = las_file.with_suffix(".ply")

    cloud.write(write_file)
    logger.info(f"cloud file written to {write_file}")

    if visualize:
        cloud = cloud.colorize_using_attribute("instance")
        render([cloud])


if __name__ == "__main__":
    typer.run(main)
