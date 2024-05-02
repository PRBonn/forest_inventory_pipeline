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
from forest_inventory_pipeline.utils import logger
from forest_inventory_pipeline.utils.math import lmap
from forest_inventory_pipeline.cloud import Cloud


def main(
    cloud_file: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False, help="The cloud file"),
    ],
    write_file: Annotated[
        Optional[Path],
        typer.Argument(file_okay=True, dir_okay=False, help="the output file to write"),
    ] = None,
    scale: Annotated[
        Optional[float],
        typer.Option(help="scale used when saving the las file. leave default"),
    ] = 0.001,
):
    cloud = Cloud.load(cloud_file)
    logger.info(f"cloud read from {cloud_file}")

    header = laspy.LasHeader(point_format=2, version="1.2")
    header.offsets = np.min(cloud.points, axis=0)
    header.scales = np.array([scale, scale, scale])

    if cloud.has_instance:
        header.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int8))

    las = laspy.LasData(header)

    las.x = cloud.points[:, 0]
    las.y = cloud.points[:, 1]
    las.z = cloud.points[:, 2]
    if cloud.has_colors:
        # las spec needs 16 bit colors
        from_range = np.array([0, 1])
        to_range = np.array([0, 256 * 256 - 1])
        las.red = lmap(cloud.colors[:, 0], from_range, to_range)
        las.green = lmap(cloud.colors[:, 1], from_range, to_range)
        las.blue = lmap(cloud.colors[:, 2], from_range, to_range)

    if cloud.has_instance:
        las.treeID = cloud.instance.ravel()

    las.header.number_of_points_by_return[0] = len(las.x)
    las.return_number = np.full(len(las.x), 1)
    las.number_of_returns = np.full(len(las.x), 1)

    if write_file is None:
        write_file = cloud_file.with_suffix(".las")

    write_file.parent.mkdir(parents=True, exist_ok=True)
    las.write(write_file.as_posix())
    logger.info(f"las file written to {write_file}")


if __name__ == "__main__":
    typer.run(main)
