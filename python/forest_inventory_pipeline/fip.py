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
from pathlib import Path
from typing_extensions import Annotated
from typing import Optional, Callable
from rich.pretty import pretty_repr

from .cloud import Cloud
from .render import render
from .utils import logger
from .utils.decorators import timer
from .utils.io import load_yaml_or_json
from .preprocess import segment_ground, normalize_height

import typer


def log_config(callable: Callable):
    def wrapper(*args, **kwargs):
        config = callable(*args, **kwargs)
        logger.info(f"Read in config: {pretty_repr(config)}")
        return config

    return wrapper


load_yaml_or_json = log_config(load_yaml_or_json)


@timer(level="info", name="Segmenting instances")
def _segment_instance(
    cloud: Cloud,
    min_z: float | None = None,
    max_z: float | None = None,
    voxel_size: float | None = None,
    upsample: bool = False,
    **quickshift_config,
):
    from .utils.upsample import upsample as _upsample
    from .cluster import project_2d_then_cluster_3d
    from .clip_z import clip_z

    cloud = clip_z(cloud, min_z=min_z, max_z=max_z)
    if voxel_size:
        high_res_cloud = cloud.clone()
        cloud = cloud.voxel_down_sample(voxel_size)
    clustered_cloud = project_2d_then_cluster_3d(cloud, **quickshift_config)
    if upsample and voxel_size:
        upsampled_cloud = _upsample(clustered_cloud, high_res_cloud, low_res_voxel_size=voxel_size)
        logger.info("upsampled clustered cloud back to original resolution")
        clustered_cloud = upsampled_cloud
    return clustered_cloud


#### Public

cli = typer.Typer(name="Forest Inventory Pipeline")


@cli.command()
def instance(
    input_fp: Annotated[
        Optional[Path],
        typer.Argument(
            metavar="INPUT",
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Input cloud to process. If passed, will be used if ground segmentation is not being skipped. It is an error to not pass this if output dir is also not specified or the output dir does not contain the appropriate stage to start processing from.",
        ),
    ] = None,
    config_fp: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="a config file which can be used for individual processing steps.",
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            file_okay=False,
            dir_okay=True,
            help="directory used to cache results. intermediate process files like ground segmented cloud, normalized cloud are written here as well. they can be used to skip stages in subsequent runs",
        ),
    ] = None,
    visualize: Annotated[
        bool, typer.Option("--visualize", "-v", help="visualize the clustered result.")
    ] = False,
    skip_ground_seg: Annotated[
        bool,
        typer.Option(
            "--skip_gs",
            help="Skip ground segmentation. Only skipped if ground_cloud.ply cloud exists inside the output dir.",
        ),
    ] = False,
    skip_normalize: Annotated[
        bool,
        typer.Option(
            "--skip_norm",
            help="Skip height normalization. Only skipped if height_normalized.ply cloud exists inside the output dir. Will not be skipped if ground segmantation has been run in the current process, so you have to specify both options.",
        ),
    ] = False,
):
    """
    Preprocess the cloud (ground segmentation and height normalization) and then instance segment the cloud.
    """
    config: dict = load_yaml_or_json(config_fp) if config_fp else {}  # type:ignore
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    # define all intermediate file paths
    gc_fp = output_dir / "ground_cloud.ply" if output_dir else None
    ngc_fp = output_dir / "non_ground_cloud.ply" if output_dir else None
    normalized_fp = output_dir / "normalized_cloud.ply" if output_dir else None
    instance_seg_fp = output_dir / "instance_segmented_cloud.ply" if output_dir else None

    # ground seg
    force_normalize = False
    if skip_ground_seg and gc_fp and gc_fp.exists() and ngc_fp and ngc_fp.exists():
        ground_cloud = Cloud.load(gc_fp)
        non_ground_cloud = Cloud.load(ngc_fp)
    else:
        if input_fp is None:
            if skip_ground_seg:
                raise Exception(
                    "Necessary files for skipping ground segmentation do no exist in the config dir if passed. Please start from processing an input point cloud"
                )
            else:
                raise Exception("Please pass an input point cloud")
        input_cloud: Cloud = Cloud.load(input_fp)
        ground_seg_config = config.get("ground_seg", {})
        ground_cloud, non_ground_cloud = segment_ground(input_cloud, **ground_seg_config)
        force_normalize = True
        if gc_fp and ngc_fp:
            ground_cloud.write(gc_fp)
            non_ground_cloud.write(ngc_fp)

    # normalization
    if skip_normalize and normalized_fp and normalized_fp.exists() and not force_normalize:
        normalized_cloud = Cloud.load(normalized_fp)
    else:
        normalization_config = config.get("normalize", {})
        normalized_cloud = normalize_height(
            non_ground_cloud=non_ground_cloud, ground_cloud=ground_cloud, **normalization_config
        )
        if normalized_fp:
            normalized_cloud.write(normalized_fp)

    cluster_config = config.get("cluster", {})
    clustered_cloud = _segment_instance(normalized_cloud, **cluster_config)
    if instance_seg_fp:
        clustered_cloud.write(instance_seg_fp)

    if visualize:
        render([clustered_cloud.colorize_using_attribute("instance", random=True)])


@cli.command()
def dbh(
    input_fp: Annotated[
        Optional[Path],
        typer.Argument(
            metavar="INPUT",
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Input cloud to process. If passed, will be used if ground segmentation is not being skipped. It is an error to not pass this if output dir is also not specified or the output dir does not contain the appropriate stage to start processing from.",
        ),
    ] = None,
    config_fp: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="a config file which can be used for individual processing steps.",
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            file_okay=False,
            dir_okay=True,
            help="directory used to cache results. intermediate process files like ground segmented cloud, normalized cloud are written here as well. they can be used to skip stages in subsequent runs",
        ),
    ] = None,
    visualize: Annotated[
        bool,
        typer.Option(
            "--visualize",
            "-v",
            help="visualize the cylinder fitting results. Needs input cloud to be passed as well.",
        ),
    ] = False,
    skip_ground_seg: Annotated[
        bool,
        typer.Option(
            "--skip_gs",
            help="Skip ground segmentation. Only skipped if ground_cloud.ply cloud exists inside the output dir.",
        ),
    ] = False,
    skip_normalize: Annotated[
        bool,
        typer.Option(
            "--skip_norm",
            help="Skip height normalization. Only skipped if height_normalized.ply cloud exists inside the output dir. Will not be skipped if ground segmantation has been run in the current process, so you have to specify both options.",
        ),
    ] = False,
    skip_cluster: Annotated[
        bool,
        typer.Option(
            "--skip_cluster",
            help="Skip clustering. Only skipped if clustered_cloud.ply cloud exists inside the output dir. Will not be skipped if earlier stages have been run.",
        ),
    ] = False,
):
    """
    Preprocess the cloud, clip the cloud to relevant height (config settings),
    cluster and then fit cylinders to estimate DBH.
    """
    config: dict = load_yaml_or_json(config_fp) if config_fp else {}  # type:ignore
    if visualize and not input_fp:
        raise Exception(
            "Please pass the path to the input cloud if you would like to visualize the results"
        )
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    # define all intermediate file paths
    gc_fp = output_dir / "ground_cloud.ply" if output_dir else None
    ngc_fp = output_dir / "non_ground_cloud.ply" if output_dir else None
    normalized_fp = output_dir / "normalized_cloud.ply" if output_dir else None
    clustered_fp = output_dir / "clustered_cloud.ply" if output_dir else None
    cylinders_fp = output_dir / "cylinders.pickle" if output_dir else None
    dbh_csv_fp = output_dir / "tree_dbh.csv" if output_dir else None

    # ground seg
    force_process = False
    if skip_ground_seg and gc_fp and gc_fp.exists() and ngc_fp and ngc_fp.exists():
        ground_cloud = Cloud.load(gc_fp)
        non_ground_cloud = Cloud.load(ngc_fp)
    else:
        if input_fp is None:
            if skip_ground_seg:
                raise Exception(
                    "Necessary files for skipping ground segmentation do no exist in the config dir if passed. Please start from processing an input point cloud"
                )
            else:
                raise Exception("Please pass an input point cloud")
        input_cloud: Cloud = Cloud.load(input_fp)
        ground_seg_config = config.get("ground_seg", {})
        ground_cloud, non_ground_cloud = segment_ground(input_cloud, **ground_seg_config)
        force_process = True
        if gc_fp and ngc_fp:
            ground_cloud.write(gc_fp)
            non_ground_cloud.write(ngc_fp)

    # normalization
    if skip_normalize and normalized_fp and normalized_fp.exists() and not force_process:
        normalized_cloud = Cloud.load(normalized_fp)
    else:
        normalization_config = config.get("normalize", {})
        normalized_cloud = normalize_height(
            non_ground_cloud=non_ground_cloud, ground_cloud=ground_cloud, **normalization_config
        )
        force_process = True
        if normalized_fp:
            normalized_cloud.write(normalized_fp)

    # clustering
    cluster_config = config.get("cluster", {})
    if skip_cluster and clustered_fp and clustered_fp.exists() and not force_process:
        clustered_cloud = Cloud.load(clustered_fp)
    else:
        clustered_cloud = _segment_instance(normalized_cloud, **cluster_config)
        if clustered_fp:
            clustered_cloud.write(clustered_fp)

    # cylinder fitting
    from .traits.dbh import estimate_dbh, write_dbh_csv

    fitting_config = config.get("fitting", {})
    voxel_size = fitting_config.get("voxel_size", None) or cluster_config.get("voxel_size", None)
    ransac_iterations = fitting_config.get("ransac_iterations", 10000)
    xyz_dbh_list, cylinders = estimate_dbh(
        clustered_cloud, voxel_size=voxel_size, ransac_iterations=ransac_iterations
    )
    if dbh_csv_fp:
        write_dbh_csv(xyz_dbh_list, dbh_csv_fp)
    if cylinders_fp:
        import pickle

        with cylinders_fp.open("wb") as cfp_file:
            pickle.dump(cylinders, cfp_file)

    if visualize:
        if "input_cloud" not in locals():
            assert input_fp is not None, "input cloud path is needed"
            input_cloud = Cloud.load(input_fp)
        geoms = [input_cloud]  # type: ignore
        geoms.extend(cylinders)
        render([geoms])


if __name__ == "__main__":
    cli()
