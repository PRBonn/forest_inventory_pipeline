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

from ..cloud import Cloud
from scipy.spatial import KDTree
from .rich_console import logger
from .decorators import log_scalars_in_arguments


@log_scalars_in_arguments()
def upsample(low_res_cloud: Cloud, high_res_cloud: Cloud, low_res_voxel_size=0.15):
    # fit a KDTree to the low_res cloud
    kdtree = KDTree(low_res_cloud.points, copy_data=False)
    # find for each point in the high_res cloud, the closest low_res point
    distances, indices = kdtree.query(
        high_res_cloud.points,
        k=1,
        distance_upper_bound=2 * low_res_voxel_size,
        workers=-1,
    )
    invalids_exist = False
    if np.any(np.isinf(distances)):
        logger.info("invalids exist")
        invalids_exist = True
        mask = np.isinf(distances)
        distances[mask] = 0
    # get the valid points within the search range
    assert len(indices) == high_res_cloud.num_points
    valid_indices = indices[distances <= 2 * low_res_voxel_size]
    if invalids_exist:
        valid_indices[mask] = 0
    assert len(valid_indices) == len(indices)
    upsampled_instances = low_res_cloud.instance[np.squeeze(valid_indices)].astype(np.int32)
    if invalids_exist:
        upsampled_instances[mask] = -3
    upsampled_cloud = high_res_cloud.select_by_mask(distances <= 2 * low_res_voxel_size)
    assert len(upsampled_instances) == upsampled_cloud.num_points
    upsampled_cloud.instance = upsampled_instances
    return upsampled_cloud
