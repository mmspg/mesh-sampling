# Copyright (C) 2022 ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland
#
#     Multimedia Signal Processing Group (MMSPG)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.

# This code was adapted from https://github.com/mauriceqch/pcc_geo_cnn_v2

import os
import glob
import time
from pyntcloud import PyntCloud
import numpy as np
import pandas as pd
import argparse

def run(args):

	os.makedirs(args.output_dir, exist_ok=True)

	for i, input_file in enumerate(glob.glob(args.input_glob)):

		pc_mesh = PyntCloud.from_file(input_file)
		mesh = pc_mesh.mesh
		pc_mesh.points = pc_mesh.points.astype('float64', copy=False)
		pc_mesh.mesh = mesh
		pc = pc_mesh.get_sample("mesh_random", n=args.target_points, as_PyntCloud=True)
		coords = ['x', 'y', 'z']
		points = pc.points.values
		points = points - points.min(axis=0)
		points = points / np.max(points)
		points = points * (args.resolution - 1)
		points = np.round(points)
		points = np.unique(points, axis=0) 

		points_df = pd.DataFrame(data=points, columns=coords)
		pc = PyntCloud(points_df)

		target_path = os.path.join(args.output_dir, f"random_sampling_vox_{i}.ply")
		pc.to_file(target_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog='mesh_sampling_geo.py', description='Samples geometric meshes using random sampling and voxelizes the generated point clouds into a regular grid.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_glob', help='Pattern for input meshes.')
    parser.add_argument('output_dir', help='Output directory for saving sampled point clouds.')
    parser.add_argument('--target_points', help='Number of points sampled before voxelization.', type=int, default=4000000)
    parser.add_argument('--resolution', help='Resolution for voxelization.', type=int, default=1024)
    args = parser.parse_args()

    run(args)



	