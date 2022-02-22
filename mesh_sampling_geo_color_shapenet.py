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

import numpy as np
import os
import glob
import pandas as pd
import argparse
import traceback
import open3d as o3d
import shutil
from tqdm import tqdm
import traceback
import subprocess
import pandas as pd
import pymeshlab


def get_duplicate_faces_to_delete(face_matrix, face_quality_array, threshold=0):

	#Loop through all the faces, identify faces with same vertices, keep only the one with higher 
	#computed value assigned by the ambient occlusion process
	perm_vecs = np.array([[0, 1, 2], [0, 2, 1],[1, 0, 2],[1, 2, 0],[2, 1, 0],[2, 0, 1]])
	already_done = np.zeros(face_matrix.shape[0], dtype=bool)
	to_delete = np.zeros(face_matrix.shape[0], dtype=bool)

	for i, face in enumerate(face_matrix):

		if already_done[i]:
			continue

		#For all possible permutations, identify faces that have same vertices
		index_faces = np.array([], dtype=int)
		for j in range(6):
			duplicate_faces = np.nonzero((face_matrix == face[perm_vecs[j]]).all(axis=1))[0]
			index_faces = np.concatenate((index_faces, duplicate_faces), dtype=int)

		#If there is more than one face with the same value of occlusion, keeps only one
		face_to_keep = index_faces[face_quality_array[index_faces] == face_quality_array[index_faces].max()][0]
		faces_to_delete = index_faces[index_faces != face_to_keep]

		already_done[index_faces] = True
		to_delete[faces_to_delete] = True

	#Delete faces with occlusion value smaller than threshold
	to_delete = to_delete | (face_quality_array < threshold)

	return to_delete

def ensure_face_correspondance(obj_file_content, face_matrix):

	non_empty_lines_obj = [(idx, x) for idx, x in enumerate(obj_file_content) if len(x) > 0]
	
	face_list_obj = [x for idx, x in non_empty_lines_obj if x[0] == 'f']
	face_matrix_obj = [[x.split()[1].split('/')[0], x.split()[2].split('/')[0], x.split()[3].split('/')[0]] for x in face_list_obj]
	face_matrix_obj = np.asarray(face_matrix_obj, dtype=int) - 1


	#If faces from the file don't correspond with face_matrix, it can be due to the fact that some faces are ignored when loading the mesh
	#with pymeshlab. If that is the case, the ignored faces are excluded from obj_file_content here
	if not np.array_equal(face_matrix, face_matrix_obj):
		
		faces_to_delete_first = []
		faces_idx_obj = [idx for idx, x in non_empty_lines_obj if x[0] == 'f']

		#Exclude first face that doesn't correspond from face_matrix_obj, until both matrices have the same size
		while(face_matrix_obj.shape[0] > face_matrix.shape[0]):

			face_correspondance = (face_matrix == face_matrix_obj[:face_matrix.shape[0], :]).all(axis=1)
			
			#Usually, face_correspondance will have a point from which all of the elements are False. This point correspond to the face to be removed
			if not face_correspondance.all():
				face_to_delete = np.nonzero(np.logical_not(face_correspondance))[0][0]
				face_matrix_obj = np.delete(face_matrix_obj, face_to_delete, axis=0)
				faces_to_delete_first.append(faces_idx_obj.pop(face_to_delete))
			#When face_correspondance is all True, the last faces from face_matrix_obj are the ones that don't correspond and must be excluded
			else:
				faces_to_delete_first += faces_idx_obj[face_matrix.shape[0]:]
				face_matrix_obj = face_matrix_obj[:face_matrix.shape[0], :]
		
		#If after the exclusion of these faces they still don't correspond, then the mesh is not sampled
		if not np.array_equal(face_matrix, face_matrix_obj):
			return None
		else:
			obj_file_content = list(np.delete(np.asarray(obj_file_content), faces_to_delete_first))
		
	return obj_file_content

def sample_mesh_set(input_glob, output_dir, occlusion_threshold, target_points, resolution, remove_intermediate_files, cloudcompare_bin_path, tqdm_description="Sampling mesh set"):

	os.makedirs(output_dir, exist_ok=True)

	current_dir = os.getcwd()

	exceptions = []
	mesh_pc_correspondance = {"Input mesh files": [], "Output point cloud files": []}

	mesh_files = glob.glob(input_glob)
	t = tqdm(total=len(mesh_files))
	t.set_description(tqdm_description)
	t.set_postfix({"Errors": len(exceptions)})

	#Go through mesh files liste as input
	for n, input_filepath in enumerate(mesh_files):
		try:
			output_pc_filepath = sample_mesh(input_filepath, n, occlusion_threshold, output_dir, current_dir, target_points, resolution, remove_intermediate_files, cloudcompare_bin_path)
			mesh_pc_correspondance["Input mesh files"].append(input_filepath)
			mesh_pc_correspondance["Output point cloud files"].append(output_pc_filepath)
		except Exception:
			exceptions.append((input_filepath, traceback.format_exc()))
			t.set_postfix({"Errors": len(exceptions)})
		
		t.update()

	return mesh_pc_correspondance, exceptions

def get_mesh_arrays(input_dir, input_filename, face_matrix_file, face_quality_file):

	os.chdir(input_dir)
	ms = pymeshlab.MeshSet()
	ms.load_new_mesh(input_filename)
	
	#Compute ambient occlusion on mesh
	ms.ambient_occlusion(occmode=1)

	occluded_mesh = ms.mesh(0)
	face_matrix = occluded_mesh.face_matrix()
	face_quality_array = occluded_mesh.face_quality_array()

	np.save(face_matrix_file, face_matrix)
	np.save(face_quality_file, face_quality_array)


def sample_mesh(input_filepath, n, occlusion_threshold, output_dir, current_dir, target_points, resolution, remove_intermediate_files, cloudcompare_bin_path):

	if input_filepath[-4:] != ".obj":
		raise Exception("Mesh file doesn't have obj extension and couldn't be sampled")

	input_dir, input_filename = os.path.split(input_filepath)

	face_matrix_file = "face_matrix.npy"
	face_quality_file = "face_quality.npy"

	#Load mesh and compute ambiance occlusion 
	subprocess.run(['python', '-c', f'from mesh_sampling_geo_color_shapenet import get_mesh_arrays; get_mesh_arrays(\"{input_dir}\", \"{input_filename}\", \"{face_matrix_file}\", \"{face_quality_file}\")'], capture_output=True)

	os.chdir(input_dir)

	face_matrix = np.load(face_matrix_file)
	face_quality_array = np.load(face_quality_file)

	os.remove(face_matrix_file)
	os.remove(face_quality_file)

	os.chdir(current_dir)

	#Gets indices of duplicate faces to be deleted
	faces_to_delete = get_duplicate_faces_to_delete(face_matrix, face_quality_array, occlusion_threshold)

	#Reads the original mesh file
	with open(input_filepath, 'r') as f:
		obj_file_content = f.read().splitlines()

	#Make sure faces from the file and from face_matrix correspond
	obj_file_content = ensure_face_correspondance(obj_file_content, face_matrix)

	if obj_file_content == None:
		raise Exception("Unable to sample mesh: faces from loaded mesh are different than obj file.")

	#Delete duplicate faces indicated in faces_to_delete
	non_empty_lines_obj = [(idx, x) for idx, x in enumerate(obj_file_content) if len(x) > 0]
	faces_idx_obj = [idx for idx, x in non_empty_lines_obj if x[0] == 'f']
	obj_lines_to_delete = np.asarray(faces_idx_obj)[faces_to_delete]
	obj_file_content = np.delete(np.asarray(obj_file_content), obj_lines_to_delete)

	#Create directory to copy the produced meshes
	mesh_clean_dir = f"clean_mesh_{n}"
	mesh_clean_dir = os.path.join(output_dir, mesh_clean_dir)
	obj_dir = os.path.join(mesh_clean_dir, "models")
	os.makedirs(obj_dir, exist_ok=True)
	
	#Get mtl file referenced in the obj file and copy it to the destination
	mtl_line_info = [(idx, x) for idx, x in enumerate(obj_file_content) if 'mtllib' in x][0]
	mtl_line = mtl_line_info[1].split()
	mtl_file = [x for x in mtl_line if x[-4:] == ".mtl"][0]
	mtl_idx = mtl_line_info[0]
	mtl_orig_filepath = os.path.join(input_dir, mtl_file)

	mesh_clean_filename_mtl = f"clean_mesh_{n}.mtl"
	mtl_dest_filepath = os.path.join(obj_dir, mesh_clean_filename_mtl)
	shutil.copyfile(mtl_orig_filepath, mtl_dest_filepath)

	#Change the name of the mtl file referenced in the obj file to the new name
	mtl_new_line = f"mtllib {mesh_clean_filename_mtl}"
	obj_file_content[mtl_idx] = mtl_new_line

	#Save the new obj file to the destination
	obj_file_content = "\n".join(obj_file_content)
	mesh_clean_filename = f"clean_mesh_{n}.obj"
	mesh_clean_filepath = os.path.join(obj_dir, mesh_clean_filename)
	with open(mesh_clean_filepath, 'w') as f:
		f.write(obj_file_content)

	#Go through all the texture files listed in the mtl file and copy them to the destination
	with open(mtl_dest_filepath, 'r') as f:
		mtl_content = f.read().splitlines()

	texture_file_lines = [x for x in mtl_content if "map_Kd" in x]
	texture_files = [x.split()[1] for x in texture_file_lines]

	for texture_file in texture_files:
		texture_file_rel_dir, texture_filename = os.path.split(texture_file)
		texture_file_dest_dir = os.path.join(obj_dir, texture_file_rel_dir)
		os.makedirs(texture_file_dest_dir, exist_ok=True)
		texture_file_dest_path = os.path.join(texture_file_dest_dir, texture_filename)
		texture_file_ori_path = os.path.join(input_dir, texture_file)
		shutil.copyfile(texture_file_ori_path, texture_file_dest_path)

	#Sample the generated mesh with CloudCompare
	sampled_mesh_filename = f"sampled_mesh_{n}.ply"
	sampled_mesh_filepath = os.path.join(output_dir, sampled_mesh_filename)
	os.popen(f"{cloudcompare_bin_path} -SILENT -AUTO_SAVE OFF -C_EXPORT_FMT PLY -O {mesh_clean_filepath} -SAMPLE_MESH POINTS {target_points} -SAVE_CLOUDS").read()

	#Get name of the point cloud saved by CloudCompare and rename it to defined name
	cloudcompare_pc_path = [x for x in glob.glob(os.path.join(obj_dir, "*.ply")) if "SAMPLED_POINTS" in os.path.split(x)[1]][0]
	os.rename(cloudcompare_pc_path, sampled_mesh_filepath)

	#Voxelize the point cloud with the target resolution after placing it in a bounding box
	o3d_pc = o3d.io.read_point_cloud(sampled_mesh_filepath)

	o3d_pc.translate(-o3d_pc.get_min_bound())
	o3d_pc.scale((resolution-1)/o3d_pc.get_max_bound().max(), [0, 0, 0])
	o3d_pc = o3d_pc.voxel_down_sample(voxel_size=1)

	points = np.asarray(o3d_pc.points)
	points = np.round(points)
	o3d_pc.points = o3d.utility.Vector3dVector(points)
	
	#Save the voxelized point cloud
	output_pc_filename = f"vox_sampled_mesh_{n}.ply"
	output_pc_filepath = os.path.join(output_dir, output_pc_filename)
	o3d.io.write_point_cloud(output_pc_filepath, o3d_pc)

	#Remove produced mesh and intermediary sampled point cloud
	if remove_intermediate_files:
		shutil.rmtree(mesh_clean_dir)
		os.remove(sampled_mesh_filepath)

	return output_pc_filepath

def run(args):

	print("Starting mesh sampling...")

	mesh_pc_correspondance, exceptions = sample_mesh_set(args.input_glob, args.output_dir, args.occlusion_threshold, 
		args.target_points, args.resolution, args.remove_intermediate_files, args.cloudcompare_bin_path)

	print("The mesh sampling process is finished!")

	if len(exceptions) > 0:
		
		exception_string = "\n".join([f"\nMesh file: {x[0]}\nException: {x[1]}" for x in exceptions])
		with open(args.error_log_file, 'w') as f:
			f.write(exception_string)
		print(f"A total of {len(exceptions)} were not correctly sampled. Check {args.error_log_file} for more details on the errors found.")
		
	output_table = pd.DataFrame(mesh_pc_correspondance)
	output_table.to_csv(args.output_csv)


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(prog='mesh_sampling_geo_color.py', description='Samples color and geometry from meshes using random sampling and voxelizes the generated point clouds into a regular grid.',
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('input_glob', help='Pattern for input meshes.')
	parser.add_argument('output_dir', help='Output directory for saving sampled point clouds.')
	parser.add_argument('--target_points', help='Number of points sampled before voxelization.', type=int, default=10000000)
	parser.add_argument('--resolution', help='Resolution for voxelization.', type=int,default=1024)
	parser.add_argument('--occlusion_threshold', help='Minimum accepted occlusion value for sampled faces.', type=float,default=0)
	parser.add_argument('--remove_intermediate_files', help='Remove meshes and point clouds generated on the process.', type=bool,default=True)
	parser.add_argument('--cloudcompare_bin_path', help='Path to the CloudCompare binary', type=str,default="CloudCompare")
	parser.add_argument('--output_csv', help='Path to output csv file containing the correspondance between input mesh files and output point cloud files.', type=str,default="sampling.csv")
	parser.add_argument('--error_log_file', help='Path to error log file.', type=str,default="error.log")
	args = parser.parse_args()

	run(args)
