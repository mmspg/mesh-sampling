# Mesh sampling script 

This repository includes two scripts that perform mesh sampling. They were developped with the goal of generating point cloud datasets for training of learning-based compression algorithms from large-scale mesh datasets such as ModelNet [1] and ShapeNet [2]. 

## Script for sampling geometry-only point clouds

The first script *mesh_sampling_geo.py* was adapted from code developed by the authors of [3]. It samples random geometric positions from the surface of a mesh. The obtained coordinates are then scaled to a fit a cubic bounding box and quantized. 

The script was tested in an environment with the following specifications: 

* Ubuntu 20.04
* Python 3.6

Prior to running the script, make sure that the following python libraries are installed: 

* pandas ~=1.1.5
* pyntcloud ~= 0.1.5

The script can be run with the following command:

```shell
python mesh_sampling_geo.py input_glob output_dir
```

The script accepts the following parameters:

* *input_glob* : pattern identifying the mesh files to be sampled.
* *output_dir* : directory where the generated point clouds are saved. 
* *--target_points* : target number of points for the sampled point cloud prior to voxelization. 
* *--resolution* : resolution of the output voxelized point cloud. 

## Script for sampling point clouds with geometry and texture

The second script *mesh_sampling_geo_color_shapenet.py* was developed to sample color and geometry from the external faces of the ShapeNet dataset. Although directly sampling the color and geometry from meshes from the ShapeNet dataset is possible, the visual result doesn't always correspond to what is observed in the external faces of these meshes. The reason of this result is because these meshes are often defined with duplicated faces sharing the same vertices, but with opposing normal vectors and sometimes different color values as well. Although mesh rendering software often don't display faces with normal vectors not facing the camera in a process known as back face culling, sampled points will carry the color of both faces at random positions over the same surface. In order to avoid this undesired effect, the proposed method detects and excludes the internal faces of the mesh using an ambient occlusion plugin. In this way, a mesh model with only the faces visible from an outside observer are used as input to the sampling algorithm. The interested reader can refer to [4] for more details. 

This script was tested in the same environment as the previous geometry-only sampling algorithm. Prior to running this script, make sure that the following python libraries are installed: 

* pymeshlab ~= 2021.10
* pandas ~= 1.1.5
* open3d ~= 0.11.2

CloudCompare software must also be installed in the computer. Additionally, if the script is being run on a headless server, be sure to install *xvfb* in order to run CloudCompare without a graphic interface. The script can be run using the following command:

```shell
python mesh_sampling_geo_color_shapenet.py input_glob output_dir
```

The script accepts the following parameters:

* *input_glob* : pattern identifying the mesh files to be sampled.
* *output_dir* : directory where the generated point clouds are saved. 
* *--target_points* : target number of points for the sampled point cloud prior to voxelization. 
* *--resolution* : resolution of the output voxelized point cloud. 
* *--remove_intermediate_files* : if set to True, the intermediary mesh and point cloud files generated on the process are removed. 
* *--cloudcompare_bin_path* : path to the CloudCompare binary. If the location of the binary is included in the PATH environment variable, this configuration attribute can be set of "CloudCompare". If the script is being run in a headless server, include the command *xvfb-run* in the beginning of this attribute. 

## Conditions of use

If you wish to use the provided script in your research, we kindly ask you to cite [4].

## Reference

[1]  Wu, Zhirong, Shuran Song, Aditya Khosla, Fisher Yu, Linguang Zhang, Xiaoou Tang, and Jianxiong Xiao. 2015. “3D ShapeNets: A Deep Representation for Volumetric Shapes.” *Proceedings of 28th IEEE Conference on Computer Vision and Pattern Recognition (CVPR2015)*

[2] Chang, Angel X., Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, et al. 2015. “ShapeNet: An Information-Rich 3D Model Repository.” *ArXiv:1512.03012 [Cs]*, December. http://arxiv.org/abs/1512.03012.

[3] Quach, M., G. Valenzise, and F. Dufaux. 2020. “Improved Deep Point Cloud Geometry Compression.” In *2020 IEEE 22nd International Workshop on Multimedia Signal Processing (MMSP)*, 1–6. https://doi.org/10.1109/MMSP48831.2020.9287077.

[4] Davi Lazzarotto and Touradj Ebrahimi. 2022. “Sampling color and geometry point clouds from ShapeNet dataset.” https://arxiv.org/abs/2201.06935

