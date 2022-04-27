import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import ProgressNerf.Dataloading.YCBVideoDataloader
# import ProgressNerf.Dataloading.ToolsPartsDataloader

import ProgressNerf.Raycasting.VoxelGridBBoxRaysampler
import ProgressNerf.Raycasting.RandomRaypicker
import ProgressNerf.Raycasting.NearFarRaysampler
import ProgressNerf.Raycasting.WeightedRaypicker
import ProgressNerf.Raycasting.PerturbedRaysampler

import ProgressNerf.NeuralRendering.NeuralRenderer
# import ProgressNerf.NeuralRendering.VoxelNeuralRenderer

import ProgressNerf.Encoders.PositionalEncoder

# import ProgressNerf.Models.OGNerf
# from ProgressNerf.Architectures.OGNerfArch import OGNerfArch
import ProgressNerf.Models.FastNerf
from ProgressNerf.Architectures.VoxelGridCachedNerf import VoxelGridCachedNerf

from ProgressNerf.Utils.CameraUtils import BuildCameraMatrix

# config_file = "./ycbVideo_Soup_OG.yml"
# arch = OGNerfArch(config_file)

config_file = "./ycbVideo_Soup.yml"
arch = VoxelGridCachedNerf(config_file)


test_pose = np.eye(4)
# just a random pose taken from the ToolParts dataset (scene 00_01, test set, sample #8)
test_pose[0:3,3] = np.array([0.052757302998853915, -0.2394366825095671, 2.0026748131467889])
pose_quat = np.array([0.28405447470305767, 0.7107763388551345, 0.5857394722169705, -0.26649450690745147])
rot_matrix = R.from_quat(np.concatenate((pose_quat[1:4], [pose_quat[0]]))).as_matrix()
test_pose[0:3,0:3] = rot_matrix
test_cam_pose = np.linalg.inv(test_pose)

eval_poses = torch.cat((torch.Tensor(test_cam_pose).unsqueeze(0),), dim = 0)

renderings = arch.doEvalRendering(eval_poses)
plt.imshow(renderings[0].squeeze().cpu().numpy())
plt.show()
