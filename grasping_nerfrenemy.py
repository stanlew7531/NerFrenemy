from __future__ import print_function

import numpy as np
import argparse
import grasping.pytorch_6dof_graspnet.grasp_estimator as grasp_estimator
import sys
import os
import glob
import mayavi.mlab as mlab
from grasping.pytorch_6dof_graspnet.utils.visualization_utils import *
import mayavi.mlab as mlab
from grasping.pytorch_6dof_graspnet.utils import utils
from grasping.pytorch_6dof_graspnet.data import DataLoader















# import sys
# sys.path.insert(1, './PoseCNN-PyTorch/lib')
# from datasets.factory import get_dataset
# from fcn.config import cfg, cfg_from_file, yaml_from_file, get_output_dir
# from fcn.test_imageset import test_image
# from ycb_renderer import YCBRenderer
# from get_available_devices import *
import torch
import numpy as np
import torch
import torch.nn.parallel 
import torch.backends.cudnn as cudnn 
# import networks
# from utils.nms import *
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [18, 9]
# from utils.blob import pad_im, chromatic_transform, add_noise
from transforms3d.quaternions import mat2quat, quat2mat
# from utils.se3 import *
# from utils.pose_error import *
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


from ProgressNerf.Dataloading.YCBVideoDataloader import YCBVideoDataloader
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
# from fcn.train import loss_cross_entropy, smooth_l1_loss


import cv2
from skimage import measure
import scipy.ndimage as ndimage 
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import (morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient)





def make_parser():
    parser = argparse.ArgumentParser(
        description='6-DoF GraspNet Demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--grasp_sampler_folder',
                        type=str,
                        default='grasping/pytorch_6dof_graspnet/checkpoints/gan_pretrained/')
    parser.add_argument('--grasp_evaluator_folder',
                        type=str,
                        default='grasping/pytorch_6dof_graspnet/checkpoints/evaluator_pretrained/')
    parser.add_argument('--refinement_method',
                        choices={"gradient", "sampling"},
                        default='sampling')
    parser.add_argument('--refine_steps', type=int, default=25)

    parser.add_argument('--npy_folder', type=str, default='demo/data/')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.8,
        help=
        "When choose_fn is something else than all, all grasps with a score given by the evaluator notwork less than the threshold are removed"
    )
    parser.add_argument(
        '--choose_fn',
        choices={
            "all", "better_than_threshold", "better_than_threshold_in_sequence"
        },
        default='better_than_threshold',
        help=
        "If all, no grasps are removed. If better than threshold, only the last refined grasps are considered while better_than_threshold_in_sequence consideres all refined grasps"
    )

    parser.add_argument('--target_pc_size', type=int, default=1024)
    parser.add_argument('--num_grasp_samples', type=int, default=200)
    parser.add_argument(
        '--generate_dense_grasps',
        action='store_true',
        help=
        "If enabled, it will create a [num_grasp_samples x num_grasp_samples] dense grid of latent space values and generate grasps from these."
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=30,
        help=
        "Set the batch size of the number of grasps we want to process and can fit into the GPU memory at each forward pass. The batch_size can be increased for a GPU with more memory."
    )
    parser.add_argument('--train_data', action='store_true')
    opts, _ = parser.parse_known_args()
    if opts.train_data:
        parser.add_argument('--dataset_root_folder',
                            required=True,
                            type=str,
                            help='path to root directory of the dataset.')
    return parser


def get_color_for_pc(pc, K, color_image):
    proj = pc.dot(K.T)
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]

    pc_colors = np.zeros((pc.shape[0], 3), dtype=np.uint8)
    for i, p in enumerate(proj):
        x = int(p[0])
        y = int(p[1])
        pc_colors[i, :] = color_image[y, x, :]

    return pc_colors


def backproject(depth_cv,
                intrinsic_matrix,
                return_finite_depth=True,
                return_selection=False):

    depth = depth_cv.astype(np.float32, copy=True)

    # get intrinsic matrix
    K = intrinsic_matrix
    Kinv = np.linalg.inv(K)

    # compute the 3D points
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)

    # backprojection
    R = np.dot(Kinv, x2d.transpose())

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R)
    X = np.array(X).transpose()
    if return_finite_depth:
        selection = np.isfinite(X[:, 0])
        X = X[selection, :]

    if return_selection:
        return X, selection

    return X


def getVisualizedImg(sample_img):
    return (sample_img[0].permute(1,2,0) + 1.0 / 2.0).cpu().numpy()

def run_net(network, dataset, im_color, device):
    im_color.to(device)
    
    K = dataset._intrinsic_matrix
    K[2, 2] = 1
    Kinv = np.linalg.pinv(K)
    meta_data = np.zeros((1, 18), dtype=np.float32)
    meta_data[0, 0:9] = K.flatten()
    meta_data[0, 9:18] = Kinv.flatten()
    meta_data = torch.from_numpy(meta_data).to(device).contiguous()
    
    out_prob, out_label, out_vertex, rois, out_pose, out_quaternion =\
        network(im_color, dataset.input_labels.contiguous(), meta_data, \
            dataset.input_extents.contiguous(), dataset.input_gt_boxes.contiguous(),\
            dataset.input_poses.contiguous(), dataset.input_points.contiguous(),\
            dataset.input_symmetry.contiguous())
    
    return out_prob, out_label, out_vertex, rois, out_pose, out_quaternion

def get_depth_rgb_from_render(render_result, num_cameras, width, height):
    rgb_output = render_result['rgb']\
        .reshape((num_cameras, width, height, 3)).transpose(1,2).contiguous()
    depth_output = render_result['depth']\
        .reshape((num_cameras, width, height, 1)).transpose(1,2).contiguous()
    acc_output = render_result['acc']\
        .reshape((num_cameras, width, height, 1)).transpose(1,2).contiguous()
    return rgb_output, depth_output, acc_output

def sample_on_cylinder(height_mm=100, radius_mm=33):
    samples = []
    theta = 0
    z_inc = 0.5
    z = -height_mm/2
    # sample points in cylinder frame
    while z < height_mm/2:
        while theta < 2 * np.pi:
            x = radius_mm * np.cos(theta)
            y = radius_mm * np.sin(theta)
            point = np.array([x, y, z])
            samples.append(point)
            theta += np.pi/6
        theta = 0
        z+=z_inc
    return samples

def sample_from_voxel_grid(grid):
    positive_values = grid.voxels[(grid.voxels[...,-1:].relu() > 0).squeeze()]
    return positive_values



def main(args):
    parser = make_parser()
    args = parser.parse_args()
    grasp_sampler_args = utils.read_checkpoint_args(args.grasp_sampler_folder)
    grasp_sampler_args.is_train = False
    grasp_evaluator_args = utils.read_checkpoint_args(
        args.grasp_evaluator_folder)
    grasp_evaluator_args.continue_train = True
    grasp_sampler_args.checkpoints_dir = './grasping/pytorch_6dof_graspnet/checkpoints'
    grasp_evaluator_args.checkpoints_dir = './grasping/pytorch_6dof_graspnet/checkpoints'
    estimator = grasp_estimator.GraspEstimator(grasp_sampler_args,
                                               grasp_evaluator_args, args)

    args.npy_folder = './grasping/pytorch_6dof_graspnet/demo/data'



    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    base_dir = "./output_data/"
    if(not os.path.exists(base_dir)):
                os.makedirs(base_dir)
    # tb_writer = SummaryWriter(log_dir=str(base_dir))


    config_file = "./soup_can/ycbVideo_Soup.yml"
    arch = VoxelGridCachedNerf(config_file)

    # select the UVW (*not* s) parameters that correspond to the can's geometry
    # get the xyz locations from the Can sampler
    an_samples = torch.from_numpy(np.array(sample_on_cylinder())).to(estimator.device) / 1000.0 # convert to meters

    config_dict = { "baseDataDir": "/home/logan/Desktop/NerFrenemy/input_data/ycb-video/tool-parts_dataset/",\
                   "scenes": ["0003"],\
                   "datasetType": "train",\
                   "samplesLimit": -1,\
                   "rel_tools": ["04"]}
    progNerfDataloader = torch.utils.data.DataLoader(YCBVideoDataloader(config_dict),\
                                                     batch_size = 1,\
                                                     shuffle=True,\
                                                     num_workers = 1)

    with torch.no_grad():
        for batch_ndx, sample in tqdm(enumerate(iter(progNerfDataloader))):
            #print(sample.keys())
            gt_seg = sample['segmentation'].to(estimator.device)
            image_color = sample['image'].contiguous()
            plt.imshow(image_color[0].cpu())
            plt.show()
            sample_pose = torch.linalg.inv(sample['04_pose'])
            render_result = arch.doFullRender(sample_pose, use_cache=False)
            nerf_rgb, nerf_depth, nerf_acc = get_depth_rgb_from_render(render_result, 1, 640, 480)
            depth = nerf_depth[0,:,:,0].cpu().detach().numpy()
            image = (nerf_rgb[0].cpu().detach().numpy()*255).astype(np.uint8)

            grayscale = rgb2gray(image)
            grayscale = gaussian(grayscale, sigma=3, channel_axis=-1)
            gimage = inverse_gaussian_gradient(grayscale)

            # Initial level set
            init_ls = np.zeros(grayscale.shape, dtype=np.int8)
            init_ls[10:-10, 10:-10] = 1
            # List with intermediate results for plotting the evolution
            ls = morphological_geodesic_active_contour(gimage, num_iter=250,
                                                       init_level_set=init_ls,
                                                       smoothing=1, balloon=-1.3,
                                                       threshold=0.98)
            print(ls.shape)

            contours = measure.find_contours(ls, 0.5)

            mx = [0,0]
            for i,contour in enumerate(contours):
                c = np.expand_dims(contour.astype(np.float32), 1)
                # Convert it to UMat object
                c = cv2.UMat(c)
                area = cv2.contourArea(c)
                if area>mx[0]:
                    mx = [area,i]
            
        
            contour = contours[mx[1]]
            r_mask = np.zeros_like(grayscale, dtype='bool')
            r_mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
            r_mask = ndimage.binary_fill_holes(r_mask)


            depth = depth * r_mask
            print(np.mean(depth[r_mask]))
            plt.imshow(depth<np.mean(depth[r_mask])-0.01)
            plt.show()
            depth[depth<np.mean(depth[r_mask])-0.01] = 0
            
            acc = nerf_acc[0,:,:,0].cpu().detach().numpy()
            plt.axis('off')
            plt.imshow(nerf_rgb[0].detach().cpu())
            plt.show()
            plt.imshow(depth)
            plt.show()
            plt.imshow(nerf_acc[0,:,:,0].detach().cpu())
            plt.show()
            plt.imshow(r_mask)
            plt.show()

            K = arch.cam_matrix.numpy()
            
            # Removing points that are farther than 1 meter or missing depth
            # values.
            #depth[depth == 0 or depth > 1] = np.nan

            np.nan_to_num(depth, copy=False)
            mask = np.where(np.logical_or(depth == 0, depth > 1))
            depth[mask] = np.nan
            pc, selection = backproject(depth,
                                        K,
                                        return_finite_depth=True,
                                        return_selection=True)
            pc_colors = image.copy()
            pc_colors = np.reshape(pc_colors, [-1, 3])
            pc_colors = pc_colors[selection, :]

            # Smoothed pc comes from averaging the depth for 10 frames and removing
            # the pixels with jittery depth between those 10 frames.
            #object_pc = data['smoothed_object_pc']
            object_pc = pc
            generated_grasps, generated_scores = estimator.generate_and_refine_grasps(
                object_pc)
            mlab.figure(bgcolor=(1, 1, 1))
            draw_scene(
                pc,
                pc_color=pc_colors,
                grasps=generated_grasps,
                grasp_scores=generated_scores,
            )
            print('close the window to continue to next object . . .')
            mlab.show()

            break
            raise Exception("asdf")

    

if __name__ == '__main__':
    main(sys.argv[1:])
