
## DATASET AND CFG SETUP ##############################################################################
import sys
sys.path.insert(1, './PoseCNN-PyTorch/lib')
from datasets.factory import get_dataset
from fcn.config import cfg, cfg_from_file, yaml_from_file, get_output_dir
from ycb_renderer import YCBRenderer
import torch
import numpy as np

cfg_from_file("./PoseCNN-PyTorch/experiments/cfgs/ycb_object.yml")
meta = yaml_from_file("./PoseCNN-PyTorch/data/demo/meta.yml")
# dataset = get_dataset("ycb_object_train")
dataset = get_dataset("ycb_object_test")




cfg.renderer = YCBRenderer(width=cfg.TRAIN.SYN_WIDTH, height=cfg.TRAIN.SYN_HEIGHT, gpu_id=0, render_marker=False)
if cfg.TEST.SYNTHESIZE:
    cfg.renderer.load_objects(dataset.model_mesh_paths, dataset.model_texture_paths, dataset.model_colors)
else:
    model_mesh_paths = [dataset.model_mesh_paths[i-1] for i in cfg.TEST.CLASSES[1:]]
    model_texture_paths = [dataset.model_texture_paths[i-1] for i in cfg.TEST.CLASSES[1:]]
    model_colors = [dataset.model_colors[i-1] for i in cfg.TEST.CLASSES[1:]]
    cfg.renderer.load_objects(model_mesh_paths, model_texture_paths, model_colors)

cfg.renderer.set_camera_default()
cfg.TEST.POSE_REFINE = False
cfg.TEST.VISUALIZE = False 

cfg.MODE = 'TRAIN'

cfg.gpu_id = 0
cfg.device = torch.device('cuda:{:d}'.format(cfg.gpu_id))
cfg.instance_id = 0









# Setup Dataloader
worker_init_fn = dataset.worker_init_fn if hasattr(dataset, 'worker_init_fn') else None
num_workers = 0 if cfg.TRAIN.SYNTHESIZE else 4
dataloader = torch.utils.data.DataLoader(dataset, 
                    batch_size=1,  #cfg.TRAIN.IMS_PER_BATCH, 
                    shuffle=True, 
                    num_workers=num_workers, 
                    worker_init_fn=worker_init_fn)


# Set meta_data 
K = dataset._intrinsic_matrix
K[2, 2] = 1
Kinv = np.linalg.pinv(K)
meta_data = np.zeros((1, 18), dtype=np.float32)
meta_data[0, 0:9] = K.flatten()
meta_data[0, 9:18] = Kinv.flatten()
meta_data = torch.from_numpy(meta_data).cuda()






## Network SETUP ##############################################################################

import torch
import torch.nn.parallel 
import torch.backends.cudnn as cudnn 
import networks
from utils.nms import *

pretrained = "./PoseCNN-PyTorch/data/checkpoints/ycb_object/vgg16_ycb_object_self_supervision_epoch_8.checkpoint.pth"
# cfg.TRAIN.FREEZE_LAYERS = False
network_data = torch.load(pretrained)
network = networks.__dict__["posecnn"](dataset.num_classes, 64, network_data).cuda()
network = torch.nn.DataParallel(network, device_ids=[cfg.gpu_id]).cuda()
cudnn.benchmark = True

network.eval()











## UNUSED CYLINDER SAMPLING CODE ##############################################################################
# - Finds the pose in the ycb data
# - splits Cameron's cylinder sampling code


import matplotlib.pyplot as plt
from utils.blob import pad_im, chromatic_transform, add_noise
from transforms3d.quaternions import mat2quat, quat2mat
from utils.se3 import *
from utils.pose_error import *

# ycb_datapoint = dataset.__getitem__(30)
ycb_datapoint = next(iter(dataloader))

can_index = 4
ycb_img = ycb_datapoint['image_color']  #.clone().detach().cpu().permute(1,2,0).numpy()
can_mask = ycb_datapoint['label'][0, can_index].int()
can_pose = ycb_datapoint['poses'][0, can_index]

# NOTE:
# data poses have: [0]->obj existence | [1]->object class id | 2:6]->quaternion | [6:9]->offset
can_rot = quat2mat(can_pose[2:6].flatten())
can_trans = can_pose[6:] * 1000


def sample_on_cylinder(height_mm=100, radius_mm=67):
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

class TransformFunctor:
    def __init__(self, rotation_mat, trans_vec):
        self.trans, self.rot = trans_vec, rotation_mat
    def __call__(self, sample):
        return self.rot @ (sample + self.trans)

samples = map(TransformFunctor(can_rot, can_trans), sample_on_cylinder())























## SETS UP CAMPBEL SOUP CAN NERF ##############################################################################


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

def toSE3Matrix(rotation, offset):
    mat = np.eye(4)
    if len(rotation) == 4:
        mat[0:3,0:3] = R.from_quat(np.concatenate((rotation[1:4], [rotation[0]]))).as_matrix()
    elif len(rotation.shape) == 2:
        mat[0:3,0:3] = rotation
    mat[0:3,3] = np.array(offset)
    return mat

def get_test_pose():    # TODO: remove
    quat = np.array([0.28405447470305767, 0.7107763388551345, 0.5857394722169705, -0.26649450690745147])
    offset = np.array([0.052757302998853915, -0.2394366825095671, 2.0026748131467889])
    return toSE3Matrix(quat, offset)
    
test_pose = get_test_pose()

# TODO: render the view based on object pose in ycb data

# test_pose = toSE3Matrix(can_rot, can_trans)
test_cam_pose = np.linalg.inv(test_pose)

test_poses = torch.cat((torch.Tensor(test_cam_pose).unsqueeze(0),), dim = 0)

# renderings = arch.doEvalRendering(test_poses)
renderings = arch.doFullRender(test_poses, use_cache=True)

nerf_rgb = renderings["rgb"].reshape(1, 480, 640, 3)

# plt.imshow(nerf_rgb .squeeze().cpu().numpy())
# plt.show()














## SELECT INPUT TO PASS TO POSECNN ##############################################################################



inputs = nerf_rgb.permute(0, 3, 1, 2).contiguous()
# inputs = ycb_img







## RUN POSECNN ##############################################################################
from fcn.train import loss_cross_entropy, smooth_l1_loss

# TODO: pad and rescale image???

def run_net(network, inputs, sample, meta_data, dataset):
    # # param setting
    # labels =  dataset.input_labels
    # extents = dataset.input_extents
    # gt_boxes = dataset.input_gt_boxes
    # poses = dataset.input_poses
    # points = dataset.input_points
    # symmetry = dataset.input_symmetry
    # if cfg.TRAIN.VERTEX_REG:
    #     vertex_targets = sample['vertex_targets'].cuda()
    #     vertex_weights = sample['vertex_weights'].cuda()
    # else:
    #     vertex_targets = []
    #     vertex_weights = []
    # return network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)


    # prepare data
    inputs.cuda()
    # inputs = sample['image_color'].cuda()
    im_info = sample['im_info']
    mask = sample['mask'].cuda()
    labels = sample['label'].cuda()
    meta_data = sample['meta_data'].cuda()
    extents = sample['extents'][0, :, :].repeat(cfg.TRAIN.GPUNUM, 1, 1).cuda()
    gt_boxes = sample['gt_boxes'].cuda()
    poses = sample['poses'].cuda()
    points = sample['points'][0, :, :, :].repeat(cfg.TRAIN.GPUNUM, 1, 1, 1).cuda()
    symmetry = sample['symmetry'][0, :].repeat(cfg.TRAIN.GPUNUM, 1).cuda()
    if cfg.TRAIN.VERTEX_REG:
        vertex_targets = sample['vertex_targets'].cuda()
        vertex_weights = sample['vertex_weights'].cuda()
    else:
        vertex_targets = []
        vertex_weights = []
    return network(inputs, labels, meta_data, extents, gt_boxes, poses, points, symmetry)

def get_loss(network_results):
    # compute output
    if cfg.TRAIN.VERTEX_REG:
        if cfg.TRAIN.POSE_REG:
            out_logsoftmax, out_weight, out_vertex, out_logsoftmax_box, \
                bbox_labels, bbox_pred, bbox_targets, bbox_inside_weights, loss_pose_tensor, poses_weight \
                = network_results

            loss_label = loss_cross_entropy(out_logsoftmax, out_weight)
            loss_vertex = cfg.TRAIN.VERTEX_W * smooth_l1_loss(out_vertex, vertex_targets, vertex_weights)
            loss_box = loss_cross_entropy(out_logsoftmax_box, bbox_labels)
            loss_location = smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights)
            loss_pose = torch.mean(loss_pose_tensor)
            loss = loss_label + loss_vertex + loss_box + loss_location + loss_pose
        else:
            out_logsoftmax, out_weight, out_vertex, out_logsoftmax_box, \
                bbox_labels, bbox_pred, bbox_targets, bbox_inside_weights \
                = network_results

            loss_label = loss_cross_entropy(out_logsoftmax, out_weight)
            loss_vertex = cfg.TRAIN.VERTEX_W * smooth_l1_loss(out_vertex, vertex_targets, vertex_weights)
            loss_box = loss_cross_entropy(out_logsoftmax_box, bbox_labels)
            loss_location = smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights)
            loss = loss_label + loss_vertex + loss_box + loss_location
    else:
        out_logsoftmax, out_weight = network_results
        loss = loss_cross_entropy(out_logsoftmax, out_weight)
    return loss



with torch.enable_grad():
    network.eval()
    initial_inputs = inputs.clone().detach()
    optimizer = torch.optim.Adam([inputs])
    optimizer.zero_grad()
    inputs.requires_grad=True
    network_results = run_net(network, inputs, ycb_datapoint, meta_data, dataset)
    pred_quaterion = network_results[-1]
    
    # NOTE: This is obviously just to prove gradients flow.  
    pred_quaterion.sum().backward()
    for i in range(1000): optimizer.step()   

    # Visualize
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(((inputs+1.0)/2).squeeze().permute(1, 2, 0).detach().cpu().numpy())
    axs[1].imshow(((initial_inputs+1.0)/2).squeeze().permute(1, 2, 0).detach().cpu().numpy())
    print("diff:  ", (initial_inputs - inputs).abs().sum())
    plt.show()
    print("HALT")





######################### 
######################### 
exit(0)     ############## 
######################### 
######################### 



# NOTE: Code below is older and mainly for visualization and testing.
























with torch.enable_grad():
    # network.eval()
    initial_inputs = inputs.clone().detach()
    optimizer = torch.optim.Adam([inputs])
    optimizer.zero_grad()
    inputs.requires_grad=True
    network_results = run_net(network, inputs, ycb_datapoint, meta_data, dataset)
    pose_loss = network_results[8]
    pose_loss.backward()
    for i in range(1000): optimizer.step()
    print("diff:  ", (initial_inputs - inputs).abs().sum())
    print("HERE")






if cfg.MODE == 'TRAIN':
    loss = get_loss(inputs, ycb_datapoint, meta_data, dataset)
    loss.backward()







elif cfg.MODE == 'TEST':

    # combine poses
    rois = rois.detach().cpu().numpy()
    out_pose = out_pose.detach().cpu().numpy()
    out_quaternion = out_quaternion.detach().cpu().numpy()
    num = rois.shape[0]
    poses = out_pose.copy()
    for j in range(num):
        cls = int(rois[j, 1])
        if cls >= 0:
            qt = out_quaternion[j, 4*cls:4*cls+4]
            qt = qt / np.linalg.norm(qt)
            # allocentric to egocentric
            poses[j, 4] *= poses[j, 6]
            poses[j, 5] *= poses[j, 6]
            T = poses[j, 4:]
            poses[j, :4] = allocentric2egocentric(qt, T)

    # filter out detections
    index = np.where(rois[:, -1] > cfg.TEST.DET_THRESHOLD)[0]
    rois = rois[index, :]
    poses = poses[index, :]

    # non-maximum suppression within class
    index = nms(rois, 0.2)
    rois = rois[index, :]
    poses = poses[index, :]

    # optimize depths
    im_depth = None
    from fcn.test_common import refine_pose
    if cfg.TEST.POSE_REFINE and im_depth is not None:
        poses_refined = refine_pose(labels, depth_tensor, rois, poses, meta_data, dataset)
    else:
        poses_refined = None


    from fcn.render_utils import render_image
    im_color = nerf_rgb  # B x H x W x C
    im_pose, im_pose_refined, im_label = render_image(dataset, im_color.squeeze().cpu().detach().numpy(), rois, poses, poses_refined, labels.cpu().numpy())

    from fcn.test_imageset import vis_test
    cfg.TEST.VISUALIZE = True
    if cfg.TEST.VISUALIZE:
        vis_test(dataset, im_color.permute(0, 3, 1, 2).cpu().detach().numpy(), im_depth, labels.cpu().numpy(), rois, poses, poses_refined, im_pose, im_pose_refined, out_vertex)

    print("DONE")





