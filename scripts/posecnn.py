import torch
import torch.nn.parallel 
import torch.backends.cudnn as cudnn 
import torch.utils.data

import numpy as np
import cv2
from ycb_renderer import YCBRenderer

import pdb
import sys
import os
sys.path.insert(1, './../PoseCNN-PyTorch/lib')
from utils.blob import pad_im
from datasets.factory import get_dataset
import networks
from fcn.test_imageset import test_image
from fcn.config import cfg, cfg_from_file, yaml_from_file, get_output_dir
img_input = sys.argv[1]
gpu_id = 0

def main():
    global img_input
    #pretrained = "./../PoseCNN-PyTorch/data/checkpoints/ycb_object/vgg16_ycb_object_detection_epoch_16.checkpoint.pth"
    pretrained = "./../PoseCNN-PyTorch/data/checkpoints/ycb_object/vgg16_ycb_object_self_supervision_epoch_8.checkpoint.pth"

    cfg_from_file("./../PoseCNN-PyTorch/experiments/cfgs/ycb_object.yml")
    meta = yaml_from_file("./../PoseCNN-PyTorch/data/demo/meta.yml")
    dataset = get_dataset("ycb_object_test")
    network_data = torch.load(pretrained)
    network = networks.__dict__["posecnn"](dataset.num_classes, 64, network_data).cuda()
    network = torch.nn.DataParallel(network, device_ids=[gpu_id]).cuda()
    cudnn.benchmark = True
    network.eval()

    # device
    cfg.gpu_id = gpu_id
    cfg.device = torch.device('cuda:{:d}'.format(cfg.gpu_id))
    cfg.instance_id = 0
    print('GPU device {:d}'.format(gpu_id))

    # dataset
    cfg.MODE = 'TEST'
    cfg.TEST.SYNTHESIZE = False

    '''
    # overwrite intrinsics
    if len(cfg.INTRINSICS) > 0:
        K = np.array(cfg.INTRINSICS).reshape(3, 3)
        if cfg.TEST.SCALES_BASE[0] != 1:
            scale = cfg.TEST.SCALES_BASE[0]
            K[0, 0] *= scale
            K[0, 2] *= scale
            K[1, 1] *= scale
            K[1, 2] *= scale
        dataset._intrinsic_matrix = K
        print(dataset._intrinsic_matrix)
    '''

    print('loading 3D models')
    cfg.renderer = YCBRenderer(width=cfg.TRAIN.SYN_WIDTH, height=cfg.TRAIN.SYN_HEIGHT, gpu_id=gpu_id, render_marker=False)
    if cfg.TEST.SYNTHESIZE:
        cfg.renderer.load_objects(dataset.model_mesh_paths, dataset.model_texture_paths, dataset.model_colors)
    else:
        model_mesh_paths = [dataset.model_mesh_paths[i-1] for i in cfg.TEST.CLASSES[1:]]
        model_texture_paths = [dataset.model_texture_paths[i-1] for i in cfg.TEST.CLASSES[1:]]
        model_colors = [dataset.model_colors[i-1] for i in cfg.TEST.CLASSES[1:]]
        cfg.renderer.load_objects(model_mesh_paths, model_texture_paths, model_colors)

    cfg.renderer.set_camera_default()
    print(dataset.model_mesh_paths)
    cfg.TEST.POSE_REFINE = False
    cfg.TEST.VISUALIZE = False 
    if os.path.isdir(img_input):
        for img in os.listdir(img_input):
            # for each image
            im = pad_im(cv2.imread(os.path.join(img_input, img), cv2.IMREAD_COLOR), 16)
            depth = None
            print('no depth image')

            # rescale image if necessary
            if cfg.TEST.SCALES_BASE[0] != 1:
                im_scale = cfg.TEST.SCALES_BASE[0]
                im = pad_im(cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR), 16)
                if depth is not None:
                    depth = pad_im(cv2.resize(depth, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST), 16)

            # run network
            im_pose, im_pose_refined, im_label, labels, rois, poses, poses_refined = test_image(network, dataset, im, depth)

            # save result
            if not cfg.TEST.VISUALIZE:

                # map the roi index
                for j in range(rois.shape[0]):
                    rois[j, 1] = cfg.TRAIN.CLASSES.index(cfg.TEST.CLASSES[int(rois[j, 1])])

                result = {'labels': labels, 'rois': rois, 'poses': poses, 'poses_refined': poses_refined, 'intrinsic_matrix': dataset._intrinsic_matrix}
                head, tail = os.path.split(img)
                filename = os.path.join(resdir, tail + '.mat')
                scipy.io.savemat(filename, result, do_compression=True)
                # rendered image
                filename = os.path.join(resdir, tail + '_render.jpg')
                cv2.imwrite(filename, im_pose[:, :, (2, 1, 0)])
                filename = os.path.join(resdir, tail + '_render_refined.jpg')
                cv2.imwrite(filename, im_pose_refined[:, :, (2, 1, 0)])
    else:

        img_input = sys.argv[1]
        # for each image
        im = pad_im(cv2.imread(img_input, cv2.IMREAD_COLOR), 16)
        depth = None
        print('no depth image')

        # rescale image if necessary
        if cfg.TEST.SCALES_BASE[0] != 1:
            im_scale = cfg.TEST.SCALES_BASE[0]
            im = pad_im(cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR), 16)
            if depth is not None:
                depth = pad_im(cv2.resize(depth, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST), 16)

        # run network
        im_pose, im_pose_refined, im_label, labels, rois, poses, poses_refined = test_image(network, dataset, im, depth)

        # save result
        if not cfg.TEST.VISUALIZE:

            # map the roi index
            for j in range(rois.shape[0]):
                rois[j, 1] = cfg.TRAIN.CLASSES.index(cfg.TEST.CLASSES[int(rois[j, 1])])

            result = {'labels': labels, 'rois': rois, 'poses': poses, 'poses_refined': poses_refined, 'intrinsic_matrix': dataset._intrinsic_matrix}
            head, tail = os.path.split(img)
            filename = os.path.join(resdir, tail + '.mat')
            scipy.io.savemat(filename, result, do_compression=True)
            # rendered image
            filename = os.path.join(resdir, tail + '_render.jpg')
            cv2.imwrite(filename, im_pose[:, :, (2, 1, 0)])
            filename = os.path.join(resdir, tail + '_render_refined.jpg')
            cv2.imwrite(filename, im_pose_refined[:, :, (2, 1, 0)])

if __name__ == '__main__':
    main()

