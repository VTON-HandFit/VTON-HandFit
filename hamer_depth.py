
from pathlib import Path
import torch
import argparse
import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['MUJOCO_GL'] = 'osmesa'
import sys
import cv2
import numpy as np
import pickle

from pathlib import Path
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.utils.geometry import aa_to_rotmat, perspective_projection
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from typing import List, Tuple
import h5py
from PIL import Image
from bps_torch.bps import bps_torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initiate the bps module
bps_t = bps_torch(bps_type='random_uniform',
                n_bps_points=1024,
                radius=1.,
                n_dims=3,
                custom_basis=None)



openpose_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
gt_indices = openpose_indices



# Download and load checkpoints
#  download_models(CACHE_DIR_HAMER)
model, model_cfg = load_hamer('./_DATA/hamer_ckpts/checkpoints/hamer.ckpt')

# Setup HaMeR model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)
model.eval()

# Load detector
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
import hamer
cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
cfg_path = './hamer/hamer/configs/cascade_mask_rcnn_vitdet_h_75ep.py'
detectron2_cfg = LazyConfig.load(str(cfg_path))
detectron2_cfg.train.init_checkpoint = "./_DATA/hamer_ckpts/checkpoints/model_final_f05665.pkl"
for i in range(3):
    detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
detector = DefaultPredictor_Lazy(detectron2_cfg)

# keypoint detector
cpm = ViTPoseModel(device)

# Setup the renderer
    # Setup the renderer
renderer = Renderer(model_cfg, faces=model.mano.faces)

rescale_factor = 2.0


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)



def norm_depth(rend_depth):
    # 1.0 - (0.8 * (depth - minval) / (maxval - minval))
    non_zero_indices = np.nonzero(rend_depth)
    non_zero_elements = rend_depth[non_zero_indices]
    minval = non_zero_elements.min()
    maxval = non_zero_elements.max()
    depth_norm = 1.0 - (0.8 * (non_zero_elements - minval) / (maxval - minval))
    rend_depth[non_zero_indices] = depth_norm
    return rend_depth

def get_depth_info(vton_img,img_path,crop_save_path=None):
    # if not os.path.exists(crop_save_path):
    #     os.makedirs(crop_save_path)
    print('start get hand depth.')
    results_dict = {}

    img_cv2 = np.array(vton_img)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
    det_out = detector(img_cv2)
    img = img_cv2.copy()[:, :, ::-1]
    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
    pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores=det_instances.scores[valid_idx].cpu().numpy()

    # Detect human keypoints for each person
    vitposes_out = cpm.predict_pose(
        img_cv2,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    bboxes = []
    is_right = []
    vit_keypoints_list = []

    # Use hands based on hand keypoint detections
    for vitposes in vitposes_out:
        # print(vitposes.keys()) # dict_keys(['bbox', 'keypoints'])
        left_hand_keyp = vitposes['keypoints'][-42:-21]
        right_hand_keyp = vitposes['keypoints'][-21:]

        # Rejecting not confident detections
        keyp = left_hand_keyp
        valid = keyp[:,2] > 0.5
        if sum(valid) > 10:
            bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
            bboxes.append(bbox)
            is_right.append(0)
            vit_keypoints_list.append(keyp)

        keyp = right_hand_keyp
        valid = keyp[:,2] > 0.5
        if sum(valid) > 10:
            bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
            bboxes.append(bbox)
            is_right.append(1)
            vit_keypoints_list.append(keyp)

    if len(bboxes) == 0:
        print('no hand detected!!!') #, results_dict[img_path]['tid'], results_dict[img_path]['tracked_ids'], results_dict[img_path]['tracked_time'])
        return None, None, None

    boxes = np.stack(bboxes)
    right = np.stack(is_right)
    vit_keypoints = np.stack(vit_keypoints_list)
   
    dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, vit_keypoints, rescale_factor=rescale_factor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    all_verts = []
    all_cam_t = []
    all_right = []
    all_vit_2d = []
    all_pred_2d = []
    all_bboxes = []
    all_mano = []
    all_mano_betas = []

    left_flag = False
    right_flag = False
    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        multiplier = (2*batch['right']-1)
        pred_cam = out['pred_cam']
        pred_cam[:,1] = multiplier*pred_cam[:,1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        multiplier = (2*batch['right']-1)
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)#.detach().cpu().numpy()

        # Render the result
        batch_size = batch['img'].shape[0]
        
        d3 = out['pred_keypoints_3d'].reshape(batch_size, -1, 3)
        for i in range(len(multiplier)):
            d3[:,:,0][i] = multiplier[i] *d3[:,:,0][i]
        
        
        out['pred_keypoints_2d'] = perspective_projection(d3,
                                    translation=pred_cam_t_full.reshape(batch_size, 3),
                                    focal_length=torch.tensor([[scaled_focal_length, scaled_focal_length]], device=d3.device),
                                    camera_center = torch.tensor([[img_size[0][0] / 2, img_size[0][1] / 2]], device=d3.device)
                                    )
 
        pred_cam_t_full = pred_cam_t_full.detach().cpu().numpy()

        for n in range(batch_size):
            if out['pred_keypoints_2d'][n][:,0].min() < -15 or out['pred_keypoints_2d'][n][:,1].min() < -15 or out['pred_keypoints_2d'][n][:,0].max() > img_cv2.shape[1]+15 or out['pred_keypoints_2d'][n][:,1].max() > img_cv2.shape[0]+15:
                print('pred_joints out of image!!!')
                # flag=1
                continue

            # Get filename from path img_path
            img_fn, _ = os.path.splitext(os.path.basename(img_path))
            person_id = int(batch['personid'][n])
            white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
            input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
            input_patch = input_patch.permute(1,2,0).numpy()

            # Add all verts and cams to list
            verts = out['pred_vertices'][n].detach().cpu().numpy()
            pred_joints = out['pred_keypoints_2d'][n].detach().cpu().numpy()

            is_right = int(batch['right'][n].cpu().numpy())

            pred_joints[:,0] = (2*is_right-1)*pred_joints[:,0]


            v = np.ones((21, 1))
            pred_joints = np.concatenate((pred_joints, v), axis=-1)
            verts[:,0] = (2*is_right-1)*verts[:,0]
            cam_t = pred_cam_t_full[n]
            all_verts.append(verts)
            all_cam_t.append(cam_t)
            all_right.append(is_right)
            # import pdb;pdb.set_trace()
            pred_joints[:,0] = (2*is_right-1)*pred_joints[:,0]

            all_pred_2d.append(pred_joints)
            all_vit_2d.append(batch['2d'][n])
            all_bboxes.append(batch['bbox'][n].detach().cpu().numpy())


            mano_out =  out['mano_theta'][n].detach().cpu().numpy()
            all_mano.append(mano_out)
            all_mano_betas.append(out['mano_betas'][n].detach().cpu().numpy())

    if len(all_vit_2d) == 0:
        return None,None,None
    all_vit_2d = torch.stack(all_vit_2d).cpu().numpy()
    all_pred_2d = np.stack(all_pred_2d)
    all_bboxes = np.stack(all_bboxes)

    all_mano = np.stack(all_mano)
    all_mano_betas = np.stack(all_mano_betas)
    all_right = np.stack(all_right)
    all_verts = np.stack(all_verts)
    all_cam_t = np.stack(all_cam_t)
    all_bboxes_bouding_pos = []

# if args.full_frame and len(all_verts) > 0:
    misc_args = dict(
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        focal_length=scaled_focal_length,
    )
    # print(all_cam_t)
    norm_depths = []
    ori_depths = []
    mask = np.zeros((768,1024), dtype=np.uint8)
    mask_bbox_padding = 10
    for i in range(len(all_verts)):
        cam_view, cam_depth_ori = renderer.render_rgba_multiple([all_verts[i]], cam_t=[all_cam_t[i]],
                                                                        render_res=img_size[n],
                                                                        is_right=[all_right[i]],
                                                                        **misc_args)
        ori_depths.append(cam_depth_ori)
        cam_depth = norm_depth(cam_depth_ori.copy())
        norm_depths.append(cam_depth)
        non_zero_indices = np.nonzero(cam_depth_ori)
        bouding_pos = [np.min(non_zero_indices[1]), np.max(non_zero_indices[1]), np.min(non_zero_indices[0]),
                        np.max(non_zero_indices[0])]
        mask[max(np.min(non_zero_indices[0]) - mask_bbox_padding,0):min(np.max(non_zero_indices[0]) + mask_bbox_padding,mask.shape[0]),
        max(np.min(non_zero_indices[1]) - mask_bbox_padding,0):min(np.max(non_zero_indices[1]) + mask_bbox_padding,mask.shape[1])] = 255
        
        bouding_pos_padding = [max(np.min(non_zero_indices[1]) - mask_bbox_padding,0), min(np.max(non_zero_indices[1]) + mask_bbox_padding,mask.shape[0]),
                                max(np.min(non_zero_indices[0]) - mask_bbox_padding,0), min(np.max(non_zero_indices[0]) + mask_bbox_padding,mask.shape[1])]
        all_bboxes_bouding_pos.append(bouding_pos_padding)



    results_dict = {}
    results_dict['pred_2d'] = all_pred_2d
    # results_dict[img_name]['vit_2d'] = all_vit_2d
    results_dict['bbox'] = all_bboxes
    results_dict['bbox_2'] = all_bboxes_bouding_pos
    results_dict['verts'] = all_verts
    results_dict['mano'] = all_mano
    results_dict['mano_betas'] = all_mano_betas
    results_dict['handtype'] = all_right
    results_dict['cam_t'] = all_cam_t

    depth_ori = np.array(ori_depths)
    depth_norm = np.array(norm_depths)
    temp_mask = depth_ori != 0
    depth_ori_masked = np.where(temp_mask, depth_ori, np.inf)
    min_channel_indices = np.argmin(depth_ori_masked, axis=0)
    rows, cols = np.meshgrid(np.arange(depth_norm.shape[1]), np.arange(depth_norm.shape[2]), indexing='ij')
    result = depth_norm[min_channel_indices, rows, cols]
    depth_result = (result * 255).astype(np.uint8)
    # cv2.imwrite(os.path.join(crop_save_path, f'{img_fn}.jpg'), depth_result)
    crop_imgs = []
    for i in range(len(all_bboxes_bouding_pos)):
        bbox = all_bboxes_bouding_pos[i]
        padding = 0
        crop_img = img_cv2[int(bbox[2]):int(bbox[3]), int(bbox[0]):int(bbox[1])]
        # cv2.imwrite(os.path.join(crop_save_path, f'{img_fn}_{i}.jpg'), crop_img)
        crop_imgs.append(crop_img)

    if len(results_dict['handtype'] ) == 1:
        results_dict['handtype'] = np.append(results_dict['handtype'], -1)
        results_dict['pred_2d'] = np.append(results_dict['pred_2d'][:,:,:2], np.zeros_like(results_dict['pred_2d'][:,:,:2]),axis=0)
        results_dict['bbox_2'] = np.append(results_dict['bbox_2'], np.zeros_like(results_dict['bbox_2']), axis=0)

        bps_enc = bps_t.encode(torch.tensor(results_dict['verts']).cuda(),
                     feature_type=['dists'],
                     x_features=None,
                     custom_basis=None)

        results_dict['verts'] = np.append(bps_enc['dists'].cpu().numpy(), np.zeros_like(bps_enc['dists'].cpu().numpy()), axis=0)
        
        results_dict['mano'] = matrix_to_rotation_6d(torch.tensor(results_dict['mano'])).numpy()
        results_dict['mano'] = np.append(results_dict['mano'], np.zeros_like(results_dict['mano']), axis=0)
    elif len(results_dict['handtype']) == 2:
        results_dict['pred_2d'] = results_dict['pred_2d'][:,:,:2]
        bps_enc = bps_t.encode(torch.tensor(results_dict['verts']).cuda(),
                               feature_type=['dists'],
                    #  feature_type=['dists','deltas'],
                     x_features=None,
                     custom_basis=None)
        results_dict['verts'] = bps_enc['dists'].cpu().numpy()
        results_dict['mano'] = matrix_to_rotation_6d(torch.tensor(results_dict['mano'])).numpy()
        results_dict['bbox_2'] = np.array(results_dict['bbox_2'])
    else:
        print('unsupperted hand number!!!')
        return None,None,None

    results_dict['bbox_2']  = results_dict['bbox_2'][:, [0, 2, 1, 3]]# [2,4]
    result_list = [ results_dict['handtype'],results_dict['pred_2d'],results_dict['verts'],results_dict['bbox_2'],results_dict['mano']]
    return depth_result,result_list,crop_imgs
