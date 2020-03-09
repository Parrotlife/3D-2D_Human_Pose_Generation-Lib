import os
import numpy as np
import json
import math
import pose_3D as pose3d
from PIL import Image
import torch
import torch.nn.functional as F

"""define limbs connections and joints indices"""
limbs = [(17,20),(17,1),(1,2),(2,3),(17,4),(4,5),(5,6),(7,8),(8,9),(10,11),(11,12),(7,10),(17,19),(19,18)]
joints = list(range(21))

limb_names = ["neck", "right shoulder", "right biceps", "right forearm", "left shoulder", "left biceps",
              "left forearm", "right quad", "right tibias", "left quad", "left tibias","hip", "upper back", "lower back"]

joint_names = ['nose', 'right shoulder','right elbow', 'right wrist', 'left shoulder', 'left elbow', 'left wrist', 
               'right hip', 'right knee', 'right ankle', 'left hip', 'left knee', 'left ankle','right eye','left eye',
               'right ear','left ear','center shoulder','center hip','center back','head']

limb_dict = dict(zip(limb_names, limbs))
joint_dict = dict(zip(joint_names, joints))

"""define pifpaf body proportions"""

BODY_PROPORTIONS = {'neck': 0.0,
                    'shoulder': 0.0,
                    'biceps': 0.0,
                    'forearm': 0.0,
                    'quad': 0.0,
                    'tibias': 0.0,
                    'hip': 0.0,
                    'back': 0.0
                    }
FACE_POSITIONS = {'nose': [0.0, 0.0, 0.0],
                  'ears': [0.0, 0.0, 0.0],
                  'eyes': [0.0, 0.0, 0.0]
                 }

NB_JOINTS = 21

"""intrinsic camera matrix"""
kk = np.array([[424.72847911,  -0.46074429, 291.65605788],
               [  0.        , 426.16973617, 146.18885257],
               [  0.        ,   0.        ,   1.        ]])

"""normalize the data so we have the center hip at the axes origins and ymax-ymin is 1 """
def tensor_normalize(list_of_poses, shift=None, resize=True):
    #we find the shift to center around the hip
    batch_size = list_of_poses.size()[0]
    
    nb_joints = list_of_poses.size()[2]
    
    if shift is None:
        shift = (list_of_poses[:,:,joint_dict['right hip']] + list_of_poses[:,:,joint_dict['left hip']])\
                .reshape((batch_size,2,1))/2

    ratio = 1
   
    if resize:
        #we find the ratio to scale down
        ratio = (torch.max(list_of_poses[:,1,:],1).values-torch.min(list_of_poses[:,1,:],1).values)
        #ratio = find_limb_length(data, 'hip')/0.1739
   
    
    return shift, ratio
    
"""Convert a tensor in pixel coordinate to absolute camera coordinates
   It accepts lists or torch/numpy tensors of (m, 2) or (m, x, 2)
   where x is the number of keypoints"""
def tensor_project_pose_2cam(list_of_poses, kk, z_shifts, joints2keep, device=torch.device("cuda:0")):
    
    batch_size = list_of_poses.size()[0]
    
    nb_joints = list_of_poses.size()[2]

    mats = np.array([kk for _ in range(batch_size)]).reshape((batch_size,3,3))
    mats = torch.from_numpy(mats).float().to(device)
    mats.requires_grad_(True)

    z_shifts = torch.stack([z_shifts for _ in range(nb_joints)], dim=1)
    
    list_of_poses[:,2] = list_of_poses[:,2]+z_shifts
    
    kk_multiplications = torch.matmul(mats, list_of_poses)
    
    projected_poses = kk_multiplications/kk_multiplications.transpose(0,1)[2].reshape((batch_size,1,21))
    
    projected_poses = projected_poses[:,:2]

    
    shift = (projected_poses[:,:,joint_dict['right hip']] + projected_poses[:,:,joint_dict['left hip']])\
                .reshape((batch_size,2,1))/2
    
    joints_ids = [joint_dict[x] for x in joints2keep]
    
    shift, ratio = tensor_normalize(projected_poses[:,:,joints_ids], shift=shift)
    ratio = torch.stack([ratio.reshape((batch_size,1)) for _ in range(projected_poses.size()[2])], dim=2)
    projected_poses = (projected_poses-shift)/ratio
    
    legs_ids = [x for x in range(nb_joints) if x not in joints_ids]
    
    projected_poses[:,:,legs_ids] = -10
    
    return projected_poses
