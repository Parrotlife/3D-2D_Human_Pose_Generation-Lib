import os
import torch
import numpy as np
import json
import math
import time

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

"""find the length of a limb from its name"""
def tensor_find_limb_length(keypoints, limb_name):
    batch_size = keypoints.size()[0]
    
    return torch.norm(keypoints[:,:,limb_dict[limb_name][0]]-keypoints[:,:,limb_dict[limb_name][1]], dim=1)\
                      .reshape((batch_size,1))

""" Change the position of a joint in 3d"""
def tensor_change_3d_joint_pos(keypoints, joint_name, pos_3d):
    keypoints[:, :, joint_dict[joint_name]] = pos_3d
    return keypoints

""" get the position of a joint in 3d"""
def tensor_get_3d_joint_pos(keypoints, joint_name):
    batch_size = keypoints.size()[0]
    return keypoints[:,:, joint_dict[joint_name]].reshape((batch_size,3,1))

""" Forward kinematics function"""  
def tensor_move_member(data, member_name, a0, a1, a2, a3, device=torch.device("cuda:0")):
    
    batch_size = data.size()[0]
    
    #### get all the useful sin and cos
    c0 = torch.cos(a0)
    c1 = torch.cos(a1)
    c2 = torch.cos(a2)
    c3 = torch.cos(a3)
    c23 = torch.cos(a2+a3)

    s0 = torch.sin(a0)
    s1 = torch.sin(a1)
    s2 = torch.sin(a2)
    s3 = torch.sin(a3)
    s23 = torch.sin(a2+a3)

    #### get what member we are moving and w
    body_side = member_name.split(maxsplit=1)[0]
    member = member_name.split(maxsplit=1)[1]

    #### define the correct limbs joints and rotation matrices
    if member == 'arm':
        member_limbs = ['biceps', 'forearm']
        member_joints = ['shoulder', 'elbow', 'wrist']
        
        r_1 = []
        r_2 = []
        for i in range(batch_size):

            r_2.append(torch.tensor([[1,       0,      0],
                                     [0,   c0[i], -s0[i]],
                                     [0,   s0[i],  c0[i]]]))


            r_1.append(torch.tensor([[ c1[i],  0,   s1[i]],
                                     [     0,  1,       0],
                                     [-s1[i],  0,    c1[i]]]))
        
    else:
        member_limbs = ['quad', 'tibias']
        member_joints = ['hip', 'knee', 'ankle']

        r_2.append(torch.tensor([[ c0[i],     0, s0[i]],
                                 [     0,     1,     0],
                                 [-s0[i],     0, c0[i]]]))

        r_1.append(torch.tensor([[ c1[i], -s1[i],    0],
                                 [ s1[i],  c1[i],    0],
                                 [     0,      0,    1]]))
        
    r_1 = torch.stack(r_1, dim=0).to(device)
    r_2 = torch.stack(r_2, dim=0).to(device)

    #### get the limbs lengths
    l2 = tensor_find_limb_length(data, body_side+' '+member_limbs[0])
    l3 = tensor_find_limb_length(data, body_side+' '+member_limbs[1])

    #### get the member origin
    origin_pos = tensor_get_3d_joint_pos(data, body_side+' '+member_joints[0])
    
    c2 = c2.reshape((batch_size,1))
    s2 = s2.reshape((batch_size,1))
    c23 = c23.reshape((batch_size,1))
    s23 = s23.reshape((batch_size,1))
    
    zeros = torch.tensor([0 for _ in range(batch_size)]).float().reshape((batch_size,1)).to(device)
    
    ### find the 2d positions of all the joints relative to the origin
    if member == 'arm':            
        factor = 1         
        if body_side == 'right':
            factor = -1

        x_joints = dict(zip(member_joints, [zeros, factor*(l2*c2), factor*(l2*c2 + l3*c23)]))
        y_joints = dict(zip(member_joints, [zeros, factor*(l2*s2), factor*(l2*s2 + l3*s23)]))
        z_joints = dict(zip(member_joints, [zeros, zeros, zeros]))
    else:
        y_joints = dict(zip(member_joints, [zeros, l2*c2, l2*c2 + l3*c23]))
        z_joints = dict(zip(member_joints, [zeros, l2*s2, l2*s2 + l3*s23]))
        x_joints = dict(zip(member_joints, [zeros, zeros, zeros]))
        

    #### find the new position of each joints and move it
    for joint in member_joints[1:]:
        pos = torch.stack([x_joints[joint], y_joints[joint], z_joints[joint]]).transpose(0,1)
        pos = torch.matmul(r_2,torch.matmul(r_1, pos))
        
        pos = pos + origin_pos
        
        data = tensor_change_3d_joint_pos(data, body_side+' '+joint, pos.reshape((batch_size,3)))
    
    return data

"""Rotate the pose according to an angle and axis"""
def tensor_rotate_pose(data, axis, angle, device=torch.device("cuda:0")):
        
        c = torch.cos(angle)
        s = torch.sin(angle)
        
        
        batch_size = data.size()[0]
        
        if axis == 'x':
            r = torch.stack([torch.tensor([[1,    0,     0],
                                           [0, c[i], -s[i]],
                                           [0, s[i],  c[i]]]) for i in range(batch_size)], dim=0).to(device)
        if axis == 'y':
            r = torch.stack([torch.tensor([[ c[i], 0, s[i]],
                                           [    0, 1,    0],
                                           [-s[i], 0, c[i]]]) for i in range(batch_size)], dim=0).to(device)
        if axis == 'z':
            r = torch.stack([torch.tensor([[ c[i], -s[i], 0],
                                           [ s[i],  c[i], 0],
                                           [    0,     0, 1]]) for i in range(batch_size)], dim=0).to(device)
        
        for joint in joint_names:
            
            pos = tensor_get_3d_joint_pos(data, joint)
            
            pos = torch.matmul(r, pos)
        
            data = tensor_change_3d_joint_pos(data, joint, pos.reshape((batch_size,3)))
        
        return data
    
"""Rotate the point according to an axis and an angle"""    
def tensor_rotate_point(point, rot_axis, angle, device=torch.device("cuda:0")):
    
    print('in tensor_rotate_point', device)
    
    
    ux, uy, uz = rot_axis[:,0],rot_axis[:,1],rot_axis[:,2]
    
    batch_size = point.size()[0]
    
    c = torch.cos(angle)
    s = torch.sin(angle)

    rot_matrix = torch.stack(
        [torch.tensor([[   c[i]+ux[i]**2*(1-c[i]), ux[i]*uy[i]*(1-c[i])-uz[i]*s[i], ux[i]*uz[i]*(1-c[i])+uy[i]*s[i]],
                       [uy[i]*ux[i]*(1-c[i])+uz[i]*s[i],    c[i]+uy[i]**2*(1-c[i]), uy[i]*uz[i]*(1-c[i])-ux[i]*s[i]],
                       [uz[i]*ux[i]*(1-c[i])-uy[i]*s[i], uz[i]*uy[i]*(1-c[i])+ux[i]*s[i],    c[i]+uz[i]**2*(1-c[i])]])
         for i in range(batch_size)], dim=0).to(device)

    return torch.matmul(rot_matrix, point)

"""Rotate the point according to an axis and an angle"""    
def tensor_euler_rotate_point(point, rot_matrix, device=torch.device("cuda:0")):
    
    print('in tensor_rotate_point', device)
 
    return torch.matmul(rot_matrix, point)

"""Rotate the back or head according to 3 angles"""
def tensor_rotate_backOrHead(data, member_name, a0, a1, a2, device=torch.device("cuda:0")):
        
        print('in tensor_rotate_backOrHead', device)
        
        joints2move = {'head': ['nose','head','right eye','left eye','right ear','left ear'],
                       'back': ['nose','head','right eye','left eye','right ear','left ear',
                                'right shoulder','right elbow', 'right wrist', 'left shoulder',
                                'left elbow', 'left wrist','center shoulder']}
        origins = {'head': 'center shoulder',
                   'back': 'center back'}
        batch_size = data.size()[0]
        
        x = torch.tensor([[1,0,0] for _ in range(batch_size)]).float().to(device).reshape((batch_size,3,1))
        y = torch.tensor([[0,1,0] for _ in range(batch_size)]).float().to(device).reshape((batch_size,3,1))
        z = torch.tensor([[0,0,1] for _ in range(batch_size)]).float().to(device).reshape((batch_size,3,1))

        for joint in joints2move[member_name]:
            
            origin_joint = tensor_get_3d_joint_pos(data, origins[member_name])
            
            pos = tensor_get_3d_joint_pos(data, joint) - origin_joint
            
            pos = tensor_rotate_point(pos, x, a0, device) + origin_joint
            
            data = tensor_change_3d_joint_pos(data, joint, pos.reshape((batch_size,3)))
        
        y = tensor_rotate_point(y, x, a0, device)
        z = tensor_rotate_point(z, x, a0, device)
        
        for joint in joints2move[member_name]:
            
            origin_joint = tensor_get_3d_joint_pos(data, origins[member_name])
            
            pos = tensor_get_3d_joint_pos(data, joint) - origin_joint
            
            pos = tensor_rotate_point(pos, y, a1, device) + origin_joint
            
            data = tensor_change_3d_joint_pos(data, joint, pos.reshape((batch_size,3)))
        
        z = tensor_rotate_point(z, y, a1, device)
        
        for joint in joints2move[member_name]:
            
            origin_joint = tensor_get_3d_joint_pos(data, origins[member_name])
            
            pos = tensor_get_3d_joint_pos(data, joint) - origin_joint
            
            pos = tensor_rotate_point(pos, z, a2, device) + origin_joint
            
            data = tensor_change_3d_joint_pos(data, joint, pos.reshape((batch_size,3)))
            
        return data
    
"""Rotate the back or head according to 3 angles"""
def tensor_euler_rotate_backOrHead(data, member_name, a0, a1, a2, device=torch.device("cuda:0")):
        
        print('in tensor_rotate_backOrHead', device)
        
        joints2move = {'head': ['nose','head','right eye','left eye','right ear','left ear'],
                       'back': ['nose','head','right eye','left eye','right ear','left ear',
                                'right shoulder','right elbow', 'right wrist', 'left shoulder',
                                'left elbow', 'left wrist','center shoulder']}
        origins = {'head': 'center shoulder',
                   'back': 'center back'}
        batch_size = data.size()[0]
        
        c0 = torch.cos(a0)
        c1 = torch.cos(a1)
        c2 = torch.cos(a2)
        
        s0 = torch.sin(a0)
        s1 = torch.sin(a1)
        s2 = torch.sin(a2)
        
        e00 = c1*c2 
        e01 = -c0*s2 + s0*s1*c2
        e02 = s0*s2 + c0*s1*c2
        e10 = c1*s2 
        e11 = c0*c2 + s0*s1*s2
        e12 = -s0*c2 + c0*s1*s2
        e20 = -s1
        e21 = s0*c1
        e22 = c0*c1
        
        rot_matrix = torch.stack(
        [torch.tensor([[e00[i], e01[i], e02[i]],
                       [e10[i], e11[i], e12[i]],
                       [e20[i], e21[i], e22[i]]]) for i in range(batch_size)], dim=0).to(device)

        for joint in joints2move[member_name]:
            
            origin_joint = tensor_get_3d_joint_pos(data, origins[member_name])
            
            pos = tensor_get_3d_joint_pos(data, joint) - origin_joint
            
            pos = tensor_euler_rotate_point(pos, rot_matrix, device) + origin_joint
            
            data = tensor_change_3d_joint_pos(data, joint, pos.reshape((batch_size,3)))
            
        return data
    
"""Rotate the back or head according to 3 angles"""
def tensor_euler_rotate_backOrHead(data, member_name, a0, a1, a2, device=torch.device("cuda:0")):
        
        print('in tensor_rotate_backOrHead', device)
        
        joints2move = {'head': ['nose','head','right eye','left eye','right ear','left ear'],
                       'back': ['nose','head','right eye','left eye','right ear','left ear',
                                'right shoulder','right elbow', 'right wrist', 'left shoulder',
                                'left elbow', 'left wrist','center shoulder']}
        origins = {'head': 'center shoulder',
                   'back': 'center back'}
        batch_size = data.size()[0]
        
        c0 = torch.cos(a0)
        c1 = torch.cos(a1)
        c2 = torch.cos(a2)
        
        s0 = torch.sin(a0)
        s1 = torch.sin(a1)
        s2 = torch.sin(a2)
        
        e00 = c1*c2 
        e01 = -c1*s2
        e02 = s1
        e10 = c0*s2 + c2*s0*s1 
        e11 = c0*c2 - s0*s1*s2
        e12 = -c1*s0
        e20 = -c0*c2*s1 + s0*s2
        e21 = c0*s1*s2 + c2*s0
        e22 = c0*c1
        
        rot_matrix = torch.stack(
        [torch.tensor([[e00[i], e01[i], e02[i]],
                       [e10[i], e11[i], e12[i]],
                       [e20[i], e21[i], e22[i]]]) for i in range(batch_size)], dim=0).to(device)

        for joint in joints2move[member_name]:
            
            origin_joint = tensor_get_3d_joint_pos(data, origins[member_name])
            
            pos = tensor_get_3d_joint_pos(data, joint) - origin_joint
            
            pos = tensor_euler_rotate_point(pos, rot_matrix, device) + origin_joint
            
            data = tensor_change_3d_joint_pos(data, joint, pos.reshape((batch_size,3)))
            
        return data
    
"""Rotates a pose according to angles in a dictionary"""
def tensor_full_pose_rotation(data, angle_dic, device=torch.device("cuda:0")):
    
    print('in tensor_full_pose_rotation', device)
    
    members = ['right arm', 'left arm', 'right leg', 'left leg']
    head_or_back = ['head','back']
    axis = ['x', 'y', 'z']
    
    final_pose = data
    
    for key in angle_dic.keys():
        a = angle_dic[key]
        print('we are moving the ',key)
        if key in members:
            final_pose = tensor_move_member(final_pose, key, a[:,0], a[:,1], a[:,2], a[:,3], device)
        if key in head_or_back:
            #a[:,0] = 0
            #a[:,1] = math.pi/7
            #a[:,2] = 0
            
            a0 = a[:,0]
            a1 = a[:,1]
            a2 = a[:,2]
            final_pose = tensor_euler_rotate_backOrHead(final_pose, key, a0, a1, a2, device)
            #final_pose = tensor_rotate_backOrHead(final_pose, key, a[:,0], a[:,1], a[:,2], device)
        if key in axis:
            final_pose = tensor_rotate_pose(final_pose, key, a[:,0], device)
    return final_pose