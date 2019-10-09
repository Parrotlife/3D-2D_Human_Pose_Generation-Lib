import os
import numpy as np
import json
import math

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
def find_limb_length(data, limb_name):

    dim = 3

    keypoints = data.reshape((NB_JOINTS,dim)).transpose()

    return np.linalg.norm(keypoints[:,limb_dict[limb_name][0]]-keypoints[:,limb_dict[limb_name][1]])

""" Change the position of a joint in 3d"""
def change_3d_joint_pos(data, joint_name, pos_3d):
    keypoints = data.reshape((NB_JOINTS,3)).transpose()
    keypoints[:, joint_dict[joint_name]] = pos_3d

    return keypoints.transpose().flatten()

""" get the position of a joint in 3d"""
def get_3d_joint_pos(data, joint_name):
    keypoints = data.reshape((NB_JOINTS,3)).transpose()

    return keypoints[:, joint_dict[joint_name]].reshape((3,1))

""" Forward kinematics function"""  
def move_member(data, angles_dic, member_name, a0, a1, a2, a3):
    
    joints_data = data.copy()

    angles_dic[member_name] = [a0, a1, a2, a3]

    #### get all the useful sin and cos
    c0 = math.cos(a0)
    c1 = math.cos(a1)
    c2 = math.cos(a2)
    c3 = math.cos(a3)
    c23 = math.cos(a2+a3)

    s0 = math.sin(a0)
    s1 = math.sin(a1)
    s2 = math.sin(a2)
    s3 = math.sin(a3)
    s23 = math.sin(a2+a3)

    #### get what member we are moving and w
    body_side = member_name.split(maxsplit=1)[0]
    member = member_name.split(maxsplit=1)[1]

    #### define the correct limbs joints and rotation matrices
    if member == 'arm':
        member_limbs = ['biceps', 'forearm']
        member_joints = ['shoulder', 'elbow', 'wrist']

        r_2 = np.array([[1,    0,   0],
                        [0,   c0, -s0],
                        [0,   s0,  c0]])

        r_1 = np.array([[ c1,  0,   s1],
                        [  0,  1,    0],
                        [-s1,  0,   c1]])

    else:
        member_limbs = ['quad', 'tibias']
        member_joints = ['hip', 'knee', 'ankle']

        r_2 = np.array([[ c0,  0,   s0],
                        [  0,  1,    0],
                        [-s0,  0,   c0]])

        r_1 = np.array([[ c1, -s1, 0],
                        [ s1,  c1, 0],
                        [-s1,   0, 1]])


    #### get the limbs lengths
    l2 = find_limb_length(joints_data, body_side+' '+member_limbs[0])
    l3 = find_limb_length(joints_data, body_side+' '+member_limbs[1])

    #### get the member origin
    origin_pos = get_3d_joint_pos(joints_data, body_side+' '+member_joints[0])

    ### find the 2d positions of all the joints relative to the origin
    if member == 'arm':            
        factor = 1         
        if body_side == 'right':
            factor = -1

        x_joints = dict(zip(member_joints, [0, factor*(l2*c2), factor*(l2*c2 + l3*c23)]))
        y_joints = dict(zip(member_joints, [0, factor*(l2*s2), factor*(l2*s2 + l3*s23)]))
        z_joints = dict(zip(member_joints, [0, 0, 0]))
    else:
        y_joints = dict(zip(member_joints, [0, l2*c2, l2*c2 + l3*c23]))
        z_joints = dict(zip(member_joints, [0, l2*s2, l2*s2 + l3*s23]))
        x_joints = dict(zip(member_joints, [0, 0, 0]))

    #### find the new position of each joints and move it
    for joint in member_joints[1:]:
        pos = np.array([x_joints[joint], y_joints[joint], z_joints[joint]]).reshape((3,1))
        pos = np.dot(r_2, np.dot(r_1, pos))
        pos = pos + origin_pos

        joints_data = change_3d_joint_pos(joints_data, body_side+' '+joint, pos.reshape((3)))
    
    return joints_data, angles_dic
