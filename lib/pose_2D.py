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
def normalize(data, shift=None):
    
    
    #we find the shift to center around the hip
    if shift is None:
        shift = (data[:,joint_dict['right hip']] + data[:,joint_dict['left hip']]).reshape((2,1))/2
    
    #we find the ratio to scale down
    ratio = (np.max(data[1,:]-np.min(data[1,:])))
    #ratio = find_limb_length(data, 'hip')/0.1739
    
    # we center and scale the joints
    data = (data[:,:]-shift)/ratio

    return data#.transpose().flatten()

"""Convert a tensor in pixel coordinate to absolute camera coordinates
   It accepts lists or torch/numpy tensors of (m, 2) or (m, x, 2)
   where x is the number of keypoints"""
def pixel_to_camera(uv_tensor, kk, z_met):

    if isinstance(uv_tensor, (list, np.ndarray)):
        uv_tensor = torch.tensor(uv_tensor)
    if isinstance(kk, list):
        kk = torch.tensor(kk)
    if uv_tensor.size()[-1] != 2:
        uv_tensor = uv_tensor.permute(0, 2, 1)  # permute to have 2 as last dim to be padded
        assert uv_tensor.size()[-1] == 2, "Tensor size not recognized"
    uv_padded = F.pad(uv_tensor, pad=(0, 1), mode="constant", value=1)  # pad only last-dim below with value 1

    kk_1 = torch.inverse(kk)
    xyz_met_norm = torch.matmul(uv_padded, kk_1.t())  # More general than torch.mm
    xyz_met = xyz_met_norm * z_met

    return xyz_met

"""convert the joints order of pifpaf to match the ones we use"""
def convert_pifpaf(og_keypoints):
    #this is the joint equivalence sequence between pifpaf anf H3.6M
    convert_seq = {0:0, 1:6, 2:8, 3:10, 4:5, 5:7, 6:9, 7:12, 8:14, 9:16, 10:11, 11:13, 12:15, 13:2, 14:1, 15:4, 16:3}
    
    #for each joint we apply the equivalence
    keypoints = []
    
    for i in range(17):
        keypoints.append(og_keypoints[:,convert_seq[i]])
    
    keypoints = np.array(keypoints).flatten()
    nb_joints = int(len(keypoints)/2)
    keypoints = keypoints.reshape((nb_joints,2)).transpose()

    r_hip = keypoints[:, joint_dict['right hip']]
    l_hip = keypoints[:, joint_dict['left hip']]
    hip = (r_hip + l_hip)/2
    
    r_ear = keypoints[:, joint_dict['right ear']]
    l_ear = keypoints[:, joint_dict['left ear']]
    head = (r_ear + l_ear)/2
    
    r_shoulder = keypoints[:, joint_dict['right shoulder']]
    l_shoulder = keypoints[:, joint_dict['left shoulder']]
    shoulder = (r_shoulder + l_shoulder)/2
    
    back = (shoulder + hip)/2
    shoulder = (shoulder+head)/2

    keypoints = np.append(keypoints, shoulder.reshape(2,1), axis = 1)
    keypoints = np.append(keypoints, hip.reshape(2,1), axis = 1)
    keypoints = np.append(keypoints, back.reshape(2,1), axis = 1)
    keypoints = np.append(keypoints, head.reshape(2,1), axis = 1)
    
    keypoints = pixel_to_camera(keypoints.transpose(), torch.tensor(kk), 1).numpy().transpose()[:2]

    keypoints = normalize(keypoints)

    return(keypoints.transpose().flatten())

"""scale an image and convert it to jpg"""
def scale_and_convert(image_path, max_size):
    if image_path[-3:] in ['jpg','png']:
        
        scaled = False
        converted = False
        im = Image.open(image_path)
        w, h = im.size
        
        if image_path[-3:] != 'jpg':
            im = im.convert('RGB')
            converted = True
        
        if (w*h) > max_size:
            scale = max_size/(w*h)
            im = im.resize((int(w*scale), int(h*scale)), Image.ANTIALIAS)
            scaled = True
        
        if scaled | converted:
            os.remove(image_path)
            im.save(image_path[:-3]+'jpg', quality=90)

"""load all pifpaf output files in a directory"""
def load_pifpaf(path):
    
    #we load all the output files' names
    labels_names=os.listdir(path)
    labels_names.sort()
    #we then create a list of dictionnaries holding all relevent data to a pedestrian
    pifpafbox = []
    for i, label in enumerate(labels_names):
        if label[-4:] == 'json':
            with open(path+label) as inputfile:
                temp = json.load(inputfile)

                if temp != []:
                    for person in temp:
                        new = {}
                        new['ppbox'] = [x for x in person['bbox']]

                        keyp = [x for x in person['keypoints']]

                        new['confidence'] = keyp[2::3]

                        del keyp[2::3]

                        new['image_id'] = label[0:6]

                        new['og_keypoints'] = np.array(keyp).reshape((int(len(keyp)/2),2)).transpose()

                        new['conv_keypoints'] = convert_pifpaf(new['og_keypoints'])

                        pifpafbox.append(new)
    print("we loaded",len(pifpafbox), "pedestrians")
    return pifpafbox

"""find the length of a limb from its name"""
def find_limb_length(data, limb_name):
    data = data.reshape((NB_JOINTS, 2)).transpose()
    return np.linalg.norm(data[:,limb_dict[limb_name][0]]-data[:,limb_dict[limb_name][1]])

"""find the distance between 2 joints from their names"""
def find_2_joints_distance(data, joint1, joint2):
    data = data.reshape((NB_JOINTS, 2)).transpose()
    return np.linalg.norm(data[:,joint_dict[joint1]]-data[:,joint_dict[joint2]])

""" get the position of a joint in 2d"""
def get_2d_joint_pos(data, joint_name):
    pedestrian = data.reshape((NB_JOINTS,2)).transpose()
        
    return pedestrian[:, joint_dict[joint_name]].reshape((2,1))

""" Change the position of a joint in 2d"""
def change_2d_joint_pos(data, joint_name, pos_2d):
    pedestrian = data#.reshape((NB_JOINTS,2)).transpose()
    pedestrian[:, joint_dict[joint_name]] = pos_2d
        
    return pedestrian#.transpose().flatten()

""" extract the proportions and positions of the body and facial 
features of a person from a front view and a side view keypoints"""
def get_proportions(front_view, side_view):
    body_prop = {}
    face_pos = {}
    body_prop_names = ['neck', 'shoulder', 'biceps','forearm', 'quad', 'tibias', 'hip', 'back']
    face_names = ['nose','ear','eye']
                 
    ## get the average limb proportions from the front view
    for limb in body_prop_names:
        temp_props = []
        
        for case in [x for x in limb_names if limb in x]:
            temp_props.append(find_limb_length(front_view, case))
        
        body_prop[limb] = np.mean(temp_props)

    
    ## get the nose, eyes and ears pos relatif to the head using the side view
    
    d_nose_head = find_2_joints_distance(side_view, 'nose', 'head')
    
    face_pos['nose'] = [0.0, 0.0, d_nose_head]
    
    nose_pos = get_2d_joint_pos(side_view, 'nose')
    head_pos = get_2d_joint_pos(side_view, 'head')
    eye_pos = np.mean([get_2d_joint_pos(side_view, 'right eye'),get_2d_joint_pos(side_view, 'left eye')], axis=0)
    
    real_nose_pos = np.array([0.0, nose_pos[1][0]-head_pos[1][0], nose_pos[0][0]-head_pos[0][0]])

    error_nose = real_nose_pos-np.array(face_pos['nose'])
        
    eye_pos = abs(eye_pos-head_pos) - error_nose
    
    z_eye = eye_pos[0]
    
    ## using the front view

    eye_pos = np.mean([abs(get_2d_joint_pos(front_view, 'right eye')),\
                       abs(get_2d_joint_pos(front_view, 'left eye'))], axis=0)
    
    head_pos = get_2d_joint_pos(front_view, 'head')
    
    x_eye = (eye_pos - abs(head_pos))[0]
    y_eye = (eye_pos - abs(head_pos))[1]
    
    face_pos['eye'] = [x_eye[0], y_eye[0], z_eye[0]]
    
    ear_pos = np.mean([abs(get_2d_joint_pos(front_view, 'right ear')),\
                       abs(get_2d_joint_pos(front_view, 'left ear'))], axis=0)
    
    x_ear = (ear_pos - abs(head_pos))[0]
    y_ear = (ear_pos - abs(head_pos))[1]
    
    face_pos['ear'] = [x_ear[0], y_ear[0], 0.0]
    
    return body_prop, face_pos
    
"""generate a standing model from the front and side view"""
def generate_3D_model(props, face):
    
    reconstructed_pose = []
    reconstructed_pose.append(face['nose'])

    head = [0,0,0]

    c_shoulder = np.array([0,+props['neck'],0])
    
    temp_shoulder = np.array([0,+2*props['neck'],0])

    r_shoulder = temp_shoulder + np.array([-props['shoulder'],0,0])
    l_shoulder = temp_shoulder + np.array([+props['shoulder'],0,0])

    r_elbow = r_shoulder + np.array([0,+props['biceps'],0])
    l_elbow = l_shoulder + np.array([0,+props['biceps'],0])

    r_wrist = r_elbow + np.array([0,+props['forearm'],0])
    l_wrist = l_elbow + np.array([0,+props['forearm'],0])

    c_hip = temp_shoulder + np.array([0,+2*props['back'],0])

    c_back = temp_shoulder + np.array([0,+props['back'],0])

    r_hip = c_hip + np.array([-0.5*props['hip'],0,0])
    l_hip = c_hip + np.array([+0.5*props['hip'],0,0])

    r_knee = r_hip + np.array([0,+props['quad'],0])
    l_knee = l_hip + np.array([0,+props['quad'],0])

    r_ankle = r_knee + np.array([0,+props['tibias'],0])
    l_ankle = l_knee + np.array([0,+props['tibias'],0])

    r_eye = [-face['eye'][0],-face['eye'][1],face['eye'][2]]
    l_eye = [face['eye'][0],-face['eye'][1],face['eye'][2]]

    r_ear = [-face['ear'][0],-face['ear'][1],face['ear'][2]]
    l_ear = [face['ear'][0],-face['ear'][1],face['ear'][2]]



    reconstructed_pose.extend([list(r_shoulder), list(r_elbow), list(r_wrist), list(l_shoulder),
                               list(l_elbow), list(l_wrist), list(r_hip), list(r_knee), list(r_ankle),
                               list(l_hip), list(l_knee), list(l_ankle), list(r_eye), list(l_eye), r_ear, l_ear, 
                               list(c_shoulder), list(c_hip), list(c_back), head])
    
    
    reconstructed_pose = pose3d.normalize(np.array(reconstructed_pose).transpose(), resize = False)
    
    return reconstructed_pose.transpose().flatten()

"""Filter joints according to a list of them"""
def filter_joints(data, list_joints, value):
    
    null_pos = np.array([value,value])
    
    joints_data = data.copy()
    
    for joint in list(set(joint_names)-set(list_joints)):
        joints_data = change_2d_joint_pos(joints_data, joint, null_pos)
        
    return joints_data