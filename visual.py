import chart_studio
import chart_studio.plotly as py
import plotly.graph_objs as go
import copy as cp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
chart_studio.tools.set_credentials_file(username='Parrotlife', api_key='HVithFEqzxhtolNyW0jU')

"""
These are the order of the joints expected as input
 'nose': 0,
 'right shoulder': 1,
 'right elbow': 2,
 'right wrist': 3,
 'left shoulder': 4,
 'left elbow': 5,
 'left wrist': 6,
 'right hip': 7,
 'right knee': 8,
 'right ankle': 9,
 'left hip': 10,
 'left knee': 11,
 'left ankle': 12,
 'right eye': 13,
 'left eye': 14,
 'right ear': 15,
 'left ear': 16,
 'center shoulder': 17,
 'center hip': 18,
 'center back': 19,
 'head': 20


"""

NB_JOINTS = 21

"""this is the function to plot any list of pedestrians in 2d or 3d using char_studio"""
def plot(list_pedestrians):
    
    pedestrians = cp.deepcopy(list_pedestrians)
    data = []
    for i, pedestrian in enumerate(pedestrians):
        
        dim = int(len(pedestrian)/NB_JOINTS)
        
        pedestrian = pedestrian.reshape((NB_JOINTS,dim)).transpose()
        
        #we plot pedestrians in a gradiant from blue to red
        color_1 = np.array([0,0,250])
        color_2 = np.array([250,0,0])
        
        color_gap = ((color_2 - color_1)/len(pedestrians)).astype(int)
        
        color = f"#{''.join(f'{hex(c)[2:].upper():0>2}' for c in (tuple(i*color_gap + color_1)))}"
        
        
        plot2d = False
        
        #these are the joint connections to draw
        connections = [(0,20),(0,13),(0,14),(13,15),(14,16),(20,17),(17,1),(1,2),(2,3),(17,4),(4,5),(5,6),(17,19),(19,18),
                       (18,7),(7,8),(8,9),(18,10),(10,11),(11,12)]
        #we check the dimension of the input
        name = '3D pose'

        if dim!=3:
            plot2d = True
            name = '2D pose'
        keypoints = pedestrian
        #we find the scale of each axis to keep the proportions
        x_scale = max(keypoints[0])-min(keypoints[0])
        y_scale = max(keypoints[1])-min(keypoints[1])
        z_scale = 1
        if not plot2d:
            z_scale = max(keypoints[2])-min(keypoints[2])
        
        
        
        #we find xyz for each tuple of joints to draw
        for i, j in connections:

            x = [keypoints[0][i],keypoints[0][j]]
            y = [keypoints[1][i],keypoints[1][j]]

            if not plot2d: 
                z = [keypoints[2][i],keypoints[2][j]]

            if plot2d:
                trace = go.Scatter(x=x, y=y, marker=dict(size=4,color=color), line=dict(color=color,width=2))
            else:
                trace = go.Scatter3d(x=x, y=y, z=z, marker=dict(size=4,color=color), line=dict(color=color,width=2))

            data.append(trace)

            
            
    #we define the layout for the figure        
    axis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        )
    zaxis = axis
    
    if plot2d:
        zaxis = None
        
    layout = dict(
    width=800,
    height=1000,
    autosize=True,
    showlegend=False,
    title=name,
    scene=dict(xaxis=axis, yaxis=axis, zaxis=zaxis,
        camera=dict(
            up=dict(
                x=0,
                y=0,
                z=0
            ),
            eye=dict(
                x=-1.7428,
                y=1.0707,
                z=0.7100,
            )
        ),
        aspectratio = dict( x=x_scale, y=y_scale, z=z_scale ),
        aspectmode = 'manual'
        ),
    )
    
    #and we finaly return the figure
    fig = dict(data=data, layout=layout)
    
    return fig

"""function to display pifpaf joints on the kitti corresponding image"""
def show_pp_on_image(pedestrian, datapath='../../jeanmarc_data/data/kitti-images/data/', im_format = '.png'):
    
    im = np.array(Image.open(datapath+pedestrian['image_id']+im_format), dtype=np.uint8)
    
    scale_r = 0.09

    # Create figure and axes
    fig,ax = plt.subplots(1, 2,figsize=(15,3.5))

    # Display the image
    
    box = pedestrian['ppbox']

    box = list(map(lambda x: 0 if x<0 else x, box))

    shift = np.array([int((1-scale_r)*box[0]),int((1-scale_r)*box[1])])
    
    ax[0].imshow(im[int((1-scale_r)*box[1]):int((1+scale_r)*box[3]),int((1-scale_r)*box[0]):int((1+scale_r)*box[2])])
    ax[1].imshow(im)
    

    kp = np.array(pedestrian['og_keypoints'].transpose())
    
    centered_kp=pedestrian['og_keypoints'].transpose()-shift
    
    rect2 = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],linewidth=2,edgecolor='g',facecolor='none')


    ax[0].scatter(centered_kp[:,0],centered_kp[:,1],c='r')
    # Add the patch to the Axes
    ax[1].add_patch(rect2)
    plt.tight_layout()