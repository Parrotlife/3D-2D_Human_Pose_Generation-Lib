# 3D-2D Human Pose Generation Library (HPGL) 
 This repository is a 3D-2D human pose joints generation Library. It can generate 2D and 3D joints of defined human pose with the corresponding joint angles.
 This can be used to train 3D joints estimation networks as shown in the example notebooks. There are 2 implementation of the library, one to generate in numpy, and one in Pytorch to take advantage of GPU processing. This library was created and used in the master thesis project of Jean Marc Bejjani. [link-to-project](https://github.com/Parrotlife/posture-app)

## Overview
- [Code Structure](#Code-Structure)
- [Tutorial](#tutorial)
- [Requirements](#Requirements)

## Code Structure
The code is structured as follow:

```
networks/ - Contains trained network for the example

lib/ - HPGL Library for data generation
    pose_2D.py - contains functions for 2D joints manipulation
    pose_3D.py - contains functions for 3D joints generation and manipulation
    pose_2D/3D_torch.py - equivalence in pytorch
    visuals.py - allows to visualize the 3D and 2D joints with the chart_studio plotting library

notebooks/ - Contains the notebooks for generating data and show a network training example
    example/ - Contains the training of a network to predict 3D joints angles from 2D
        v1 - trained for predicting head, back and absolute orientation angles.
        v2 - added the arms
        v3 - trained on data collected from the kinect
    data_generation.ipynb - tutorial to generate the data
```
## Tutorial

A detailed tutorial is presented in the data_generation.ipynb notebook.


## Requirements

The code was run on python 3.7 on Ubuntu 18.04, MacOS and Windows 10.

requirements:
```
numpy
torch
PIL
chart_studio
matplotlib
scipy
```