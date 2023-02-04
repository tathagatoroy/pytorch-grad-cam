import torch
import torchvision
import os
import sys
import time
import cv2
import numpy as np
import pytorch_grad_cam
import pickle
import json
import matplotlib.pyplot as plt

filepath = "/scratch/tathagato/DREYEVE_DATA_OUTPUT/01/4950/box.jpg"
img = cv2.imread(filepath,cv2.IMREAD_UNCHANGED)
plt.imshow(img)