import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import sys
import cv2
import os
import numpy as np
import torch
import torchvision
from pytorch_grad_cam import EigenCAM,EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels , scale_cam_image
