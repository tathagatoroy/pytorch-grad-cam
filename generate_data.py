
""" code to generate data and object annotations for experiments """

import torch
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import sys
import cv2
import os
import numpy as np
import torchvision
from pytorch_grad_cam import EigenCAM,EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels , scale_cam_image
from config import *
from utils import *
from multiprocessing import Pool
import json
import pickle
import argparse 
import time
import random
import pytorch_grad_cam
import matplotlib.pyplot as plt
from metric import *

LOCAL_DATASET = "./../DREYEVE_DATA_OUTPUT/"
ADA_DATASET = "/scratch/tathagato/DREYEVE_DATA_OUTPUT"
TOP = 150
BOTTOM = 330
RIGHT = 560
LEFT = 380
GAP = 10
dataset = ADA_DATASET

""" DATASET STRUCTURE 
    video number ->
            garmin 
                x.jpg
            saliency
                x.jpg
            image_number:
                images
""" 
""" data processing 
    classes = [car, dog]
    box = [[l11,l12,l13,l14],[l21,l22,l23,l24]]
"""


def load_image(img_path, transform = 1):
    """
    input is an image path and transform code
    returns an image corresponding to the path
    1 : BGR 2 RGB
    2 : BGR 2 GRAY
    3 : GRAY to GRAY
    """
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if transform == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif transform == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def show_images(images):
    """ given a list of images 
        shows the image in figure"""
    size = len(images)
    fig = plt.figure(figsize = (30,30))
    col = size // 2 + size % 2
    for i in range(size):
        fig.add_subplot(col,2,i + 1)
        img = images[i]
        if len(img.shape) == 3:
            plt.imshow(img)
        else:
            plt.imshow(img,cmap = "gray")
            
        
    
    
def get_image_pair_from_indexes(video_index, image_index):
    """ given an video index and image index, returns the garmnin and saliency image """
    video = str(video_index)
    if video_index < 10:
        video = "0" + str(video_index)
    image = str(image_index) + ".jpg"
    garmin_image_path = os.path.join(os.path.join(dataset,os.path.join(video,"garmin")),image)
    saliency_image_path = os.path.join(os.path.join(dataset,os.path.join(video,"saliency")),image)
    garmin_image = load_image(garmin_image_path, 1)
    saliency_image = load_image(saliency_image_path,1)
    
    return garmin_image, saliency_image

def get_random_image_pair():
    """ generates a random sample of the dataset """
    video_index = random.choice(video_indexes)
    image_index = random.choice(image_indexes)
    garmin, saliency = get_image_pair_from_indexes(video_index, image_index)
    return garmin, saliency

def get_random_images(num):
    """draws num samples of random"""
    img_pairs = []
    for i in range(num):
        garmin, saliency = get_random_image_pair()
        img_pairs.append([garmin, saliency])
    return img_pairs


def show_image_pairs(img_pairs):
    """ display list of image pairs """
    fig = plt.figure(figsize = (20,50))
    size = len(img_pairs)
    for i in range(size):
        garmin,saliency = img_pairs[i][0], img_pairs[i][1]
        fig.add_subplot(size,2,2*i + 1)
        plt.imshow(garmin)
        fig.add_subplot(size,2,2*i + 2)
        plt.imshow(saliency)



def draw_lines(img,height = 350):
    """ for experimentation, draws a thick line for removing the front cam """
    img[height-10:height+10,:,:] = 0
    return img
def draw_lines_img_pair(img_pair,height = 350):
    new_img_pair = []
    for images in img_pair:
        garmin,saliency = images[0],images[1]
        garmin = draw_lines(garmin,height)
        new_img_pair.append([garmin,saliency])
    return new_img_pair

def cut_image(img, height = 350):
    "to remove dashboard, cut the image below dashboard,provided height"
    new_img = np.zeros((height, img.shape[1], img.shape[2]),dtype = img.dtype)
    new_img = img[0:height,:,:]
    return new_img

def central_cut(img):
    return img[TOP:BOTTOM,LEFT:RIGHT,:]

    
def generate_eigenmap(img):
    image_float_np = np.float32(img) / 255
    # define the torchvision image transforms
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    info = {}
    input_tensor = transform(img)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device("cpu")
    input_tensor = input_tensor.to(device)
    input_tensor = input_tensor.unsqueeze(0)
    boxes, classes, labels, indices = predict(input_tensor, model, device , 0.7)
    #print(input_tensor.device, input_tensor.dtype)
    num_object = len(boxes)
    info['size'] = num_object
    objects = {}
    for i in range(num_object):
        box = boxes[i]
        cls = classes[i]
        label = labels[i]
        this_object = {}
        list_box = list(box)
        indice = indices[i]
        new_box = [int(i) for i in list_box]
        this_object['box'] = new_box 
        this_object['class'] = cls
        this_object['label'] = int(label)
        this_object['indice'] = int(indice)
        objects[i] = this_object
    info['objects'] = objects
    box_image = draw_boxes(boxes,labels,classes,img)
    targets = [FasterRCNNBoxScoreTarget(labels = labels,bounding_boxes = boxes)]
    #print(input_tensor.dtype)
    grayscale_cam = cam(input_tensor, targets)
    grayscale_cam = grayscale_cam[0,:]
    final_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb = True)
    return grayscale_cam,final_image,box_image,info

    

def process_image(video_id,image_id):

    """ given image and video index, both in integer do the necessary """
    garmin,saliency = get_image_pair_from_indexes(video_id, image_id)
    #note both the image is now in RGB 
    cut_garmin, cut_saliency = cut_image(np.copy(garmin)), cut_image(np.copy(saliency))
    central_garmin, central_saliency = central_cut(np.copy(garmin)), central_cut(np.copy(saliency))
    garmin, saliency = cv2.resize(garmin, dsize = None, fx = 0.50, fy = 0.50), cv2.resize(saliency, dsize = None, fx = 0.50, fy = 0.50)
    normal_mask, normal_final, normal_box, normal_info = generate_eigenmap(garmin)
    cut_mask, cut_final, cut_box, cut_info = generate_eigenmap(cut_garmin)
    central_mask, central_final, central_box, central_info = generate_eigenmap(central_garmin)
    image_info = {}
    image_info['image_id'] = image_id
    image_info['video_id'] = video_id
    image_info['normal'] = normal_info
    image_info['cut'] = cut_info
    image_info['central'] = central_info

    video_name = str(video_id)
    if video_id <= 9:
        video_name = "0" + video_name
    video_path = os.path.join(ADA_DATASET,video_name)
    if not os.path.isdir(video_path):
        os.makedirs(video_path)
    image_path = os.path.join(video_path,str(image_id))
    if not os.path.isdir(image_path):
        os.makedirs(image_path)
    
    # save the normal images
    save_RGB_image(normal_final, os.path.join(image_path, "normal_final.jpg"))
    save_RGB_image(normal_box, os.path.join(image_path, "normal_box.jpg"))
    np.save(os.path.join(image_path,"normal_mask.npy"),normal_mask,allow_pickle = True)
    
    # save the cut images
    save_RGB_image(cut_final, os.path.join(image_path, "cut_final.jpg"))
    save_RGB_image(cut_box, os.path.join(image_path, "cut_box.jpg"))
    np.save(os.path.join(image_path,"cut_mask.npy"),cut_mask,allow_pickle = True)
    
    # save the central images
    save_RGB_image(central_final, os.path.join(image_path, "central_final.jpg"))
    save_RGB_image(central_box, os.path.join(image_path, "central_box.jpg"))
    np.save(os.path.join(image_path,"central_mask.npy"),central_mask,allow_pickle = True)
    
    image_info_path = os.path.join(image_path, "image_info.json")
    f = open(image_info_path,"w")
    json.dump(image_info, f, indent = 4)
    

def process_video(video_id):
    print("Processing Video ",video_id)
    for i in range(0,7500):
        if i % GAP == 0:
            process_image(video_id, i)
        if i % 200:
            print("Currently processing frame {0} from video {1}".format(i,video_id))


def process_videos(video_list):
    for video in video_list:
        process_video(video)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
#print(device)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
target_layers = [model.backbone]
cam = EigenCAM(model,target_layers, use_cuda = torch.cuda.is_available() ,reshape_transform = fasterrcnn_reshape_transform)
#cam = EigenCAM(model,target_layers, use_cuda = False ,reshape_transform = fasterrcnn_reshape_transform)

model.eval().to(device)
#video_list = [1,2]
video_list = [i for i in range(11,16)]
process_videos(video_list)
