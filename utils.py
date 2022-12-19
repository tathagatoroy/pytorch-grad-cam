import cv2
import numpy as np
import os
import sys
import torch
import torchvision
from config import *

COLORS = np.random.uniform(0,255,size = (len(coco_names), 3))
def predict(input_tensor, model, device, detection_threshold):
    outputs = model(input_tensor)
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

    boxes, classes, labels, indices = [],[],[],[]\

    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_bboxes[index].astype(np.int32))
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)

    boxes = np.int32(boxes)
    return boxes, classes, labels, indices


def draw_boxes(boxes, labels, classes, image):
    print("Number of objects : {0}".format(len(boxes)))
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(image, (int(box[0]), int(box[1])),(int(box[2]),int(box[3])),color, 2)
        cv2.putText(image,classes[i], (int(box[0]),int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color , 2 , lineType = cv2.LINE_AA)
    return image

def load_image(img_path):
    #print(img_path)
    rgb = cv2.cvtColor(cv2.imread(img_path,cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, dsize = None, fx = 0.50, fy = 0.50)
    return rgb

def save_RGB_image(img,path):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def generate_paths(output_directory,video_index,image_index):
    #print(output_directory,image_index,video_index)
    video_directory = os.path.join(output_directory,video_index)
    image_directory = os.path.join(video_directory,str(image_index))
    if not os.path.isdir(image_directory):
        os.makedirs(image_directory)
    gray_map_path = os.path.join(image_directory,"mask.npy")
    bounding_box_path = os.path.join(image_directory,"box.jpg")
    visual_path = os.path.join(image_directory, "final.jpg")

    return gray_map_path, bounding_box_path, visual_path

