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
import time
from config import *
from utils import *
from multiprocessing import Pool
import json
import pickle
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]= "2"

FRAMES = 7501
def generate_eigenmaps(combined,data,output_directory = "/scratch/tathagato/DREYEVE_DATA_OUTPUT"):
    #make video indiex string
    image_dict = {}
    img_path , video_index , image_index = combined
    print(img_path)
    new_video_index = None
    if video_index > 9:
        new_video_index = str(video_index)
    else:
        new_video_index = "0" + str(video_index)
    #print(new_video_index)
    mask_path , box_path, final_path = generate_paths(output_directory,new_video_index, image_index)
    
    img = load_image(img_path)

    image_float_np = np.float32(image) / 255
    # define the torchvision image transforms
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    input_tensor = transform(img)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device("cpu")
    input_tensor = input_tensor.to(device)
    # Add a batch dimension:
    input_tensor = input_tensor.unsqueeze(0)

    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    #model.eval().to(device)

    # Run the model and display the detections
    #boxes, classes, labels, indices = predict(input_tensor, model, device, 0.7)
    #img = draw_boxes(boxes, labels, classes, img)

    # Show the image:
    #target_layers = [model.backbone]
    #targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
    #cam = EigenCAM(model,
    #            target_layers, 
    #            use_cuda=torch.cuda.is_available(),
    #            reshape_transform=fasterrcnn_reshape_transform)

    #grayscale_cam = cam(input_tensor, targets=targets)
    # Take the first image in the batch:
    #grayscale_cam = grayscale_cam[0, :]
    #cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
    # And lets draw the boxes again:
    #image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image)

    #img = load_image(img_path)
    #img_float_np = np.float32(img) / 255
    #transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
    #input_tensor = transform(img_float_np)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)
    #input_tensor = input_tensor.to(device)
    #print(input_tensor.device, device)
    #add batch dimension
    #input_tensor = input_tensor.unsqueeze(0)
    #print(input_tensor.device,input_tensor.dtype)
    #model.eval().to(device)
    boxes, classes, labels, indices = predict(input_tensor, model, device , 0.7)
   
    num_objects = len(boxes)
    if num_objects == 0:
        image_dict['index'] = image_index
        image_dict['size'] = 0
        data[image_index] = image_dict
        return 
    image_dict['index'] = image_index
    image_dict['size'] = num_objects
    objects = {}
    for i in range(num_objects):
        box = boxes[i]
        cls = classes[i]
        label = labels[i]
        indice = indices[i]
        this_object = {}
        list_box = list(box)
        new_box = [int(i) for i in list_box]
        this_object['box'] = new_box 
        this_object['class'] = cls
        this_object['label'] = int(label)
        this_object['indice'] = int(indice)
        objects[i+1] = this_object
    image_dict['objects'] = objects
    data[image_index] = image_dict


        
    box_image = draw_boxes(boxes,labels,classes,img)
    #target_layers = [model.backbone]
    targets = [FasterRCNNBoxScoreTarget(labels = labels,bounding_boxes = boxes)]
    grayscale_cam = cam(input_tensor, targets)
    grayscale_cam = grayscale_cam[0,:]
    print(grayscale_cam.max(), grayscale_cam.min())
    #grayscale_cam = 1 - grayscale_cam
    final_image = show_cam_on_image(img_float_np, grayscale_cam, use_rgb = True)
    save_RGB_image(box_image, box_path)
    save_RGB_image(final_image, final_path)
    #cv2.imwrite(mask_path, grayscale_cam)
    np.save(mask_path,grayscale_cam,allow_pickle = True)



def process_videos(video_index, dataset_directory):
    str_video_index = None
    if video_index > 9:
        str_video_index = str(video_index)
    else:
        str_video_index = "0" + str(video_index)

    image_paths = [dataset_directory + "/" + str_video_index + "/garmin/" + str(i) + ".jpg" for i in range(0,FRAMES)]
    video_indexes = [video_index for i in range(0,FRAMES)]
    image_indexes = [i for i in range(0,FRAMES)]
    #print(image_paths)
    #print(video_indexes)
    combined = []
    data = {}
    for i in range(0,FRAMES):
        combined.append((image_paths[i],video_indexes[i],image_indexes[i]))
    saved_path = os.path.join(dataset_directory,str_video_index)
    saved_path = saved_path + "/data.json"
    print(saved_path)
    f = open(saved_path, 'w')
    #json.dump(data
    #print(combined)

    #for i in range(7501):
        #img_path = image_paths[i]
        #print(img_path)
        #generate_eigenmaps(img_path,video_index,i,dataset_directory)
    #pool = Pool(10)
    #pool.map(generate_eigenmaps,combined)
    #pool.close()
    #pool.join()
    start = time.time()
    for i in range(FRAMES):
        if i % GAP == 0:
            generate_eigenmaps(combined[i],data)
        cur = time.time()
        print("Rates : {0}".format((i+1)/(cur-start)))
    #print(data)
    json.dump(data, f, indent = 4)


GAP = 10
time1 = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
target_layers = [model.backbone]
cam = EigenCAM(model,target_layers, use_cuda = torch.cuda.is_available(),reshape_transform = fasterrcnn_reshape_transform)

model.eval().to(device)
dataset_directory = "/scratch/tathagato/DREYEVE_DATA_OUTPUT"
    #target_layers = [model.backbone]


for i in range(1,2):
    process_videos(i,dataset_directory)




   







