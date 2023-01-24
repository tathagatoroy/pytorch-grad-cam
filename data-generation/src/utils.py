
""" contains all form of utilties functionalities """
#import the necesary imports
import torch 
from torch.utils.data import dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import os
import sys
import cv2
import time

def parse_video(video_index, dataset_folder = "/scratch/tathagato/DREYEVE_DATA/",output_folder = "/scratch/tathagato/DREYEVE_DATA_OUTPUT"):
    """ given the video number in the dataset and the output folder,
    parse the the garmin video and the saliency video to individual frames in
    the output folder
    
    ARGS :
        output_folder : the folder which should store all the processed outputs
        video_index : the video number/index
        dataset_folder : the root dataset folder 
        
    Note that folder structure of dataset folder is 
    root -> video_index ->      
                    video_garmin : the roof mounted cam
                    video_saliency : ground truth 
                    video_etg : the eye tracking glasses video
    """
    # now the index if 12 is 12 but for 2 is 02
    # so we need to convert int to appropriate string format
    start_time = time.time()
    print("Parsing video with index : {0}".format(video_index))
    video_name = None
    if video_index < 10:
        video_name = str(0) + str(video_index)
    else:
        video_name = str(video_index)
    video_path = os.path.join(dataset_folder,video_name)
    if os.path.exists(video_path):
        #garmin_sequence = []
        #saliency_sequence = []
        #gray_saliency_sequence = []
        
        garmin_path = os.path.join(video_path,"video_garmin.avi")
        saliency_path = os.path.join(video_path,"video_saliency.avi")
        
        #create the output_folders from garmin and saliency videos and gray saliency videos
        output_garmin_path = os.path.join(os.path.join(output_folder,video_name),"garmin")
        output_saliency_path = os.path.join(os.path.join(output_folder,video_name),"saliency")
        output_gray_saliency_path = os.path.join(os.path.join(output_folder,video_name),"gray saliency")
        output_combined_path = os.path.join(os.path.join(output_folder,video_name),"combined")
        #get the garmin frames
        if not os.path.isdir(output_garmin_path):
            os.makedirs(output_garmin_path)
        garmin_camera = cv2.VideoCapture(garmin_path)
        current_frame = 0
        while(True):
            ret, frame = garmin_camera.read()
            if ret :
                frame_name = os.path.join(output_garmin_path,str(current_frame) + ".jpg")
                #garmin_sequence.append(frame)
                frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5)
                cv2.imwrite(frame_name, frame)
                current_frame += 1
                #print("Number of Frames : {0} ".format(current_frame))
            else:
                break
        garmin_camera.release()
        cv2.destroyAllWindows()
        print("Video {0} garmin done ".format(video_index))
        #get the saliency frames
        if not os.path.isdir(output_saliency_path):
            os.makedirs(output_saliency_path)
        saliency_camera = cv2.VideoCapture(saliency_path)
        current_frame = 0
        while(True):
            ret, frame = saliency_camera.read()
            if ret :
                frame_name = os.path.join(output_saliency_path,str(current_frame) + ".jpg")
                gray_frame_name = os.path.join(output_gray_saliency_path,str(current_frame) + ".jpg")
                #gray_frame = cv2.cvtColor(np.copy(frame),cv2.COLOR_BGR2GRAY)
                #saliency_sequence.append(frame)
                #gray_saliency_sequence.append(gray_frame)
                
                frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5)
                cv2.imwrite(frame_name, frame)
                #cv2.imwrite(gray_frame_name,frame)
                current_frame += 1
                #print("Number of Frames : {0} ".format(current_frame))

            else:
                break
        saliency_camera.release()
        cv2.destroyAllWindows()
        """
        size = len(saliency_sequence)
        for i in range(size):
            img1 = saliency_sequence[i]
            img2 = garmin_sequence[i]
            alpha = 0.5
            beta = 0.5
            combined = cv.addWeighted(img1, alpha, img2, beta, 0.0)
            frame_path = os.path.join(output_combined_path,str(i) + ".jpg")
            cv2.imwrite(frame_path,combined)
        """
        end_time = time.time()
        total_time = end_time - start_time
        print("Saliency done for video {0} , total time taken : {1}".format(video_index,total_time))


            


            
                
                    
            
        
         
    
