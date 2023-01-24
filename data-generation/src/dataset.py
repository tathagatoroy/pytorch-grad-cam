""" code to handle the dataset that is the DR(eye)VE dataset """

#import the necesary imports
import torch 
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import os
import sys
import cv2 
from utils import *
from config import *
import csv

class DREYEVE(torch.utils.data.Dataset):
    def __init__(self,csv_path):
        """ class which builds the dataset object for gaze estimation,
        each index return   
            garmin_image : A RGB image taken from the dashboard
            saliency_image : A GRAYSCALE image with gaze annotations
        """
        self.csv_path = csv_path
        #self.X = [] #stores the garmin image
        #self.Y = [] #stores the saliency image
        self.paths = []
        self.size = 0

        self.get_dataset_images()

    

    def get_dataset_images(self):
        csv_reader = csv.reader(open(self.csv_path,'r'))
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                self.paths.append(row)
            line_count += 1 
        print("number of paths is {0}".format(len(self.paths)))
        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self,index):
        
        path = self.paths[index]
        garmin_path = path[0]
        saliency_path = path[1]
        garmin_image = cv2.cvtColor(cv2.imread(garmin_path, cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)
        garmin_image = garmin_image.astype(np.float32) / 255
        saliency_image = cv2.cvtColor(cv2.imread(saliency_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2GRAY)
        saliency_image = saliency_image.astype(np.float32) / 255
        return garmin_image, saliency_image
        




