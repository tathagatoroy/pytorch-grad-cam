""" file for testing code """
from utils import *
import time 
from dataset import DREYEVE

def test_parse_video():
    #output_folder = "../../DREYEVE_DATA_OUTPUTS/"
    video_index = 1
    #dataset_folder = "../../DREYEVE_DATA/"
    t1 = time.time()
    print("starting parsing video")
    parse_video(video_index)
    print("Done , time taken : {0}".format(time.time() - t1))
    
    
dataset = DREYEVE("/scratch/tathagato/train.csv")
print(len(dataset))
x,y = dataset[0]
print(x.shape)
print(y.shape)
print(x.min(),x.max(),y.min(),y.max())
