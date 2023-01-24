#generate the dataset using multiprocessing 
from multiprocessing import Pool
from utils import *

if __name__ == "__main__":

    video_indexes = [i for i in range(1,41)]
    output_folder = "/scratch/tathagato/DREYEVE_DATA_OUTPTUS"
    dataset_folder = "/scratch/tathagato/DREYEVE_DATA"

    #do multiprocessing 
    pool = Pool(10)
    pool.map(parse_video, video_indexes)
    pool.close()
    pool.join()


