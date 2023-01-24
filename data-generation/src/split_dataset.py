#split the dataset into test, train and val 
import os
from sklearn.model_selection import train_test_split
import csv 

test_file = "/scratch/tathagato/test.csv"
train_file = "/scratch/tathagato/train.csv"
val_file = "/scratch/tathagato/val.csv"
output_directory = "/scratch/tathagato/DREYEVE_DATA_OUTPUT"
NUM_FRAMES = 7501

def get_image_path_from_id(index):
    """ index is of the form 3.40(str)  which indicates 3 video 40 index """
    video_index = int(index.split(".")[0])
    frame_index = index.split(".")[1]
    video_name = None
    if video_index < 10:
        video_name = "0" + str(video_index)
    else:
        video_name = str(video_index)
    directory_path = os.path.join(output_directory,video_name)
    garmin_image_path = os.path.join(os.path.join(directory_path,"garmin"),frame_index + ".jpg")
    saliency_image_path = os.path.join(os.path.join(directory_path,"saliency"),frame_index + ".jpg")
    return garmin_image_path, saliency_image_path

def split_files(num_videos = 40):
    """ return 3 files with path to the respective images distributed randomly 
    each entry in  the list contains video index and frame number ,
    now the assumption is each video contains 7501 frames"
    """
    test_f = open(test_file,"w")
    val_f = open(val_file,"w")
    train_f = open(train_file,"w")


    #generate indices for each frame, that is if video index is 3 and index 4201 then the indice is "3.4201"
    indices = []
    for i in range(1,num_videos + 1):
        for j in range(0,7501):
            indices.append(str(i) + "." + str(j))

    train, test_val = train_test_split(indices,test_size = 0.2)
    test,val = train_test_split(test_val,test_size = 0.5)
    print("Number of Train Pairs : {0}".format(len(train)))
    print("Number of Test Pairs : {0}".format(len(test)))
    print("Number of Val Pairs : {0}".format(len(val)))

    train_writer = csv.writer(train_f)
    test_writer = csv.writer(test_f)
    val_writer = csv.writer(val_f)
    header = ["garmin", "saliency"]
    train_writer.writerow(header)
    test_writer.writerow(header)
    val_writer.writerow(header)

    for example in train:
        garmin_image_path,saliency_image_path = get_image_path_from_id(example)
        row = [garmin_image_path,saliency_image_path]
        train_writer.writerow(row)

    for example in test:
        garmin_image_path,saliency_image_path = get_image_path_from_id(example)
        row = [garmin_image_path,saliency_image_path]
        test_writer.writerow(row)

    for example in val:
        garmin_image_path,saliency_image_path = get_image_path_from_id(example)
        row = [garmin_image_path,saliency_image_path]
        val_writer.writerow(row)


    train_f.close()
    test_f.close()
    val_f.close()

    

     
if __name__ == "__main__":
    split_files(40)






