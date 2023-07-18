### Masterfile from jpg to dataframe with Eigenvector turn and width
### TO DO YAML data and all options with standard and advanced settings
### Test runs, Guarantee the import of functions work
### Clean files as well as useless code and ML Code
### Write MA

import tensorflow as tf
import cv2 as cv2
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os
import shutil
import subprocess
import pandas as pd
from TifzuJPEG import ConvertTiftoJPEG
from Yaml_load  import ConfigImport
from Annotations_Post_process import JsonToMeasurement
from DaphnidMeasure import DaphnidMeasurements
from Scale_detect_EasyOCR import Images_list, getLineLength, group_lines, RoughCrop, CropImage,Sortlist,detect_Number, makeDfwithfactors
from DataframeMerge import FuseDataframes, ShowMeasureImage
from BodyWidthChooser import Perform_Chosen_method
from Image_cropper import CropImages
from ClassificationDeploy import Classify_Species

# Get the current file's directory
current_directory = os.path.dirname(os.path.abspath(__file__))
print()
### 1) Yaml_load_test.py: Read all the settings in our yaml and save them in variables
settings = ConfigImport(current_directory + "/settings_segment.yml")

### Paths
Original_Images = settings["Original_images"] 
Analysis_data = settings["Analysis_data"]
Classifier_loc = settings["Classifier_loc"]
### Image Converter
Convert_to_JPEG = settings["Convert_to_JPEG"]

## Ginjinn settings
visualize = settings["visualize"]  
Crop = settings["crop"]
refine = settings["refine"]
Classify = settings["Classify"]
# Measurement visualization
detection_vis = settings["detection_vis"]
Body_width_eval = settings["Body_width_eval"]
organs = settings["organs"]

## Scale settings
Conv_factor = settings["Conv_factor"]
scale_mode = int(settings["Scale_detector_mode"])
psm_mode = int(settings["psm_mode"])

#Body width method

Width_method = settings["Width_method"]
#### Create the final data saveloc: Here we will find the final images and csv's
#### If the folder exists we delete and overwerite it
### 2) TifzuJPEG.py: Change all Images to JPEG, name misleading changes not only tifs 
if Convert_to_JPEG == True:
  
  ConvertTiftoJPEG(Original_Images, Original_Images)
  Original_Images = Original_Images + "/JPG"



## Find path to original images and names of the images
Paths_of_Images, Name_of_Images = Images_list(Original_Images)

### Now we have a dataset of the size 11xn on which we perform the full detection
### 6) FullDetect: Detect Eye, Body, Daphnid, Spina base, Spina tip, heart, head and brood chamber
mode = "" 
if detection_vis == True:
  mode += "-v "

predict_command = f'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ginjinn && ginjinn predict ~/Master_thesis_data/Model_Data_MA/Loose_box_final_2204 -i {Original_Images} -o {Analysis_data} {mode}'
subprocess.run(predict_command, shell=True, executable='/bin/bash')

### Now the user needs to check the bounding boxes of his unique Images
### and correct errors. After that we can start the evaluation

response = input(f"Do you want to check your labels? This is highly advised. Extract your data from {Analysis_data} [y(es)/n(o)/c(ancel)]")

if response == 'c':
    # Cancel the code
    print("Aborting...")
    exit()
elif response == 'y':
    # Pause the code
    input("Press enter to continue...")
else:
    # Continue the code
    pass

## Detect scale
CleanUnits = []
Lines = []
if int(scale_mode) != 0:
  
  Lengths, Line_Coors, List_of_images = getLineLength(Paths_of_Images) ### Line lengths and lower right aprt of image
  Rough_Images = RoughCrop(Line_Coors, List_of_images) #### Makes one number or list out of list(list(n,n1,n2), list(n,n1,n2),...)
  Small_Images = CropImage(Rough_Images)
 
  start_time = time.time()
  Detected_Numbers = detect_Number(Small_Images)
  elapsed_time = time.time() - start_time
  print("Elapsed time for Detected_Numbers",":", elapsed_time, "seconds")
  
  Numbers = Sortlist(Detected_Numbers)
  print(Numbers)
  
ScaleDataframe = makeDfwithfactors(Name_of_Images,scale_mode,Numbers,Lengths) ### Calculate the factor mm per px
ScaleDataframe.to_csv(Analysis_data + "/Scale.csv", index = False) ### Save as df

### 9) AnnotationRead.py: Read the annotations and turn into pd.dataframe # Performs postprocessing
JsonToMeasurement(Analysis_data + "/annotations.json") ### Saves as Annotations[:-5] ".csv"

## 7) Optional: Crop Images and Return df with name and crops
## This code needs RGB images as input 

if Crop == True:
  Cropped_Images = CropImages(Paths_of_Images, Analysis_data + "/annotations.csv", organs, Analysis_data + "/crops",Name_of_Images)
  print(Cropped_Images)
  if Classify == True:
    
    Species = Classify_Species(Original_Images, Classifier_Loc)

### 10) DaphnidMeasure.py: Calculate the pixel values of distances
Measurements = DaphnidMeasurements(Analysis_data + "/annotations.csv") ### Saves as Annotations[:-5] ".csv"

#### 11) Now compute the body width if wished

if Body_width_eval == True:
  print(f"Starting body width evaluation, method = {Width_method}")
  mode = "" 
  if detection_vis == True:
    mode += "-v "

  if refine == True:
    mode += "-r"
  
  try:
    subprocess.run(f'/bin/bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate ginjinn && ginjinn predict ~/Instance_segmentation/Segment_both_split/daphnid_instances_0.1 -i {Original_Images} -o {Analysis_data + "/Segmentation"} {mode} "', check=True, shell=True)
  
  except:
    print("WARNING: Skipping body width estimation")

Measurements = Perform_Chosen_method(Width_method, Analysis_data, Original_Images +"/")## A error here but prgram ignores it


### Now we want to merge all data here
### 12) DataframeMerge.py: Merge the data into one datframe and calculate µM values
### and Visualize the Image into a Folder with the width and length
### Visualization has two sources it draws from: The Values_to_be_Drawn dataframe in annotation order
### And the CompleteData.csv which does not have to have the same order

DataFrame = FuseDataframes(Measurements, ScaleDataframe, Analysis_data)

### The Species to the data
if Classify == True:
  ### Now add species and save it
  import pandas as pd
  
  Data = pd.read_csv(DataFrame)
  ## Now at the predictions to it
  Data["species"] = species
  
  Data.to_csv(DataFrame)

if visualize == True:
  ShowMeasureImage(Paths_of_Images, DataFrame,Analysis_data)

