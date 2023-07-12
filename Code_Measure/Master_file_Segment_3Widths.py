### Masterfile from jpg to dataframe with Eigenvector turn and width
### TO DO YAML data and all options with standard and advanced settings
### Test runs, Guarantee the import of functions work
### Clean files as well as useless code and ML Code
### Write MA

import os
import shutil
import subprocess
import pandas as pd
from TifzuJPEG import ConvertTiftoJPEG
from Yaml_load  import ConfigImport
from Annotations_Post_process import JsonToMeasurement
from DaphnidMeasure import DaphnidMeasurements
from Scale_detect_EasyOCR import Images_list, getLineLength, group_lines, RoughCrop, CropImage,Sortlist,makeDfwithfactors
from DataframeMerge import FuseDataframes, ShowMeasureImage
from BodyWidthChooser import Perform_Chosen_method
from Image_cropper import CropImages

### 1) Yaml_load_test.py: Read all the settings in our yaml and save them in variables
settings = ConfigImport("/home/philipp/DaphniaDetector/Code_Measure/settings_segment.yml")

### Paths
Original_Images = settings["Original_images"] 
Analysis_data = settings["Analysis_data"]

### Image Converter
Convert_to_JPEG = settings["Convert_to_JPEG"]

## Ginjinn settings
visualize = settings["visualize"]  
Crop = settings["crop"]
refine = settings["refine"]

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
os.makedirs(Analysis_data, exist_ok=True)
### 2) TifzuJPEG.py: Change all Images to JPEG, name misleading changes not only tifs 
if Convert_to_JPEG == True:
  ConvertTiftoJPEG(Original_Images, Original_Images)
  Original_Images = Original_Images + "/JPG"

## Find path to original images and names of the images
Paths_of_Images, Name_of_Images = Images_list(Original_Images)
print(Paths_of_Images)

### Now we have a dataset of the size 11xn on which we perform the full detection
### 6) FullDetect: Detect Eye, Body, Daphnid, Spina base, Spina tip, heart, head and brood chamber
mode = "" 
if detection_vis == True:
  mode += "-v "

predict_command = f'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ginjinn && ginjinn predict ~/Master_thesis_data/Model_Data_MA/Loose_box_final_2204 -i {Original_Images} -o {Analysis_data} {mode}'
subprocess.run(predict_command, shell=True, executable='/bin/bash')

### Now the user needs to check the bounding boxes of his unique Images
### and correct errors. After that we can start the evaluation

response = input(f"Do you want to check your labels? This is highly adivised. Extract your data from {Analysis_data} [y(es)/n(o)/c(ancel)]")

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
  Lines, CroppedImages, LineCoordinates = getLineLength(Paths_of_Images) ### Line lengths and lower right aprt of image
  Units = group_lines(CroppedImages, Lines, LineCoordinates, psm_mode) ## recognizes with tesseract ocr the Scale values
  Rough_Images = RoughCrop(Units) #### Makes one number or list out of list(list(n,n1,n2), list(n,n1,n2),...)
  Small_Images = CropImage(Rough_Images)
  Detected_Numbers = detect_Number(Small_Images)
  Numbers = Sortlist(Detected_list)

ScaleDataframe = makeDfwithfactors(Name_of_Images,scale_mode,CleanUnits, Lines,) ### Calculate the factor mm per px
ScaleDataframe.to_csv(Analysis_data + "/Scale.csv", index = False) ### Save as df

### 9) AnnotationRead.py: Read the annotations and turn into pd.dataframe # Performs postprocessing
JsonToMeasurement(Analysis_data + "/annotations.json") ### Saves as Annotations[:-5] ".csv"

## 7) Optional: Crop Images and Return df with name and crops

if Crop == True:
  Cropped_Images = CropImages(Paths_of_Images, Analysis_data + "/annotations.csv", organs, Analysis_data + "/crops",Name_of_Images)
  
  ### 8) Run the species Classifier and create a list of the predicted labels
  
  Body_crops = Cropped_Images['Whole_daphnid'].values
  Classifier = tf.keras.models.load_model(Classifier_loc)
  predictions = Classifier.predict(Original_Images)
  label_mapping = model.class_labels
  labels = [label_mapping[np.argmax(predictions)]]

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
  
  subprocess.run(f'/bin/bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate ginjinn && ginjinn predict ~/Instance_segmentation/Segment_both_split/daphnid_instances_0.1 -i {Original_Images} -o {Analysis_data + "/Segmentation"} {mode} "', check=True, shell=True)
  

Measurements = Perform_Chosen_method(Width_method, "/home/philipp/Data0407/", Original_Images +"/")

### 12) DataframeMerge.py: Merge the data into one datframe and calculate µM values
### and Visualize the Image into a Folder with the width and length
### Visualization has two sources it draws from: The Values_to_be_Drawn dataframe in annotation order
### And the CompleteData.csv which does not have to have the same order

DataFrame = FuseDataframes(Measurements, ScaleDataframe, Analysis_data)

## Now at the predictions to it
#DataFrame["species"] = labels

if visualize == True:
  ShowMeasureImage(Paths_of_Images, DataFrame,Analysis_data)

