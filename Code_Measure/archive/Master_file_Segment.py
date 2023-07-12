### Masterfile from jpg to dataframe with Eigenvector turn and width
### TO DO YAML data and all options with standard and advanced settings
### Test runs, Guarantee the import of functions work
### Clean files as well as useless code and ML Code
### Write MA

import os
import shutil
import subprocess
import pandas as pd
os.chdir('/home/philipp/Code_Measurements/CopyOfWorkflow_code/')

from TifzuJPEG import ConvertTiftoJPEG
from Yaml_load  import ConfigImport
from Annotations_Post_process import JsonToMeasurement
from DaphnidMeasure import DaphnidMeasurements
from DpiToMm_experimental import Images_list, getLineLength, get_Scale, NormalizeScale, makeDfwithfactors
from DataframeMerge import FuseDataframes, ShowMeasureImage
from Instance_segment_PCA_RabusLaforsch import (getOrientation,drawAxis,point_trans, Create_Mask, 
Image_Rotation,Detect_Midpoint,Measure_Width, AddToData, Create_Visualization_Data)
from BodyWidthChooser import Perform_Chosen_method
### 1) Yaml_load_test.py: Read all the settings in our yaml and save them in variables
settings = ConfigImport("/home/philipp/Code_Measurements/CopyOfWorkflow_code/settings_segment.yml")

### Paths
Original_Images = settings["Original_images"] 
Analysis_data = settings["Analysis_data"]

### Image Converter
Convert_to_JPEG = settings["Convert_to_JPEG"]

## Ginjinn settings
visualize = settings["visualize"]  
crop = settings["crop"]
refine = settings["refine"]

# Measurement visualization
detection_vis = settings["detection_vis"]
Body_width_eval = settings["Body_width_eval"]
## Scale settings
Standard_value_px_per_µm = settings["Standard_value_px_per_µm"]
scale_mode = int(settings["Scale_detector_mode"])
psm_mode = int(settings["psm_mode"])

#Body width method

Width_method = settings["Width_method"]
#### Create the final data saveloc: Here we will find the final images and csv's
#### If the folder exists we delete and overwerite it


if os.path.exists(Analysis_data):
    shutil.rmtree(Analysis_data)
os.makedirs(Analysis_data)

### 2) TifzuJPEG.py: Change all Images to JPEG, name misleading changes not only tifs 
if Convert_to_JPEG == True:
  ConvertTiftoJPEG(Original_Images, Original_Images)
  Original_Images = Original_Images + "/JPG"

### 4) DpiToMm.py: Detect scale and calculate conversion factor based on the unrotated Images

Paths_of_Images, Name_of_Images = Images_list(Original_Images)
CleanUnits = []
Lines = []
if int(scale_mode) != 0:
  Lines, CroppedImages, LineCoordinates = getLineLength(Paths_of_Images) ### Line lengths and lower right aprt of image
  Units = get_Scale(CroppedImages, Lines, LineCoordinates, psm_mode) ## recognizes with tesseract ocr the Scale values
  CleanUnits = NormalizeScale(Units) #### Makes one number or list out of list(list(n,n1,n2), list(n,n1,n2),...)

ScaleDataframe = makeDfwithfactors(Name_of_Images,scale_mode,CleanUnits, Lines,) ### Calculate the factor mm per px
ScaleDataframe.to_csv(Analysis_data + "/Scale.csv", index = False) ### Save as df

### Now we have a dataset of the size 11xn on which we perform the full detection
### 6) FullDetect: Detect Eye, Body, Daphnid, Spina base, Spina tip, heart, head and brood chamber
mode = "" 
if detection_vis == True:
  mode += "-v "

if crop == True:
  mode += "-c"

subprocess.run(f'/bin/bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate ginjinn && ginjinn predict ~/Master_thesis_data/Model_Data_MA/Loose_box_final_2204 -i {Original_Images} -o {Analysis_data} {mode}"', check=True, shell=True)

### Now the user needs to check the bounding boxes of his unique Images
### and correct errors. After that we can start the evaluation

response = input(f"Do you want to check your labels? Extract your data from {Analysis_data} (y/n/c)")

if response == 'c':
    # Cancel the code
    exit()
elif response == 'y':
    # Pause the code
    input("Press enter to continue...")
else:
    # Continue the code
    pass



### 8 Run Classifier and create a list of the predicted labels

Classifier = tf.keras.models.load_model(f'{Classifier_loc}')

predictions = Classifier.predict("{Original_Images}")
label_mapping = model.class_labels
labels = [label_mapping[np.argmax(predictions)]]

### 9 AnnotationRead.py: Read the annotations and turn into pd.dataframe
JsonToMeasurement(Analysis_data + "/annotations.json") ### Saves as Annotations[:-5] ".csv"

### 10 DaphnidMeasure.py: Calculate the pixel values of distances
Measurments = DaphnidMeasurements(Analysis_data + "/annotations.csv") ### Saves as Annotations[:-5] ".csv"


#### 11 Now compute the body width if wished

if Body_width_eval == True:
  print(f"Starting body width evaluation, method = {Width_method}")
  mode = "" 
  if detection_vis == True:
    mode += "-v "

  if refine == True:
    mode += "-r"
  
  subprocess.run(f'/bin/bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate ginjinn && ginjinn predict ~/Instance_segmentation/Segment_both_split/daphnid_instances_0.1 -i {Original_Images} -o {Analysis_data + "/Segmentation"} {mode} "', check=True, shell=True)
  
  Perform_Chosen_method(Width_method, Analysis_data + "/Segmentation/annotations.json", Original_Images)
  Image_sort, Mask = Create_Mask(Analysis_data + "/Segmentation/annotations.json", Original_Images)
  Rotation_angles, Rotated_masks = Image_Rotation(Mask, Image_sort) 
  Eye_Spina_df  = pd.read_csv(Analysis_data + "/annotations.csv", dec = ".")

  ### Calculate the middle of the eye spin base axis as most Evaluation in papers do
  ### Right now we have the approach by Rabus & Laforsch
  ### To keep the rotation correct we have to make sure to keep the images sorted how they are in the annotations

  Midpoints = Detect_Midpoint(Eye_Spina_df,Rotation_angles,Image_sort,Rotated_masks,Mask)

  Body_width, X_Start, X_End = Measure_Width(Rotated_masks,Midpoints)
  

  Values_To_Be_Drawn = Create_Visualization_Data(Image_sort,Rotation_angles, Rotated_masks, Mask, X_Start, X_End, Midpoints)

  AddToData(Body_width,Values_To_Be_Drawn,Image_sort,Analysis_data + "/annotations.csv")
### 12) DataframeMerge.py: Merge the data into one datframe and calculate µM values
### and Visualize the Image into a Folder with the width and length
### Visualization has two sources it draws from: The Values_to_be_Drawn dataframe in annotation order
### And the CompleteData.csv which does not have to have the same order

DataFrame = FuseDataframes(Measurments, ScaleDataframe, Analysis_data,Body_width_eval)

## Now at the predictions to it
DataFrame["species"]
Image_Paths_final, Image_Names_final = Images_list(Analysis_data +"/images",Body_width_eval)
if visualize == True:
  ShowMeasureImage(Image_Paths_final, DataFrame)

