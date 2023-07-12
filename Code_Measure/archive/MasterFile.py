### Masterfile from jpg to dataframe
### TO DO YAML data and all options with standard and advanced settings
### Test runs, Guarantee the import of functions work
### Clean files as well as useless code and ML Code
### Write MA

import os
import shutil
import subprocess
os.chdir('/home/philipp/Grouped_Code/CopyOfWorkflow_code/')

from TifzuJPEG import ConvertTiftoJPEG
from Yaml_load_test import ConfigImport
from AnnotationRead import JsonToMeasurement
from DaphnidMeasure import DaphnidMeasurements
from DpiToMm_experimental import Images_list, getLineLength, get_Scale, NormalizeScale, makeDfwithfactors
from DataframeMerge import FuseDataframes, ShowMeasureImage
from GetUprightImages import JsonToMeasurement_NoSave, CleanDataPD, CleanJSON, CleanJPEGS
from TurnForGinJinn2 import point_trans, body_width_to_csv, Rotate_by_eye, CreateDuplicates, EyeCenter, JsonToMeasurement_turn


### 1) Yaml_load_test.py: Read all the settings in our yaml and save them in variables
settings = ConfigImport("/home/philipp/Grouped_Code/CopyOfWorkflow_code/settings.yml")

Original_Images = settings["Original_images"] 
Eye_Annotations = settings["Eye_Annotations"]
Duplicate_Folder = settings["Duplicate_Folder"]
Analysis_data = settings["Analysis_data"]

Convert_to_JPEG = settings["Convert_to_JPEG"]
visualize = bool(settings["visualize"])   
crop = settings["crop"]
detection_vis = bool(settings["detection_vis"])
Standard_value_px_per_µm = settings["Standard_value_px_per_µm"]
scale_mode = int(settings["Scale_detector_mode"])
psm_mode = int(settings["psm_mode"])

#### Create the final data saveloc: Here we will find the final images and csv's
#### If the folder exists we delete and overwerite it


if os.path.exists(Analysis_data):
    shutil.rmtree(Analysis_data)
os.makedirs(Analysis_data)

### 2) TifzuJPEG.py: Change all Images to JPEG, name misleading changes not only tifs 
if Convert_to_JPEG == True:
  ConvertTiftoJPEG(Original_Images, Original_Images)
  Original_Images = Original_Images + "/JPG"
### 3) EyeDetectorDC: Detect Eyes using the GinJinnModel

import subprocess
import os

### Runs the first detection model which only detects eyes ## do we want to enable -c -v?
subprocess.run(f'/bin/bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate ginjinn && ginjinn predict ~/Model_Data/EyeDetectorFPN -i {Original_Images} -o {Eye_Annotations}"', check=True, shell=True)

# Now we are able to perform two steps 1) get the scale of the images, which is possible 
# even without any annotations and 2) Turn the Daphnids ~upright using the eye coordinate

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

### 5) TurnForGinJinn.py: Rotate the images up on the eye coordinate and create 10 duplicates turned by 5°

import pandas as pd

Annotationfile = JsonToMeasurement_turn(Eye_Annotations +"/annotations.json") ### Is it saved as this or in annotations/annotations.json?
AnnotationsWithEyeCenter = EyeCenter(Annotationfile)
RotationAngleByEye, RotationImageEye, RadiantEye = Rotate_by_eye(AnnotationsWithEyeCenter,Paths_of_Images,Name_of_Images,False)
CreateDuplicates(RotationImageEye, Paths_of_Images, Name_of_Images, Duplicate_Folder)

### Now we have a dataset of the size 11xn on which we perform the full detection
### 6) FullDetect: Detect Eye, Body, Daphnid, Spina base, Spina tip, heart, head and brood chamber
mode = "" 
if detection_vis == True:
  mode += "-v "

if crop == True:
  mode += "-c"

subprocess.run(f'/bin/bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate ginjinn && ginjinn predict ~/Model_Data/Loose_box_final_2204 -i {Duplicate_Folder} -o {Analysis_data + "/Full_detect"} {mode}"', check=True, shell=True)

### 7) GetUprightImages.py: Reduce duplicates from 11 to 1 again 
### To get actual upright daphnids we reduce the dataset down again, by the smallest body bounding box

### Read data 
Image_Paths_dup, Image_Names_dup = Images_list(Duplicate_Folder)
data = JsonToMeasurement_NoSave(Analysis_data + "/Full_detect/annotations.json")
### Clean it by smallest box
List_of_Straight_Image_names = CleanDataPD(data)

CleanJSON(Analysis_data + "/Full_detect/annotations.json", List_of_Straight_Image_names, Image_Paths_dup, Analysis_data) # Save values as Unique_Values.json
CleanJPEGS(Image_Paths_dup, List_of_Straight_Image_names, Analysis_data + "/images")

### Now the user needs to check the bounding boxes of his unique Images
### and correct errors. After that we can start the evaluation
### How to implement this pause? A check box?
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


### 8) AnnotationRead.py: Read the annotations and turn into pd.dataframe
JsonToMeasurement(Analysis_data + "/Unique_Values.json") ### Saves as Annotations[:-5] ".csv"

### 9) DaphnidMeasure.py: Calculate the pixel values of distances
Measurments = DaphnidMeasurements(Analysis_data + "/Unique_Values.csv") ### Saves as Annotations[:-5] ".csv"


### 10) DataframeMerge.py: Merge the data into one datframe and calculate µM values
### and Visualize the Image into a Folder with the width and length

DataFrame = FuseDataframes(Measurments, ScaleDataframe, Analysis_data)

Image_Paths_final, Image_Names_final = Images_list(Analysis_data +"/images")
if visualize == True:
  ShowMeasureImage(Image_Paths_final, DataFrame)

