### Masterfile from jpg to dataframe with Eigenvector turn and width
### TO DO YAML data and all options with standard and advanced settings
### Test runs, Guarantee the import of functions work
### Clean files as well as useless code and ML Code

from math import e
import time
import os
import subprocess
import pandas as pd
from Convert_to_JPG import ConvertToJPEG
from Yaml_Load  import ConfigImport, update_yaml_setting
from Annotations_Post_Process import JsonToMeasurement, AddSegment
from Daphnid_Measure import DaphnidMeasurements
from Scale_Detect import Images_list, getLineLength, RoughCrop, CropImage,Sortlist,detect_Number, makeDfwithfactors
from Dataframe_Merge import FuseDataframes, ShowMeasureImage
from Body_Width_Chooser import Perform_Chosen_method
from Image_Cropper import CropImages
from Classification_Deploy import Classify_Species
import warnings
import sys

try:

  ## Set the response (needed for strg + c abort)
  response = 0

  ## Track time
  start_time = time.time()

  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
  # Get the current file's directory
  current_directory = os.path.dirname(os.path.abspath(__file__))

  ### Step 1 Read all the settings in our yaml and save them in variables
  settings = ConfigImport(current_directory + "/settings_segment.yml")

  ### Paths 
  Original_Images = settings["Original_images"] 
  Analysis_data = settings["Analysis_data"]
  Classifier_loc = settings["Classifier_loc"]
  Cache_Org_Img = Original_Images

  # Prints the nicely formatted dictionary

  if settings["status_flag"] == 0:

    ### Create the final data saveloc: Here we will find the final images and csv's
    ### If the folder exists we delete and overwerite it
    ### Change all Images to JPEG

    if settings["Convert_to_JPEG"] == True:
      JPG_folder = ConvertToJPEG(Original_Images, Analysis_data)

      ## Update setting detect the newly created /JPG folder
      ## First save the path to reset the setting at the end

      update_yaml_setting(current_directory + "/settings_segment.yml", "Original_images", JPG_folder)
      Original_Images = settings["Original_images"] 
      
    ### Read in all image paths and names

    Paths_of_Images, Name_of_Images = Images_list(Original_Images)
    print(f"{len(Paths_of_Images)} images detected for analysis.")
    ### Object detection of the Eye, Body, Daphnid, Spina base, Spina tip, heart, head and brood chamber
    ### mode -v for visualization options

    mode = "" 
    
    if settings["detection_vis"] == True:
      mode += "-v "

    obj_detect_start = time.time()
    
    ### Catch GinJinnWarnings to provide cleaner output and perform object detection 
    IgnoredWarnings = "ignore:This overload of nonzero is deprecated"
    predict_command = f'PYTHONWARNINGS="{IgnoredWarnings}" conda run --no-capture-output -n ginjinnCPU ginjinn predict /app/Models/Loose_box_final_2204 -i {Original_Images} -o {Analysis_data} {mode}'
    
    subprocess.run(predict_command, check=True, shell=True)

    obj_detect_end = time.time()

    ### Add ",segmentation: []" to the annotations to amake them CVAT compatible

    AddSegment(Analysis_data + "/annotations.json")

    ###  Read the annotations and turn into pd.dataframe and
    ###  extend the body bounding box to limits of the spina tip
    ###  Makes JSON compatible with CVAT
    ###  This data is saved in the JSON
    ###  This causes a discrepancy between the visualzed images by
    ###  ginjinn and the images the users gets to correct
    
    JsonToMeasurement(Analysis_data + "/annotations.json") 
    
    ### Now the user needs to check the bounding boxes of his unique Images
    ### and correct errors. After that we can start the evaluation again
    ### c(ancel): Cancels the code, Restarting the code will start from the beginning
    ### y(es):    Exit the code and resume at scale detection at restart
    ### n(o):     Continue the code instantly

    response = input(f"Do you want to check your labels? This is highly advised. Extract your data from {Analysis_data} [y(es)/n(o)/c(ancel)]: ")

    if response == 'c':
        # Cancel the code
        print("Aborting the code")
        exit()
    elif response == 'y':
        # Pause the code
        update_yaml_setting(current_directory + "/settings_segment.yml", "status_flag", 1)
        print("Quit program to check bounding boxes, changing status to 1")
        print(f"Extract data out of {Analysis_data}/annotations.json")
        exit()
    else:
        # Continue the code
        print("Continuing code without checking annotations")
        pass
    

  ###  We start from this point if user chose to correct annotations
  if settings["status_flag"] == 1:
    print("Continuing evaluation with checked annotations")
    Paths_of_Images, Name_of_Images = Images_list(Original_Images)

  ### Detect the scale in the image or us the provided calculation factor to build a 
  ### Dataframe with Image name, coordinates, lenghts and scale values

  scale_detect_start = time.time()

  if int(settings["Scale_detector_mode"]) != 0:
    
    Lengths, Line_Coors, List_of_images = getLineLength(Paths_of_Images)  ### Line lengths and lower right apart of image
    Rough_Images = RoughCrop(Line_Coors, List_of_images)                  ### Makes one number or list out of list(list(n,n1,n2), list(n,n1,n2),...)
    Small_Images = CropImage(Rough_Images)
    Detected_Numbers = detect_Number(Small_Images)
    Numbers = Sortlist(Detected_Numbers)

  ScaleDataframe = makeDfwithfactors(Name_of_Images,settings["Scale_detector_mode"],Numbers,Lengths,Line_Coors,settings["Conv_factor"])
  ScaleDataframe.to_csv(Analysis_data + "/scale.csv", index = False)      ### Save as df for safety? or just toss it?

  scale_detect_end = time.time()

  ### Crop Images and Return df with name and crops
  ### This code takes RGB images as input 
  ### Creates a folder containing folders 
  ### with the cropped body parts

  crop_classify_start = time.time()

  if settings["crop"] == True:
    Cropped_Images = CropImages(Paths_of_Images, Analysis_data + "/annotations.csv", settings["organs"], Analysis_data + "/crops",Name_of_Images)

    if settings["Classify"] == True:
      Species = Classify_Species(Original_Images, Classifier_loc)

  crop_classify_end = time.time()

  ### Measure Body length and Spina length
  ### Add the Data to the DataFrame

  Measurements = DaphnidMeasurements(Analysis_data + "/annotations.csv") 

  ### Start Body width evaluation with instance
  ### segmentation and refinement if wished
  ### Refinement is necessary but computational costly

  seg_det_ref_start = time.time()

  if settings['Body_width_eval'] == True:
    print(f"Starting body width evaluation, method = {settings['Width_method']}")
    mode = "" 

    if settings["detection_vis"] == True:
      mode += "-v "

    if settings["refine"] == True:
      mode += "-r"
    
    try:

      predict_command = f'PYTHONWARNINGS="{IgnoredWarnings}" conda run --no-capture-output -n ginjinnCPU ginjinn predict /app/Models/daphnid_instances_0.1 -i {Original_Images} -o {Analysis_data + "/segmentation"} {mode}'
      subprocess.run(predict_command, check=True, shell=True)

      
    except:
      print("WARNING: Skipping body width estimation")

    seg_det_ref_end = time.time()

    ### Perfrom the Body width calculation based
    ### on one of 3 methods (Imhof, Sperfeld, Rabus)

    Measurements = Perform_Chosen_method(settings["Width_method"], Analysis_data + "/segmentation/annotations.json", Analysis_data + "/annotations.csv", Original_Images +"/")


  ### Here we Merge the data into one Dataframe 
  ### If Classification was performed we add the
  ### coloumn species

  DataFrame = FuseDataframes(Measurements, ScaleDataframe, Analysis_data, settings['Body_width_eval'])

  ### Add the Species to the data
  if settings["Classify"] == True:
    ### Now add species and save it

    Data = pd.read_csv(DataFrame)
    ## Now add the predictions to it
    Data["species"] = Species
    
    Data.to_csv(DataFrame)

  ### Visualize all data in the image
  ### Shows length width and scale if detected
  ### If scale mode was 0 no visualization
  ### is performed. If scale mode 1 is selected
  ### we visualize every image as if mode 2 
  ### was selected

  vis_start = time.time()

  if settings["visualize"] == True:

    ShowMeasureImage(Paths_of_Images, DataFrame, Analysis_data, settings["Scale_detector_mode"])

  vis_end = time.time()

### Reset the state flag to 0 if code is finished
### And Reset the Original file to /app/images
### This should also catch Strg + C abort
### Important if Strg + C is occuring multiple times it aborts
### the finally block

finally:
  if response != "y":
    update_yaml_setting(current_directory + "/settings_segment.yml", "status_flag", 0)
    update_yaml_setting(current_directory + "/settings_segment.yml", "Original_images", Cache_Org_Img)
    
print("Finished detection. You may have to wait shortly for docker to transfer results onto your local machine.")

### Track the time
end_time = time.time()

time_taken_seconds =divmod(end_time - start_time,60)
vis_all = divmod(vis_end - vis_start,60)
seg_det_ref_all = divmod(seg_det_ref_end - seg_det_ref_start,60)
crop_classify_all = divmod(crop_classify_end - crop_classify_start,60)
scale_detect_all = divmod(scale_detect_end - scale_detect_start,60)
obj_detect_all =divmod(obj_detect_end - obj_detect_start,60 )

cpu_info = os.popen('lscpu').read()  # This will give CPU information

with open(Analysis_data + '/time_and_hardware_info.txt', 'w') as f:
      f.write(f"Overall time taken (min, secs): {time_taken_seconds}\n")
      f.write(f"Object detection (min, secs): {obj_detect_all}\n")
      f.write(f"Scale detection (min, secs): {scale_detect_all}\n")
      f.write(f"Classification & Crop (min, secs): {crop_classify_all}\n")
      f.write(f"Seg refinement (min, secs): {seg_det_ref_all}\n")
      f.write(f"Visualization (min, secs): {vis_all}\n")      
      f.write("\nCPU Information:\n")
      f.write(cpu_info)

