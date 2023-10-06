### GinJinn2 cropping has some problems: We cannot choose what to crop
### We do not now which crop is which image
### Thats we want to crop images based on the dataframe

def CropImages(Original_Images, Dataframe, Crop_mode, Save_folder, Image_names):
  # Input: path to Original_Images as RGB or Greyscale
  # Dataframe: Df with Bounding box coordinates 
  # Crop_mode: A list of features to be cropped
  # List can contain Body, Whole Daphnid, Eye, Brood chamber,
  # Spina tip, Spina base, Heart and Head
  # Save_folder
  # Names of the images
  
  # Returns list of cropped images and saves them in folder
  import os
  import cv2 as cv2
  import pandas as pd
  import numpy as np
  
  
  Coordinate_coloumns = ["Xmin_", "Ymin_", "bboxWidth_", "bboxHeight_"]
  Data = pd.read_csv(Dataframe)
  ### The dataframe might not be sorted how we want it
  sorted_df = Data.sort_values(by='image_id', key=lambda x: x.map(dict(zip(Image_names, range(len(Image_names))))))

  for x in Crop_mode:
    temp_crops = []
    os.makedirs(Save_folder + "/" + x, exist_ok=True)
    New_Save_folder = Save_folder + "/" + x
    ### Original images is sorted differently than the Dataframe 
    for y in range(len(Original_Images)):
      try:
        ### We yield the sorting of Image_names
        Row = sorted_df.iloc[y]
        Xmin = Row[Coordinate_coloumns[0] + x]
        Ymin = Row[Coordinate_coloumns[1] + x]
        Xmax = Row[Coordinate_coloumns[2] + x] + Row[Coordinate_coloumns[0] + x]
        Ymax = Row[Coordinate_coloumns[3] + x] + Row[Coordinate_coloumns[1] + x]
        # Now Crop the image
        img = cv2.imread(Original_Images[y])
        
        crop = img[int(Ymin):int(Ymax),int(Xmin):int(Xmax)]
        
        #### Save the crops for control 
        
        base_name = os.path.basename(os.path.normpath(Image_names[y]))
        filename = New_Save_folder + "/" + base_name + "_" + x +".jpg"
        
        cv2.imwrite(filename, crop)
        temp_crops.append(crop)
      
      except:
        print(f"No {x} box detected for {Image_names[y]}")
        temp_crops.append(0)
  return temp_crops


