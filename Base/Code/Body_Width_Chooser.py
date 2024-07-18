## This script compactly chooses which evaluation is performed for body width

def Perform_Chosen_method(Method, Segment_annotations, Object_detect_annotations, ImagesFolder):
  ### Check input and choose adeqaute Method
  ### Input: str(Method), Sperfeld, Rabus or Imhof
  ### ImagesFolder with the location the original images
  ### Segment_annotations: Annotations.json resulting from instance segment
  ### Object_detect_annotations: Annotations.csv resulting from object detection
  
  #### All methods have the same first step: Get Masks 
  #### Rabus and Imhof rotate the image before measuring due to orientaton to the 
  #### body axis, while Sperfeld does not need it and measures directly
  #### If no mode is provided we assume Imhof
  
  from Instance_Segment_PCA_Imhof import (Create_Mask, 
  Image_Rotation,Detect_Midpoint,Measure_Width_Imhof, AddToData, Create_Visualization_Data)

  from Instance_Segment_PCA_RabusLaforsch import (Measure_Width_Rabus)

  from Instance_Segment_PCA_Sperfeld import (PerpendicularLine_Eye_Sb, Measure_Width_Sperfeld)
  
  import pandas as pd
  
  Image_sort, Mask = Create_Mask(Segment_annotations, ImagesFolder)
  if Method == "Sperfeld":
    print("Proceeding with method Sperfeld")
    
    Eye_Spina_df  = pd.read_csv(Object_detect_annotations, decimal = ".")
    Midpoints, Rotated_masks, Rotation_angles = PerpendicularLine_Eye_Sb(Eye_Spina_df,Image_sort,Mask)
    Body_width, X_Start, X_End = Measure_Width_Sperfeld(Rotated_masks,Midpoints)
    
  elif Method == "Rabus":
    print("Proceeding with method Rabus")
    
    Rotation_angles, Rotated_masks = Image_Rotation(Mask, Image_sort) 
    Eye_Spina_df  = pd.read_csv(Object_detect_annotations,decimal = ".")
    Body_width, X_Start, X_End,Midpoints = Measure_Width_Rabus(Rotated_masks)

  elif Method == "Imhof":
    print("Proceeding with method Imhof")
    Rotation_angles, Rotated_masks = Image_Rotation(Mask, Image_sort) 
    Eye_Spina_df  = pd.read_csv(Object_detect_annotations,decimal = ".")
    Midpoints = Detect_Midpoint(Eye_Spina_df,Rotation_angles,Image_sort,Rotated_masks,Mask)
    Body_width, X_Start, X_End = Measure_Width_Imhof(Rotated_masks,Midpoints)

  else: ### We want to execute Imhof as standard and not abort the process after the time intensive
    ## steps are already done
    print("No method chosen proceeding with method Imhof")
    
    Rotation_angles, Rotated_masks = Image_Rotation(Mask, Image_sort) 
    Eye_Spina_df  = pd.read_csv(Object_detect_annotations,decimal = ".")
    Midpoints = Detect_Midpoint(Eye_Spina_df,Rotation_angles,Image_sort,Rotated_masks,Mask)
    Body_width, X_Start, X_End = Measure_Width_Imhof(Rotated_masks,Midpoints)
  
  ## Int error in Values_to_be_drawn does not affect functionality
  
  Values_To_Be_Drawn = Create_Visualization_Data(Image_sort,X_Start, X_End, Midpoints, Rotation_angles, Rotated_masks, Mask)
  Full_Measures_px = AddToData(Body_width,Values_To_Be_Drawn,Image_sort,Object_detect_annotations)

  return Full_Measures_px





