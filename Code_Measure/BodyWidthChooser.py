## This script compactly chooses which evaluation is performed for body width

def Perform_Chosen_method(Method, AnalysisFolder, ImagesFolder):
  ### Check input and choose adeqaute Method
  ### Input: str(Method), Sperfeld, Rabus or Imhof
  ### ImagesFolder with the location the original inages
  ### AnaylsisFolder: Folder with the annotations 
  
  #### All methods have the same first step: Get Masks 
  #### Rabus and Imhof rotate the image before measuring due to orientaton to the 
  #### body axis, while Sperfeld does not need it and measures directly
  from Instance_segment_PCA_Imhof import (getOrientation,drawAxis,point_trans, Create_Mask, 
  Image_Rotation,Detect_Midpoint,Measure_Width_Imhof, AddToData, Create_Visualization_Data)

  from Instance_segment_PCA_RabusLaforsch import (Measure_Width_Rabus)

  from Instance_segment_PCA_Sperfeld import (PerpendicularLine_Eye_Sb, Measure_Width_Sperfeld)
  
  import pandas as pd
  
  Image_sort, Mask = Create_Mask(AnalysisFolder+ "Segmentation/annotations.json", ImagesFolder)
  if Method == "Sperfeld":
    print("Proceeding with method Sperfeld")
    
    Eye_Spina_df  = pd.read_csv(AnalysisFolder + "annotations.csv", decimal = ".")
    Midpoints, Rotated_masks, Rotation_angles = PerpendicularLine_Eye_Sb(Eye_Spina_df,Image_sort,Mask)
    Body_width, X_Start, X_End = Measure_Width_Sperfeld(Rotated_masks,Midpoints)
    
  elif Method == "Rabus":
    print("Proceeding with method Rabus")
    
    Rotation_angles, Rotated_masks = Image_Rotation(Mask, Image_sort) 
    Eye_Spina_df  = pd.read_csv(AnalysisFolder + "annotations.csv",decimal = ".")
    Body_width, X_Start, X_End,Midpoints = Measure_Width_Rabus(Rotated_masks)

  elif Method == "Imhof":
    print("Proceeding with method Imhof")
    
    Rotation_angles, Rotated_masks = Image_Rotation(Mask, Image_sort) 
    Eye_Spina_df  = pd.read_csv(AnalysisFolder + "annotations.csv",decimal = ".")
    Midpoints = Detect_Midpoint(Eye_Spina_df,Rotation_angles,Image_sort,Rotated_masks,Mask)
    Body_width, X_Start, X_End = Measure_Width_Imhof(Rotated_masks,Midpoints)

  else: ### We want to execute Imhof as standard and not abort the process after the time intensive
    ## steps are already done
    print("No method chosen proceeding with method Imhof")
  
    Rotation_angles, Rotated_masks = Image_Rotation(Mask, Image_sort) 
    Eye_Spina_df  = pd.read_csv(AnalysisFolder + "annotations.csv",decimal = ".")
    Midpoints = Detect_Midpoint(Eye_Spina_df,Rotation_angles,Image_sort,Rotated_masks,Mask)
    Body_width, X_Start, X_End = Measure_Width_Imhof(Rotated_masks,Midpoints)

  Values_To_Be_Drawn = Create_Visualization_Data(Image_sort,X_Start, X_End, Midpoints, Rotation_angles, Rotated_masks, Mask)
  Full_Measures_px = AddToData(Body_width,Values_To_Be_Drawn,Image_sort,AnalysisFolder + "/annotations.csv")
  
  return Full_Measures_px




