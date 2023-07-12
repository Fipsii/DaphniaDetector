### This Code is planned to merge the two existing Dataframes (Measurments and Scale)
### and calculate mm values ### Important to note: We should set paths from configs


### Test run with simonas data ### Make new Datamerge.py later

def FuseDataframes(dfPixel,dfScales, savelocation):
  import pandas as pd
  
  ### Input Dataframe with mesaurments in px, df with scales and saveloc
  ### First create a Name column for dfPixel. As we have the image:id with degree turned
  
  import re
  
  ### This is only for turned images from the old code important

  dfPixel.rename(columns={"image_id": "Name"}, inplace=True)
  #dfPixel["Name"] = List_of_Org_names

  DataFinished = pd.DataFrame()
  DataFinished = pd.merge(dfPixel,dfScales, on = "Name", how = "inner")
  #print(dfPixel["Name"])
  DataFinished["Spinalength[mm]"] = DataFinished["Spinalength[px]"] * DataFinished["distance_per_pixel"]
  DataFinished["Bodylength[mm]"] = DataFinished["Bodylength[px]"] * DataFinished["distance_per_pixel"]
  

  DataFinished["Bodywidth[mm]"] = DataFinished["Width[px]"] * DataFinished["distance_per_pixel"]
  
  savename = savelocation + "/Completed_data.csv"
  
  DataFinished.to_csv(savename, index = False)
  print(f"Data saved under {savename}")
  return savename

### This Works now we have to sort out the doubles and false positives (mostly 0s that are easy to discard) 
### We also have to discuss the few false line values in the second funtion

def ShowMeasureImage(Imagepath, data_eval, AnalysisFolder): ### Input the dataframe and 1 Image
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os as os
    import cv2 as cv2
    ## Input paths to original iamges
    ## Dataframe with data of measurements
    ## The output Folder
    
    #### Read in one Image
    
    data = pd.read_csv(data_eval)
    save_folder = AnalysisFolder + "/Measurement_Visualization"
    os.mkdir(save_folder)
    
    
    if type(Imagepath) != list: ## Make a list if only one Image is submitted
      Imagepath = [Imagepath]

    for x in range(len(Imagepath)):
      im = cv2.imread(Imagepath[x])
      
      plt.clf()
      ### split the path:
      filename = Imagepath[x].split("/")[-1]
      Image_index = data.index[data['Name'] == filename]
      plt.imshow(im)
      #print(filename, Image_index)
      
      Image_index = Image_index[0] ## Make Int64IndexSeries of one number into a number
      
      plt.imshow(im)
      ### Body with still needs to implemented
      ### Plot the lines
      plt.plot([data["Center_X_Sb"][Image_index],data["Center_X_Eye"][Image_index]],[data["Center_Y_Sb"][Image_index],data["Center_Y_Eye"][Image_index]], color = "red", linewidth=0.5)
      plt.plot([data["Center_X_Sb"][Image_index],data["Center_X_St"][Image_index]],[data["Center_Y_Sb"][Image_index],data["Center_Y_St"][Image_index]], color = "red", linewidth=0.5) 
      ### Plot the the length
      plt.plot(data["Center_X_Eye"][Image_index],data["Center_Y_Eye"][Image_index],marker="o", markersize=2, markeredgecolor="black", markerfacecolor="white")
      plt.plot(data["Center_X_Sb"][Image_index],data["Center_Y_Sb"][Image_index],marker="o", markersize=2, markeredgecolor="black", markerfacecolor="white") 
      plt.plot(data["Center_X_St"][Image_index],data["Center_Y_St"][Image_index],marker="o", markersize=2, markeredgecolor="black", markerfacecolor="white") 
      
 
      # plot the width
      # The width needs to be transposed back from our turned image into the real data Easiest way would be while turning
      plt.plot([data["Width_X1"][Image_index],data["Width_X2"][Image_index]],[data["Width_Y1"][Image_index],
      data["Width_Y2"][Image_index]], color = "blue", linewidth=0.5, linestyle='dotted')
      plt.axis('off')
  
      plt.savefig(save_folder +"/visualization_of_" + filename,dpi=1000,bbox_inches= "tight",pad_inches=0)
      print("Visualization of " + filename + " printed")
  
### Which visualization options? True = Image or Image list and False = No #dpi=1000
