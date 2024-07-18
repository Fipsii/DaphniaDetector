### This Code is planned to merge the two existing Dataframes (Measurments and Scale)
### and calculate mm values ### Important to note: We should set paths from configs


### Test run with simonas data ### Make new Datamerge.py later

def FuseDataframes(dfPixel,dfScales, savelocation, bodywidth):
  import pandas as pd
  
  ### Input Dataframe with mesaurments in px, df with scales and saveloc
  ### First create a Name column for dfPixel. As we have the image:id with degree turned
    
  ### This is only for turned images from the old code important

  dfPixel.rename(columns={"image_id": "Name"}, inplace=True)
  #dfPixel["Name"] = List_of_Org_names

  DataFinished = pd.DataFrame()
  DataFinished = pd.merge(dfPixel,dfScales, on = "Name", how = "inner")

  DataFinished["Spinalength[mm]"] = DataFinished["Spinalength[px]"] * DataFinished["distance_per_pixel"]
  DataFinished["Bodylength[mm]"] = DataFinished["Bodylength[px]"] * DataFinished["distance_per_pixel"]
  
  ### Check uf we need bodywidth coloumn
  
  if bodywidth == True:
    DataFinished["Bodywidth[mm]"] = DataFinished["Width[px]"] * DataFinished["distance_per_pixel"]
  
  savename = savelocation + "/results.csv"
  
  DataFinished.to_csv(savename, index = False)
  print(f"Data saved under {savename}")
  return savename

### This Works now we have to sort out the doubles and false positives (mostly 0s that are easy to discard) 
### We also have to discuss the few false line values in the second funtion

def ShowMeasureImage(Imagepath, data_eval, AnalysisFolder, Scale_Mode): ### Input the dataframe and 1 Image
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os as os
    import cv2 as cv2
    import ast
    ## Input paths to original images
    ## Dataframe with data of measurements
    ## The output Folder
    
    #### Read in one Image
    
    data = pd.read_csv(data_eval)
    save_folder = AnalysisFolder + "/measurement_vis"
    os.makedirs(save_folder, exist_ok = True)
    
    
    if type(Imagepath) != list: ## Make a list if only one Image is submitted
      Imagepath = [Imagepath]
      
    print(f"Scale visualization calculated with mode {Scale_Mode}")
    
    if Scale_Mode == 1:
      print("NOTE: Your dataset has only one scale, but scale detection will always display the individual scale that was detected in an image")
      print("To inspect the calculated value over the whole dataset check the results.csv. To see more info look into the program documentation")
    print("Creating visualizations: ")
    for x in range(len(Imagepath)):
      im = cv2.imread(Imagepath[x])
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

      plt.clf()
      ### split the path:
      filename = Imagepath[x].split("/")[-1]
      Image_index = data.index[data['Name'] == filename]
      plt.imshow(im)
      #print(filename, Image_index)
      
      Image_index = Image_index[0] ## Make Int64IndexSeries of one number into a number
      
      plt.imshow(im)

      ## Drawing measurements
      
      try:
        plt.plot([data["Center_X_Sb"][Image_index],data["Center_X_Eye"][Image_index]],[data["Center_Y_Sb"][Image_index],data["Center_Y_Eye"][Image_index]], color = "red", linewidth=0.5)
        plt.plot([data["Center_X_Sb"][Image_index],data["Center_X_St"][Image_index]],[data["Center_Y_Sb"][Image_index],data["Center_Y_St"][Image_index]], color = "red", linewidth=0.5) 
        ### Plot the the length
        plt.plot(data["Center_X_Eye"][Image_index],data["Center_Y_Eye"][Image_index],marker="o", markersize=2, markeredgecolor="black", markerfacecolor="white")
        plt.plot(data["Center_X_Sb"][Image_index],data["Center_Y_Sb"][Image_index],marker="o", markersize=2, markeredgecolor="black", markerfacecolor="white") 
        plt.plot(data["Center_X_St"][Image_index],data["Center_Y_St"][Image_index],marker="o", markersize=2, markeredgecolor="black", markerfacecolor="white") 
      except:
        pass
      
      try:
        # plot the width
        # The width needs to be transposed back from our turned image into the real data Easiest way would be while turning
        plt.plot([data["Width_X1"][Image_index],data["Width_X2"][Image_index]],[data["Width_Y1"][Image_index],
        data["Width_Y2"][Image_index]], color = "blue", linewidth=0.5, linestyle='dotted')
        plt.plot(data["Width_X2"][Image_index],data["Width_Y2"][Image_index],marker="o", markersize=2, markeredgecolor="black", markerfacecolor="blue")
        plt.plot(data["Width_X1"][Image_index],data["Width_Y1"][Image_index],marker="o", markersize=2, markeredgecolor="black", markerfacecolor="blue")
      except:
        pass

      ### Drawing scale values  
      try:

        ## If scale is prevented skip scale illustration

        if Scale_Mode == 0:
          print("Scale given manually no calculation possible")
        
        else:
          
          ## Illustrate measured lines in the image
          ## Lines in format (x1, y1, x2, y2):
          ## Elements are string

          Coordinates = ast.literal_eval(data["coordinates_scale"][Image_index])
          X1 = Coordinates[0][0]
          X2 = Coordinates[0][2]
          Y = Coordinates[0][3]
          
          plt.plot([X1,X2],[Y -50, Y - 50], color = "green", linewidth=0.5)
          
          ## Make a dashed line that shows the connection downwards to the real scale
          
          plt.plot([X1,X1],[Y -50 , Y], color = "green", linewidth=0.5, linestyle = "dashed")
          
          plt.plot([X2,X2],[Y -50 , Y], color = "green", linewidth=0.5, linestyle = "dashed")
          
          
          ## Add textbox to the image that shows detected values
          font_size = min(im.shape[1], im.shape[0]) // 400
          text = str(data["metric_length"][Image_index]) + 'mm'
          text_color = 'green'
          text_x = X1 + (X2-X1)/2
          text_y = Y - 75
          plt.text(text_x, text_y, text, color=text_color, fontsize=font_size, ha='center', va='center')
          
      except:

        ### If no values were detected we print a message
        font_size = min(im.shape[1], im.shape[0]) // 200
        plt.text(im.shape[1] -(im.shape[1]/4), im.shape[0]/2, "No scale or number detected", color="green", fontsize=font_size)
        pass

      plt.axis('off')

      ## Save the image
      plt.savefig(save_folder +"/visualization_of_" + filename,dpi=1000,bbox_inches= "tight",pad_inches=0)

      from Convert_to_JPG import progress_bar
      
      progress_bar(x+1,len(Imagepath))