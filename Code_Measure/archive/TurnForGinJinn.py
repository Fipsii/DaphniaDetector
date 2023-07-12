###### This script is designed to rotate daphnia Images up by their eye coordinate
###### Then make 5 5 degree steps in every direction
###### Save these images and drop them put them into ginjinn with a defined model
###### Drop every duplicate except the one with smallest bbox

def Images_list(path_to_images):
  import os as os
  PureNames = []
  Image_names = []
  for root, dirs, files in os.walk(path_to_images, topdown=False):
    #print(dirs, files)
    for name in files:
      #print(os.path.join(root, name))
      Image_names.append(os.path.join(root, name))
      PureNames.append(name)
      #print(files)
  return Image_names, PureNames

def point_trans(ori_point, angle, ori_shape, new_shape):
    
    # Transfrom the point from original to rotated image.
    # Args:
    #    ori_point: Point coordinates in original image.
    #    angle: Rotate angle.
    #    ori_shape: The shape of original image.
    #    new_shape: The shape of rotated image.
    # Returns:
    #    Numpy array of new point coordinates in rotated image.
    
    import math
    import numpy as np

    dx = ori_point[0] - ori_shape[1] / 2.0
    dy = ori_point[1] - ori_shape[0] / 2.0

    t_x = round(dx * math.cos(angle) - dy * math.sin(angle) + new_shape[1] / 2.0)
    t_y = round(dx * math.sin(angle) + dy * math.cos(angle) + new_shape[0] / 2.0)
    return np.array((int(t_x), int(t_y)))

def getDaphnidlength(RotatedCutImages):
  ## We want to find the length of a Daphnid (body) to calculate the body width
  ## as we want to measure at 1/3 of the Distance counting from the head
  ## we count from both sides to avoid needing to detect the head
  ## take every 50th row and calcualted the amount of pixel that are non-zero?
  ## do we get problem with the eye? Binary mask has no eye!
  line_coordinates = []
  for x in range(0,len(RotatedCutImages)):
    import numpy as np
    ### Find the index row where the line is the longest
    ### 
    
    columns = np.copy(RotatedCutImages[x][:, ::50]) ### take every 50th column
    #np.shape(RotatedCutImages[0][:, ::50])
    #np.shape(RotatedCutImages[0])
    sum_of_columns = np.sum(columns,axis=0)
    Longest_column = np.where(sum_of_columns == np.amax(sum_of_columns))
    Index_of_the_Longest_column = int(Longest_column[0])*50 ## If mutliple same values exist take the first one
    print(x)
    ### Now we need to find the y coordinate where this line goes from 0 to some value
    ### Therefore we extract the line out of the image
    ### Somewhere we confuse columns and rows!!!!
  
    full_image = np.copy(RotatedCutImages[x])
    full_column = full_image[:,Index_of_the_Longest_column]
    
    #### No I want the index of the first number of 0 and the last over 0
    
    firstNonZero = np.argmax(full_column>0) ### Argmax stops at first true
    lastNonZero = len(full_column) - np.argmax(np.flip(full_column>0)) ## Flip to come from the other side
    
    #### Check if we have a continuus line but does it matter? Its a definition thing and dependent on how well rembg cuts out
    #### We have to test if making a binary mask may help exculding extremities while including the body -> species dependent
    
    line_coordinates.append((firstNonZero,lastNonZero))
  return line_coordinates
  #### Plot the line into the image

def DaphnidWidth(coordinates, RotatedCutImages):
  import numpy as np
  from matplotlib import pyplot as plt
  
  body_width_px = []
  
  for x in range(len(RotatedCutImages)):
    
    plt.clf()
    plt.imshow(RotatedCutImages[x])
    
    upper_third = int(coordinates[x][0] + (coordinates[x][1] - coordinates[x][0]) * 1/3)
    
    #### Get upper measurement ### We avoid a nested for loop for readability
    #### We measure from the outside on the upper third until we reach the daphnid
    #### We then want to find the midpoint of these two coordinates
    #### and measure from the inside
    #### We have to split the image on the centre point and measure the flipped left and
    #### normal right half and add them together
    #### This lets us avoid to detect extremities on the outside
    #######################################################################################
    body_width_temp = []
    
    row = RotatedCutImages[x][upper_third,:] 
    row_Left = np.argmax(row > 100) ## get the start coordinate of the body on the left
    row_Right = len(row) - np.argmax(np.flip(row > 100)) ## get the end 
    
    ### find the halfway coordinate between left and Right
    # Cut the image into left and right along the middle of the outsides we found
    row_middle = round((row_Right + row_Left)/2)
    
    left_half = RotatedCutImages[x][:, :row_middle]
    right_half = RotatedCutImages[x][:, row_middle:]
    
    #### Find the values for the split daphnid
    
    row_middle_to_left = np.argmax(np.flip(left_half[upper_third,:]  == 0))
    row_middle_to_right = np.argmax(right_half[upper_third,:]  == 0)
    
    ### No we want to translate the the points back to our old image
    ### right coordinate would be the row_upper_third_middle_to_right
    ### + width of the left half
    
    coor_right = len(left_half[upper_third,:]) + row_middle_to_right
    
    ### the left coordinate the width of left box - row_upper_third_middle_to_right we found
    coor_left = len(left_half[upper_third,:]) - row_middle_to_left
    
    ### The length is the difference between these left and rigth
    body_width_px.append(coor_right - coor_left) 
    
    #plt.plot(coor_left, Thirds[y], coor_right, Thirds[y], marker = 'o', ls = '-')
    ######################################################################### right
    
  return body_width_px
    
    ## PRELIMINAIRY: 
    ## We also need to consider extremities increasing width - unsolved; Fit with mask/oval
    ## As well as intestines being 0 values and fragments - solved; fill contour fill 
    ## And how to measure width do we take most outer point or not? - solved; measure from the inside
    ## Variance in Daphnid length -> do we take Daphnid length from mask or csv? - in work; Csv values -> transform and get x,y of uncropped image
    ## Cropping: Might not be needed -> reduces calc time
    ## Rotation: Eye rotation leaves many Daphnids skewed to a side not straight ca. 5-10°- unsolved 

def body_width_to_csv(List_of_names ,List_of_widths):
  from Yaml_load_test import ConfigImport
  import pandas as pd
  
  settings = ConfigImport("/home/philipp/GitRStudioServer/Workflow_code/settings.yml")
  
  ## First reduce Imagenames path to Image names
  
  body_widths_df = pd.DataFrame()
  body_widths_df['Name'] = Image_Names
  body_widths_df['Bodywidth[px]'] = List_of_widths
  print(body_widths_df)
  

  dfPixel = pd.read_csv(settings["Annotation_path"][:-5]+".csv")
  dfPixel = dfPixel.rename(columns={'image_id': 'Name'})
  print(dfPixel)
  
  dfPixel = pd.merge(dfPixel,body_widths_df, on = "Name", how = "inner")
  dfPixel.to_csv(settings["Annotation_path"][:-5]+".csv", index = False)

def Rotate_by_eye(Annotationfile, path_to_images, Image_title, saveloc=False):
  # Turn img upright based on the eye position.
  # Args:
  #    AnnotationFrame: Point coordinates in original image.
  #    path_to_images: The path to the images
  #    Image_title: A list of the names XXXX.xyz
  #    saveloc: a folder that is goin to be generated (Should not exist!)
  # Returns:
  #    rotated_image_list as list of images with every iamge turned up.
  #
  # Do not feed calibration images into this function
  import cv2 as cv2
  import pandas as pd
  from scipy import ndimage
  import math
  import numpy as np
  import time
  import imutils
  import matplotlib.pyplot as plt
  start_time = time.time()
  if saveloc != False:
    import os
    os.mkdir(saveloc)
  #data = pd.read_csv(AnnotationFrame)
  data = Annotationfile
  #FrameCopy = rotated_image_list.copy()
  turning_angle = []
  turning_radiant = []
  Rotated_Images = []
  for item in range(len(path_to_images)):
    #print(item)
    #try:
      #### img = cv2.imread(path_to_images[item])
      #### Now we try to find the new middle coordinate of the eye
      #### We need the rotation angles from before
      #### MiddleEyes = list(zip(data["Center_X_Eye"],data["Center_Y_Eye"]))
  
      #### Now we want to rotate the image and y coordinates 180 degrees in both diretions
      #### if y gets higher by rotation we want interrupt the while loop and rotate into
      #### the other direction
      
      if Image_title[item] in data['Name'].tolist():
        #print('What')
        Row_of_Image = data[Image_title[item] == data['Name']] ### get the data for the image
         
        CoorEye = int(Row_of_Image["Center_X_Eye"].iloc[0]),int(Row_of_Image["Center_Y_Eye"].iloc[0])
        OrgShape = int(Row_of_Image["Imageheight"].iloc[0]),int(Row_of_Image["Imagewidth"].iloc[0])
          
        ### Now rotate the image by a set amount of radians
        list_of_y_coordinates = []  
        deg = list(range(0,360,1)) ## needed for ndimage.rotate
        rad = np.deg2rad(deg) ## needed for pointtrans
        y_old = CoorEye[1] ## Starting point
      
        for angles in range(len(rad)):
          
          #rotation = ndimage.rotate(img, deg[angles])
          #NewShape = np.shape(rotation)
          new_Coor = point_trans(CoorEye, rad[angles] , OrgShape, OrgShape)
          list_of_y_coordinates.append(new_Coor[1])
        
        turning_angle.append(-deg[list_of_y_coordinates.index(max(list_of_y_coordinates))]+180)
        turning_radiant.append(rad[list_of_y_coordinates.index(max(list_of_y_coordinates))])
        
        img = cv2.imread(Image_Path[item])
        img_rotate = imutils.rotate_bound(img, -turning_angle[item])
        Rotated_Images.append(img_rotate) 
        #print('What2')
        if saveloc != False:
          plt.clf()
          plt.imshow(img_rotate)
          plt.savefig(saveloc + "/" + Image_Path[item].split("/")[-1])
        
        else:
          continue
      else:
        print(path_to_images[item] + " could not be used not in dataframe and/or folder")
        
    #except: 
      #print("Error: File " + Image_title[item] + " not turned")
      
  print("--- %s seconds ---" % (time.time() - start_time)) # or turned images?
  return turning_angle, Rotated_Images, turning_radiant

def CreateDuplicates(RotateImages, Image_Path, Image_title, savefolder, InFolders=False):
  import imutils
  import cv2 as cv2
  import os
  degrees_rot = list(range(-25,30,5))
  for images in range(len(Image_title)):
    if InFolders == False:
      for degrees in degrees_rot:
        temp_rot = imutils.rotate_bound(RotateImages[images],degrees)
        cv2.imwrite(savefolder + Image_title[images] + "_" + str(degrees) +"_deg.jpg", temp_rot)
    else:
      os.mkdir(savefolder + Image_title[images])
      for degrees in degrees_rot:
        temp_rot = imutils.rotate_bound(RotateImages[images],degrees)
        cv2.imwrite(savefolder + Image_title[images] + "/" + str(degrees) +"_deg.jpg", temp_rot)

import pandas as pd
TestDf = pd.read_csv("/home/philipp/GitRStudioServer/ImageData/SimonaRelabelled/annotations/instances_default.csv")
Image_Path, Image_Names = Images_list("/home/philipp/GitRStudioServer/ImageData/SimonaAig21d")
RotationAngleByEye, RotationImageEye, RadiantEye = Rotate_by_eye(TestDf,Image_Path,Image_Names,"/home/philipp/BodyWidthApproaches/FastRotateCut/")
### Why does it only work on second try?
### Here we yield upright Daphnids which we can cut and modify how we want
### Next step would be cropping the body to reduce the amount of background and/or use rembg
CreateDuplicates(RotationImageEye, Image_Path, Image_Names, "/home/philipp/RotatedImages_Folder/")


