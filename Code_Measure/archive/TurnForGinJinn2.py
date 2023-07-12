###### This script is designed to rotate daphnia Images up by their eye coordinate
###### Input Files needed: PATH to the original images, and annotations of the Eye Measurements
###### outpurfile for the Duplicates as well as the optional output of prerotated files
###### Then make 5 5 degree steps in every direction
###### Save these images and drop them put them into ginjinn with a defined model
###### Drop every duplicate except the one with smallest bbox Input

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
  #print(dfPixel)
  
  dfPixel = pd.merge(dfPixel,body_widths_df, on = "Name", how = "inner")
  dfPixel.to_csv(settings["Annotation_path"][:-5]+".csv", index = False)

def Rotate_by_eye(Annotationfile, Image_Path, Image_title, saveloc=False):
  
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
  
  for item in range(len(Image_Path)):
    #try:
      #### img = cv2.imread(path_to_images[item])
      #### Now we try to find the new middle coordinate of the eye
      #### We need the rotation angles from before
      #### MiddleEyes = list(zip(data["Center_X_Eye"],data["Center_Y_Eye"]))
  
      #### Now we want to rotate the image and y coordinates 180 degrees in both diretions
      #### if y gets higher by rotation we want interrupt the while loop and rotate into
      #### the other direction
      #print("Iteration Nr:", item)
      if Image_title[item] in data['image_id'].tolist():
        
        if item % 5 == 0: ### Print progress every 5. iteration
          print(f'Rotating Images {round((item/len(Image_title)*100),2)} % done')
        
        Row_of_Image = data[Image_title[item] == data['image_id']] ### get the data for the image
        CoorEye = int(Row_of_Image["Center_X_Eye"].iloc[0]),int(Row_of_Image["Center_Y_Eye"].iloc[0])
        OrgShape = int(Row_of_Image["Imageheight"].iloc[0]),int(Row_of_Image["Imagewidth"].iloc[0])
         
        ### Now rotate the image by a set amount of radians
        list_of_y_coordinates = []  
        deg = list(np.round(np.arange(0,360,0.1),2)) ## needed for ndimage.rotate
        rad = np.deg2rad(deg) ## needed for pointtrans
        y_old = CoorEye[1] ## Starting point
        
        for angles in range(len(rad)):
          
          #rotation = ndimage.rotate(img, deg[angles])
          #NewShape = np.shape(rotation)
          new_Coor = point_trans(CoorEye, rad[angles] , OrgShape, OrgShape)
          list_of_y_coordinates.append(new_Coor[1])
        #print(max(list_of_y_coordinates))
        turning_angle.append(deg[list_of_y_coordinates.index(max(list_of_y_coordinates))]+180)
        turning_radiant.append(rad[list_of_y_coordinates.index(max(list_of_y_coordinates))])
        
        img = cv2.imread(Image_Path[item])
        #print(-turning_angle[item])
        img_rotate = imutils.rotate_bound(img, turning_angle[item])
        Rotated_Images.append(img_rotate) 
        #print(img_rotate)
        if saveloc != False:
          plt.clf()
          plt.imshow(img_rotate)
          plt.savefig(saveloc + "/" + Image_Path[item].split("/")[-1])
        
        else:
          continue
      else:
        print(Image_Path[item] + " was not turned, but still appended to dataframe")
        turning_angle.append(np.nan)
        img = cv2.imread(Image_Path[item])
        Rotated_Images.append(img) 
    #except: 
      print("Error: File " + Image_title[item] + " not turned and not appended")
      
  print("--- %s seconds ---" % (time.time() - start_time)) # or turned images?
  return turning_angle, Rotated_Images, turning_radiant

def CreateDuplicates(RotateImages, Image_Path, Image_title, savefolder, InFolders=False):
  
  import imutils
  import cv2 as cv2
  import os
  import shutil
  degrees_rot = list(range(-25,30,5))
  #degrees_rot = 0
  if os.path.exists(savefolder):
    shutil.rmtree(savefolder)
  os.makedirs(savefolder)
  for images in range(len(Image_title)):
    if images % 5 == 0: ### Print progress every 5. iteration
          print(f'Creating duplicate images {round((images/len(Image_title))*100,2)} % done')
          
    if InFolders == False:
      for degrees in degrees_rot:
        temp_rot = imutils.rotate_bound(RotateImages[images],degrees)
        tempName = Image_title[images]
        cv2.imwrite(savefolder +"/"+ tempName[:-4] + "_" + str(degrees) +"_deg.jpg", temp_rot)
    
    else:
      os.mkdir(savefolder + Image_title[images])
      for degrees in degrees_rot:
        temp_rot = imutils.rotate_bound(RotateImages[images],degrees)
        cv2.imwrite(savefolder + Image_title[images] + "/" + str(degrees) +"deg.jpg", temp_rot)

def EyeCenter(DfWithEyes):
  import numpy as np
  import pandas as pd
  
  #### Calculate the center of the eye to rotate the image ###
  #### Input: Df with X,Y min and width and height of bbox ###
  #### Output df with X,Y max and X,Y centre ###
  data = DfWithEyes.copy()
  
  data["Xmax_Eye"] = data["Xmin_Eye"] + data["bboxWidth_Eye"]
  data["Ymax_Eye"] = data["Ymin_Eye"] + data["bboxHeight_Eye"]
  data["Center_X_Eye"] = (data["Xmax_Eye"] + data["Xmin_Eye"])/2
  data["Center_Y_Eye"] = (data["Ymax_Eye"] + data["Ymin_Eye"])/2
  
  return(data)

def JsonToMeasurement_turn(Annotation_file):
  import json as json
  import pandas as pd
  data = json.load(open(Annotation_file))
  ann = data["annotations"]
  Image_Ids = data["images"]
  
  Label_Ids = data["categories"]
  Imageframe = pd.DataFrame(Image_Ids)
  
  Labelframe = pd.DataFrame(Label_Ids)
  Annotationframe = pd.DataFrame(ann)

  ### Make ids into a readable format in the csv

  for x in range(1,len(Labelframe["name"])+1):
    Annotationframe["category_id"] = Annotationframe["category_id"].replace(x, Labelframe["name"][x-1])
  
  ### Same for Images ids

  for x in range(1,len(Imageframe["file_name"])+1):
    Annotationframe["image_id"] = Annotationframe["image_id"].replace(x, Imageframe["file_name"][x-1])
  
  #### Dewrangle the coordinates

  Annotationframe[['Xmin','Ymin','bboxWidth','bboxHeight']] = pd.DataFrame(Annotationframe.bbox.tolist(), index= Annotationframe.index)

  ### Make useful columns into a new data frame
  
  SmallFrame = Annotationframe[["id", "image_id","category_id","area","Xmin","Ymin","bboxWidth","bboxHeight"]]
  
  #### Make everything to one row per individual
  count = 0
  for y in Labelframe["name"]:
    temp = SmallFrame[SmallFrame["category_id"] == y] 
    temp.columns = ['id_'+ str(y), 'image_id', 'category_id_'+ str(y),'area_'+ str(y),'Xmin_'+ str(y),'Ymin_'+ str(y),'bboxWidth_'+ str(y),'bboxHeight_'+ str(y)]
    count += 1
    
    if count == 1:
      CompleteDataframe = temp
    else:
      CompleteDataframe = pd.merge(CompleteDataframe, temp, on = ["image_id"], how = "outer")
  CompleteDataframe.drop_duplicates(subset=["image_id"], keep='first', inplace=True, ignore_index=True)
  CompleteDataframe.insert(3, 'Imagewidth', Imageframe['width']) 
  CompleteDataframe.insert(4, 'Imageheight', Imageframe['height']) 
  
  List_image_id = []
  ### save only the name of the image not the path as image_id
  for x in range(len(CompleteDataframe["image_id"])):
    List_image_id.append(CompleteDataframe["image_id"][x].split("/")[-1])
  
  ## save the data
  CompleteDataframe["image_id"] = List_image_id
  CompleteDataframe.to_csv(Annotation_file[:-5] + ".csv", index = False)
  #print(CompleteDataframe.head())
  return CompleteDataframe






