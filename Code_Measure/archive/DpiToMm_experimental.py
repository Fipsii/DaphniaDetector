### Goal: Read in an Image an autoamtically detect the Scale used:
## Caveats: Maybe we will need manual input of scale, 
##Do we perform this per Image or once provided the person doesn't change the zoom 


#### 
#### Read in all Images and save their respective paths and filenames
def Images_list(path_to_images):
  import os as os
  PureNames = []
  Image_names = []
  for root, dirs, files in os.walk(path_to_images, topdown=False):
    #print(dirs, files)
    for name in files:
      #print(os.path.join(root, name))
      Image_names.append(os.path.join(root, name))
  for x in range(0,len(Image_names)):
    PureNames.append(Image_names[x].split("/")[-1])
  return Image_names, PureNames

def getLineLength(Image_names):
  ## Gaussiaun blur and image read ##
  ###################################
  import cv2 as cv2
  import numpy as np
  from PIL import Image
  import matplotlib.pyplot as plt
  
  list_of_lengths = []
  list_of_cropped_images = []
  line_Coors = []
  for x in range(len(Image_names)):
    img = cv2.imread(Image_names[x])
    
    height = img.shape[0]
    width = img.shape[1]
    cropped_image = img[int(height*(3/4)):height,int(width*(1/2)):width]
    gray = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY)
    thresh, gray_thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    #thresh = cv2.threshold()
    list_of_cropped_images.append(gray_thresh)
    
    ## why should we blur? Image lines should be sharp -> thresholding should be the most effective
    ## In thresholding we need to consider colors but should be managable if we make an OTSU and 
    ## hard cut off for white backgrounds with white scales
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray_thresh,(kernel_size, kernel_size),0)
    
    low_threshold = 100
    high_threshold = 200
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    
    #### Get lines 
    
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100 # minimum number of pixels making up a line
    max_line_gap = 0  # maximum gap in pixels between connectable line segments
    line_image = np.copy(cropped_image) * 0  # creating a blank to draw lines on
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)
    line_Coors.append(lines)
    if lines is None:
      list_of_lengths.append(0) 
    else:
      for line in lines:
        for x1,y1,x2,y2 in line:
          cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
    
      lines_edges = cv2.addWeighted(cropped_image, 0.8, line_image, 1, 0)
      #plt.clf()
      #plt.imshow(lines_edges)
      #plt.show()
      
      #### How do we get the right line?
      #### IDEA seperate all rows with values over 0. Then take the shortest one as the others would represent the frame
      #### Could also optimize to take frame and look into it
      #### Problem: we have more than one value for the middle row. Mean? Smallest? Biggest?
      
      Summe = np.sum(line_image[:,:,0], axis = 1) ## Takes the red values and sums every row
      SumNoZeros = Summe[Summe != 0] ### Drop all 0s from the frame
      ### To prevent a non existing value changing the lenght of a list (like missing scale we set pixelper Unit to 0)
      ### and then assign it the real value if existent
      PixelPerUnit = SumNoZeros.min()/255 ## take the min value 
      list_of_lengths.append(PixelPerUnit) 
      
  return(list_of_lengths, list_of_cropped_images, line_Coors)

#### No we need to get the Number above the line
# def getUnit(cropped_Image) ### For py Tesseract we need sudo install
### white only needs mask
### We need to check if the scale is white
### Therefore we check if we have more 0 pixels in the mask than 255
def get_Scale(cropped_images, lines_edges, Line_coordinates, in_Range_upper_limit=200, psm_mode=7):
  import pytesseract as pytesseract
  import matplotlib.pyplot as plt
  import cv2 as cv2
  ScaleUnit = []
  pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
  for x in range(len(cropped_images)):
    # Convert the image to grayscale
    #img_gray = cv2.cvtColor(cropped_images[x], cv2.COLOR_BGR2GRAY)
    # Define the region of interest (ROI) to crop
    height, width = cropped_images[x].shape
    # Crop the image
    
    ymin = 0
    ymax = 0
    xmin = 0
    xmax = 0
    
    if Line_coordinates[x] is None:
      crop_img = cropped_images[x]
    else:
      for n in Line_coordinates[x]:
        ##get the longest line
        for x1,y1,x2,y2 in n:
          if ymin < y1:
            ymin = y1
          if ymax < y2:
            ymax = y2
          if xmin < x1:
            xmin = x1
          if xmax < x2:
            xmax = x2
            
      if abs(xmin - xmax) > 100:
        crop_img = cropped_images[x][ymin - int(height*0.2):ymax + int(height*0.2), xmin:xmax]
      
      else:
        crop_img = cropped_images[x][ymin - int(height*0.2):ymax + int(height*0.2), int(width*0.5):width]
    
    #print("Y", ymin,ymax, "X", xmin, xmax, "Img", crop_img.shape, "Org_Img", cropped_images[x].shape)
    plt.clf()
    plt.imshow(crop_img)
    plt.show()
    try:
      gaussian_3 = cv2.GaussianBlur(crop_img, (0, 0), 2.0)
      unsharp_image = cv2.addWeighted(crop_img, 2.0, gaussian_3, -1.0, 0)
    except:
      gaussian_3 = crop_img
    # Apply thresholding
    thresh, img_thresh = cv2.threshold(crop_img, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Invert the binary image
    img_thresh_inv = cv2.bitwise_not(img_thresh)
    
    
    #img_whiteBG = cv2.resize(img_thresh_inv, None, fx=2, fy=2)
    #img_blackBG = cv2.resize(img_thresh, None, fx=2, fy=2)
    #print(height, width, "&&", crop_img.shape)
    #plt.clf()
    #plt.imshow(img_whiteBG, cmap = "gray")
    #plt.show()
    ## tesseract detect
    
    tesseract_config = r'--oem 3 --psm ' + str(psm_mode) +" -c tessedit_char_whitelist=0123456789."
  
    ## as we dont no which kind of scale was put in we use all 3 (inverse, contrast for white and MasCorrNormBare) and then compare.
    ## We could find out the colour of the scale when we use the lines, but is it necessary?
  
    number = pytesseract.image_to_string(img_thresh_inv, config= tesseract_config).strip()
    number2 = pytesseract.image_to_string(img_thresh, config= tesseract_config).strip()
    ScaleUnit.append([number,number2])
  return(ScaleUnit)

########### We also should detect mm or µm this can be achieved with pytesseract. but we also can guess as 
########### high values should always be µm and low ones mm ## psm 7 is good by far not perfect... Run multiple modes?

def NormalizeScale(ReadInsScale): ### Make all values into mm and decide which value is true and which not
  #### We will drop 0 values out of the list and convert 7 into 1s as nobody has a 7 as scale
  #### We then compare if its the only value take it
  #### If we have mutliple same take that
  #### If we have multiple but different values: Ask user? have a list of likely numbers and throw a warning? likely numbers: 1,2,100,200,250,300,400,500,600,700,750,800,900,1000,5
  #### If we have multiple likely numbers? ask USer? 
  import re

  likely_numbers = [1,2,100,200,250,300,400,500,600,700,750,800,900,1000,5]
  ScaleUnitsClear = []
  str_list = []

  for x in range(len(ReadInsScale)): ## for all
    list_of_sub = []

    for sub_numbers in range(len(ReadInsScale[x])):
      Real_Number = re.findall(r"[-+]?(?:\d*\.*\d+)", ReadInsScale[x][sub_numbers])
      
      if not Real_Number:
        list_of_sub.append(0)
      else:
        list_of_sub.append(int(Real_Number[0])) ### Drop all empty entries
    
    str_list.append(list_of_sub)
  
  #print(str_list)
  empty = []
  x = 0
  for x in range(len(str_list)): ## for all entries
    #print(x)
    temp = []
    for y in str_list[x]: ### if all values in every list entry
      if int(y) in likely_numbers: ### if a value is in the likely numbers list
        temp.append(y) ## make the entry the number
    empty.append(temp)
  
  for i in range(len(empty)):
    if len(empty[i]) == 0:
        empty[i] = 0
  
  for i in range(len(empty)):
    if isinstance(empty[i],list):
        empty[i] = int(empty[i][0])
  

  ### Delete uncertainties and set them to 0
  ### Make empty values 

  ### Now we need to make every value above 50/1000
  for x in range(len(empty)):
    if empty[x] > 49 and empty[x] != 0:
        empty[x] = empty[x]/1000
  return(empty)  
  
def makeDfwithfactors(list_of_names, One_scale,ScaleUnitClean=[], list_of_lengths=[], ConvFactor=0.002139):
  ### This function has two modes. 1) If the user declares taht we only have one 
  ### scale we take the most common values of length and unit and 2) if more 
  ### than one exist we keep the list as they are.
  ### Then we enter the singular or mutliple values into the df
  import pandas as pd
  LengthOpt = [int(item) for item in list_of_lengths] ## Make linelengths int
  if One_scale == 0:
    Scale_df = pd.DataFrame(list_of_names, columns =['Name']) 
    Scale_df["distance_per_pixel"] = ConvFactor
    print(f"Using manual factor of {ConvFactor} px/mm")
    return Scale_df
  
  elif One_scale > 0:
    
    if One_scale == 1:
      LengthOpt = max(set(list_of_lengths), key = list_of_lengths.count)
      UnitOpt = max(set(ScaleUnitClean),key = ScaleUnitClean.count)
      
    else:
	    LengthOpt = list_of_lengths
	    UnitOpt = ScaleUnitClean
    
    Scale_df = pd.DataFrame(list_of_names, columns =['Name'])
    #print(list_of_lengths)
    Scale_df["metric_length"] = UnitOpt
    Scale_df["scale[px]"] = LengthOpt
    Scale_df["distance_per_pixel"] = Scale_df["metric_length"]/Scale_df["scale[px]"]
    
    return Scale_df

##### Testing space for image prep
##225 or 468
#### Make Lines to Int ### Do we make a config for all Codes?
### 0 = no scale native resolution of 2.8 px/µm with standard zoom (If you know the configuration of
### your own microscope enter it into the config.cfg)
### 1 = one scale for all images (automatically detects scale and Unit for every
### image but takes the most common value for all images -> robust)
### 2 = Multiple scales: tries to find a scale for every image. May be prone to
### error

## Set cutouts to uniform DPI

Images_list

def NormalizeDPI(path):
  # Resize Cutouts of the scale and enforce an uniform DPI scale ##
  # This helps tesseract ocr-to detect scales more confidently  ###
  # according to documentation
  # Input: Folder_path
  # Output: Saved Scale Images 512x512, 600 DPI at the folder path called
  # Converted Scales as Image_name_ConvScale.jpg
  
  import os
  from PIL import Image
  import re
  
  directory = "Converted_Scales"
  convert_path = os.path.join(path,directory)
  
  if os.path.exists(convert_path) == False:
    os.mkdir(convert_path)
  i = 1
  
  for filename in os.scandir(parent_path):
    
    
    result = str(filename).split('\'')

    try:
        new_file = convert_path + "/" + result[1][:-4] + "_ConvScale.jpg"
        i = i+1
        im = Image.open(filename.path)
        im.thumbnail((512,512))
        im.save(new_file,"JPEG",dpi=(600,600))
    
    except:
        continue
