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
  # Detects lines used for scale, always
  # counts the saves the shortest scale
  # in image
  # Input: image names in folder
  # Output: line coordinates and length
  ###################################
  
  import cv2 as cv2
  import numpy as np
  from PIL import Image
  import matplotlib.pyplot as plt
  
  list_of_lengths = []
  list_of_cropped_images = []
  line_Coors = []
  Index_Lines = []
  for x in range(len(Image_names)):
      img = cv2.imread(Image_names[x])
      
      
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      #thresh, gray_thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
      #thresh = cv2.threshold()

      ## why should we blur? Image lines should be sharp -> thresholding should be the most effective
      ## In thresholding we need to consider colors but should be managable if we make an OTSU and 
      ## hard cut off for white backgrounds with white scales
      kernel_size = (5,1) # We do not wnt to blur the x axis as we need the length information 
      blur_gray = cv2.GaussianBlur(gray,(kernel_size),0)
    
      low_threshold = 100
      high_threshold = 200
      edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
      
      #### Get lines 
      
      rho = 1 # distance resolution in pixels of the Hough grid
      theta = np.pi / 180  # angular resolution in radians of the Hough grid
      threshold = 10  # minimum number of votes (intersections in Hough grid cell)
      min_line_length = 100 # minimum number of pixels making up a line
      max_line_gap = 0  # maximum gap in pixels between connectable line segments
      line_image = np.copy(img) * 0  # creating a blank to draw lines on
      # Run Hough on edge detected image
      # Output "lines" is an array containing endpoints of detected line segments
      
      lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)
      
      if lines is None:
        list_of_lengths.append(0) 
      
      else:
        
        combined = merge_lines(lines)
        line_Coors.append(combined)
        for line in combined:
          x1, y1, x2, y2 = line
          cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
      
        lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
        
        #plt.clf()
        #plt.imshow(line_image)
        #plt.show()
        #### How do we get the right line?
        #### IDEA seperate all rows with values over 0. Then take the shortest one as the others would represent the frame
        #### Could also optimize to take frame and look into it
        #### Problem: we have more than one value for the middle row. Mean? Smallest? Biggest?
        
        # Calculate the length of a line by taking the difference in x-values
        lengths = []
        for entries in range(len(combined)):
          temp_len = abs(combined[entries][0] - combined[entries][2])
          lengths.append(temp_len)
          
 
        ### To prevent a non existing value changing the lenght of a list (like missing scale we set pixelper Unit to 0)
        ### and then assign it the real value if existent
        
        PixelPerUnit = min(lengths) ## take the min value 
        #IndexOfMinLine = np.argmin(SumNoZeros)
        
        list_of_lengths.append(PixelPerUnit) 
        #Index_Lines.append(IndexOfMinLine)
  return(list_of_lengths, line_Coors)

def combine_lines(line_input, min_line_length=25):
    
    # Combine lines that are close in y coordinate.
    # Input: lines, line length
    # Output merged lines
    # initialize dictionary to store lines by y2 coordinate
    import numpy as np
    line_dict = {}

    # iterate over each line
    for line in line_input:
        # get y2 coordinate of line
        y1 = line[0][1]  # Modification: Retrieve y1 coordinate
        y2 = line[0][3]  # Modification: Retrieve y2 coordinate
        
        # if y2 already exists in dictionary, combine with existing line
        if y2 in line_dict:
            existing_line = line_dict[y2]
            xmin = min(line[0][0], existing_line[0])
            xmax = max(line[0][2], existing_line[2])
            line_dict[y2] = [xmin, y1, xmax, y2]  # Modification: Update y1 coordinate
        # otherwise, add line to dictionary
        else:
            line_dict[y2] = line[0]

    # filter out lines shorter than min_line_length pixels
    combined_lines = [line for line in line_dict.values() if (line[2] - line[0]) >= min_line_length]
    
    # sort lines by y2 coordinate
    combined_lines.sort(key=lambda x: x[1])

    # merge lines that are within 5 pixels in y coordinate of each other into one line
    merged_lines = []
    current_line = combined_lines[0]
    for line in combined_lines[1:]:
        if line[1] - current_line[3] <= 5:
            current_line[2] = max(current_line[2], line[2])
            current_line[3] = line[3]
        else:
            merged_lines.append(current_line)
            current_line = line
    merged_lines.append(current_line)
    
    # convert merged lines to array
    merged_lines_arr = np.array(merged_lines, dtype=np.int32)
    
    # Resulting lines can be slanted which does not matter as we 
    # compare only x-values in the end
    # extract unused lines
    unused_lines = [line for line in line_input if line[0][3] not in line_dict]

    # convert unused lines to array
    unused_lines_arr = np.array([line[0] for line in unused_lines], dtype=np.int32)
    print(merged_lines_arr,unused_lines_arr)
    # combine merged and unused lines into one array
    # Check if unused_lines_is empty
    if np.any(unused_lines_arr) == True:
      result = np.concatenate((merged_lines_arr, unused_lines_arr), axis=0)
    else:
      result = merged_lines_arr
    return result

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

### Testing space for image prep
### 225 or 468
### Make Lines to Int ### Do we make a config for all Codes?
### 0 = no scale native resolution of 2.8 px/µm with standard zoom (If you know the configuration of
### your own microscope enter it into the config.cfg)
### 1 = one scale for all images (automatically detects scale and Unit for every
### image but takes the most common value for all images -> robust)
### 2 = Multiple scales: tries to find a scale for every image. May be prone to
### error
### Set cutouts to uniform DPI


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


### The new approach is split differently to increase the accuracy

# 1. Make image gray
# 2. Detect the line, find color and length of it
# 3. Cut out sourrinding area 
# 3a. Optionally delete only the line
# 4. Normalize DPI
# 5. Binarise, Invert and Threshold the image based
# 6. Detect with Tesseract
# 7. Save in Df

a,b = Images_list("/home/philipp/Image_repository/SimonaAig21d")
len(a)
c,d = getLineLength(a)
set()
### Debug length a lot of 0s
### Still need to merge lines
d[1]
set(c)
def CutAndColour(Line_coordinates, Image_names, Line_index):
  # Input: Coordinates of line Best case this line already sorted so taht only one exists
  # Output: Two lists, one containing all colour data in format white (0/1)
  # and a list of the scale cut out based on coordinates
  import cv2 as cv2
  import numpy as np
  
  colour_list = []
  list_of_crops = []
  for x in range(len(Image_names)):
      img = cv2.imread(Image_names[x])
      
      ### To get length we iterate over the detecte line
        y = (Line_coordinates[x][0][1], Line_coordinates[x][0][3])
        x = (Line_coordinates[x][0][0], Line_coordinates[x][0][2])
        
        tempx.append(Line_coordinates[x][0][0])
        tempx.append(Line_coordinates[x][0][2]) 
        
      Min_Max_values = (max(x), min(x), max(y), min(y)) ## Format xmax, xmin, ymax, ymin
      
      ## Cut the image we give 5% in x coordinate as buffer to detect the number
      
      height, width = img.shape
      
      buffer = int(width*0.05)
      
      cropped_img = img[Min_Max_values[3]:Min_Max_values[2],Min_Max_values[0]+buffer:Min_Max_values[1]+buffer]
      
      ## Now we want the colour of the line
      ## So we make the image gray and extract the pixel values of of it
      
      cropped_img_gray = cv2.cvtCOLOR(cropped_img, cv2.COLOR_BGR2GRAY)
      
      Scale_colour = sum(cropped_img_gray[:,Min_Max_values[1]])/255
      
      # Scale and number should normally be the same colour so Scale_colour = Value_colour
      
      if Scale_colour > (255/2): ## If colour is white
        ret,thresh = cv2.threshold(gray,Scale_colour - 10,255,cv.THRESH_BINARY_INV) # All values equal or higher our scale are set to 0  
     
      else: # If black
        ret,thresh = cv2.threshold(gray,Scale_colour + 10,255,cv.THRESH_BINARY) # All values equal or lower our scale are set to 255 
      
      
      list_of_crops.append(thresh)
      
      
  return(list_of_crops)
c[11]
import cv2 as cv2
import matplotlib.pyplot as plt
img = cv2.imread(a[-2])
plt.imshow(img)
plt.show()
from collections import Counter
Counter(c)

combine_lines(d[-1])
c.index(0)
d[-2]
#### Sometimes two line exists that shouldn't like entry -2 why where in combine lines is this happening
lineds = [[72, 48, 33, 48],
[3, 48, 85, 48],
[49, 48, 77, 48],
[56, 48, 19, 48],
[96, 58, 87, 58]]

def merge_lines(lines):
    if len(lines) > 1:
      merged_lines = []
  
      # Sort lines based on y-coordinate
      lines = sorted(lines, key=lambda line: line[1])
      print(lines)
      # Merge lines within 5 pixels in y-coordinate
      merged_line = lines[0]
      for line in lines[1:]:
          if line[1] - merged_line[3] <= 5:
              merged_line[2] = max(merged_line[2], line[2])
              merged_line[3] = line[3]
          else:
              merged_lines.append(merged_line)
              merged_line = line
  
      merged_lines.append(merged_line)
  
      return merged_lines
    else:
      return lines
    
    
merged = merge_lines(lineds)
def merge_lines(lines):
    merged_lines = []

    # Sort lines based on y-coordinate
    lines.sort(key=lambda line: line[1])

    # Merge lines within 5 pixels in y-coordinate
    current_line = lines[0]
    for line in lines[1:]:
        if line[1] - current_line[3] <= 5:
            current_line[2] = max(current_line[2], line[2])
            current_line[3] = line[3]
        else:
            merged_lines.append(current_line)
            current_line = line

    merged_lines.append(current_line)

    # Calculate mean y-coordinate for merged lines
    mean_y = sum(line[1] for line in merged_lines) / len(merged_lines)

    # Get x-coordinate of the first line as xmin
    xmin = merged_lines[0][0]

    # Calculate xmax
    xmax = max(line[2] for line in merged_lines)

    merged_line = [xmin, mean_y, xmax, mean_y]

    return merged_line

def group_lines(lines):
    groups = []
    
    # Sort lines based on y-coordinate
    lines.sort(key=lambda line: line[1])
    
    current_group = [lines[0]]
    
    for line in lines[1:]:
        if line[1] - current_group[-1][3] <= 5:
            current_group.append(line)
        else:
            groups.append(current_group)
            current_group = [line]
    
    groups.append(current_group)
    
    return groups
