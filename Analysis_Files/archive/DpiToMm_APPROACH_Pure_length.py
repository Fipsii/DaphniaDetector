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
  list_of_images = []
  line_Coors = []
  list_of_lines = []
  for x in range(len(Image_names)):
      img = cv2.imread(Image_names[x])

      #print(x)
      # Load the original RGB image
      normalized_image = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
      # Extract the R, G, B channels
      b, g, r = cv2.split(normalized_image)
      # Create a mask for pixels where R = G = B
      mask = np.logical_and(r == g, g == b)
      # Create a new grayscale image with the same size as the original image
      gray_image = np.zeros_like(r, dtype=np.uint8)
      # Set the pixels in the grayscale image where R = G = B to the corresponding pixel values
      gray_image[mask] = r[mask]
      
      ## why should we blur? Image lines should be sharp -> thresholding should be the most effective
      ## In thresholding we need to consider colors but should be managable if we make an OTSU and 
      ## hard cut off for white backgrounds with white scales
      kernel_size = (5,1) # We do not wnt to blur the x axis as we need the length information 
      blur_gray = cv2.GaussianBlur(gray_image,(kernel_size),0)
      
      low_threshold = 0
      high_threshold = 1
      edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
  
      #### Set HoughLinesP Parameters
      
      rho = 1 # distance resolution in pixels of the Hough grid
      theta = np.pi / 180  # angular resolution in radians of the Hough grid
      threshold = 100  # minimum number of votes (intersections in Hough grid cell)
      min_line_length = 100 # minimum number of pixels making up a line
      max_line_gap = 0  # maximum gap in pixels between connectable line segments
      line_image = np.copy(img) * 0
      line_image2 = np.copy(img) * 0  # creating a blank to draw lines on
      # Run Hough on edge detected image
      # Output "lines" is an array containing endpoints of detected line segments
      
      lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)
 
      if lines is None: ## If no lines are found
        line_Coors.append(0)
        list_of_lengths.append(0) 
        list_of_images.append(blur_gray)
        list_of_lines.append(0)
      else: # if there are lines
      
        Coordinates, Lengths = group_lines(lines) ## combine close lines
        #print(Lengths, Coordinates)
        # Create an empty image for drawing the lines
        
        # Discard extreme lines###
        # Case 1 Long lines at image edge
        # Hough Lines sometimes shows behavior taking edges of the iamge as line
        # But y = 0 and y = image_height is never a applicable line
        
        max_y = edges.shape[0] - 1  # Subtract 1 since indexing starts from 0
        filtered_lines = []
        filtered_lengths = []
        
        for length, line in zip(Lengths, Coordinates): ## If one of the rows is the beginnig or end of the image delete
          #print(line,length)
          # and delete the length value # y is always the same for both
          if line[0][1] != max_y and line[0][1] != 0:
            
            filtered_lines.append(line)
            filtered_lengths.append(length)
        
        # Case two lines that are over the minimum 100px but over 50% smaller than
        # the other lines in the list, which would make it a fragment
        
        filtered_lines_step2= []  # List to store the filtered lines
        filtered_lengths_step2 = []
        
        max_length = max(filtered_lengths)  # Find the maximum length
        print(max_length, filtered_lengths, x)
        if len(filtered_lengths) > 1:
            for line, length in zip(filtered_lines, filtered_lengths):
                if length < 0.5 * max_length:  # Check if length is less than 50% of max length
                    continue  # Skip this line and length
                else:
                    filtered_lines_step2.append(line)
                    filtered_lengths_step2.append(length)
        
        else: ## If we have only one value -> no list but int then
          filtered_lines_step2 = filtered_lines
          filtered_lengths_step2 = filtered_lengths
        
        ## Now we want to select for the right line We have two conditions:
        ## If we find only one or two lines we take the shortest line we find
        ## If we have more lines we take the inner lines as the scale is contained
        ## in the box. This allows resilience against not completly detected boxe edges
        ##########################################################################
        
        print(filtered_lengths_step2,filtered_lines_step2)
        ## Get the coordinates of the shortest line
        
        if len(filtered_lines_step2) < 3: ## If only 1-2 lines left we take the 
          # shortest line we find
          Idx = filtered_lengths_step2.index(min(filtered_lengths_step2))
          Correct_Coor = filtered_lines_step2[Idx]
        
          x1, y1, x2, y2 = Correct_Coor[0]
          
          cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        else: 
          # If more lines are left we take a line from the middle
          # This is only robust if we 1) merged all lines correctly
          # or upper and lower box lines are detected.
          # If we detect two lines for upper boundary and none for the lower
          # as well as 1 for the real scale we select a false value
          
          Middle_line = len(filtered_lines_step2)//2
          Correct_Coor = filtered_lines_step2[Middle_line]
          x1, y1, x2, y2 = Correct_Coor[0]
          
          cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    
        ## Control Area
        test = cv2.addWeighted(img, 0.2, line_image, 0.8, 0)
        
        for line in lines: ## No plot raw lines
          for x1,y1,x2,y2 in line:
            cv2.line(line_image2,(x1,y1),(x2,y2),(255,0,0),1)
        
        test2 = cv2.addWeighted(img, 0.2, line_image2, 0.8, 0)
        plt.imsave(f"/home/philipp/Scale_Values_combined/Simona/Image_test1_{x}.jpg",arr = test, dpi = 600)
    
        
        ## Result Area
        PixelPerUnit = min(filtered_lengths_step2) ## take the min value of length as scale
  
        # Append values to lists
        line_Coors.append(Correct_Coor)
        list_of_lengths.append(PixelPerUnit) 
        list_of_images.append(img)
        list_of_lines.append(lines)
  return list_of_lengths, line_Coors, list_of_images, list_of_lines 

def group_lines(lines):
    groups = []
    fused_lines = []
    extracted_lengths = []

    # Convert lines to a list if it is a NumPy array
    if isinstance(lines, np.ndarray):
        lines = lines.tolist()
    
    # Sort lines based on y-coordinate
    lines.sort(key=lambda line: line[0][1])

    # Group lines that are within 3 pixels of each other
    current_group = [lines[0]]
    for line in lines[1:]:
        if line[0][1] - current_group[-1][0][3] <= 3:
            current_group.append(line)
        else:
            groups.append(current_group)
            current_group = [line]
    groups.append(current_group)

    # Fuse lines within each group
    for group in groups:
        x_min = min(line[0][0] for line in group)
        x_max = max(line[0][2] for line in group)
        y_mean = int(sum(line[0][1] for line in group) / len(group))
        fused_lines.append([[x_min, y_mean, x_max, y_mean]])
        extracted_lengths.append(abs(x_max - x_min))

    # Discard lines that are under 100 pixels in length
    filtered_lines = []
    filtered_lengths = []
    for line, length in zip(fused_lines, extracted_lengths):
        if length >= 100:
            filtered_lines.append(line)
            filtered_lengths.append(length)
    
    return filtered_lines, filtered_lengths

def CutAndColour(Line_coordinates, Image_names):
  # Input: Coordinates of the shortest line (calcualted in getLineLength)
  # Output: Two lists, one containing all colour data in format white (0/1)
  # and a list of the scale cut out based on coordinates
  import cv2 as cv2
  import numpy as np
  import matplotlib.pyplot as plt
  
  list_of_crops = []
  for x in range(len(Image_names)):

      if Line_coordinates[x] != 0:
        print(1)
        try:
          img_gray = cv2.cvtColor(Image_names[x], cv2.COLOR_BGR2GRAY)
        except:
          img_gray = Image_names[x]
        TempCoor = Line_coordinates[x][0] # Assign the Coordinates to a value
        ## Cut the image we give 5% in x coordinate as buffer to detect the number
        
        height, width = img_gray.shape
        buffer = int(width*0.05)
  
        ## Check if the y value still lies within the range of the image
        if int(TempCoor[1] - buffer) > 0 and int(TempCoor[1] + buffer) < height:
          
          cropped_img = img_gray[TempCoor[1]-buffer:TempCoor[1]+buffer,TempCoor[0]:TempCoor[2]]
          
        else: ## If the image is not big enough/the scale too low we pad the size of the buffer
              ## In the unlikely case we also add it on the upper part
          padded_img = np.pad(img_gray, ((buffer, buffer), (0, 0)), mode='constant', constant_values= int(img_gray[0,0]))
          cropped_img = padded_img[TempCoor[1]:TempCoor[1] + 2*buffer,TempCoor[0]:TempCoor[2]]
          
        print(2)
        # Recalculate the position of the scale which is at position buffer as buffer*2 is the new height
        ## Now we want the colour of the line
        ## So we make the image gray and extract the pixel values of of it
        
        # How about resizing the image. Thos would allow cv2.Canny
        # on small pixel numbers? 
        # Now that we do not need exact values for the scale anymore we can process the image
        #Empty = np.zeros(cropped_img)
        #img_normalized = cv2.normalize(cropped_img, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(cropped_img, (3,3), sigmaX=0, sigmaY=0) ## Blur important
        
        #Increase size for Tesseract OCR and Canny
        
        upscaled_image = cv2.resize(blurred, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)

        # Scale and number should normally be the same colour so Scale_colour = Value_colour
        
        #Can = cv2.Canny(cropped_img,50,100, L2gradient = True)
        #test = np.invert(Can)
        
        # Sobel Alternative to Canny 
        #print("Here", cropped_img.shape)
        
        #grad_x = cv2.Sobel(upscaled_image, cv2.CV_64F, 1, 0, ksize=7)
        #grad_y = cv2.Sobel(upscaled_image, cv2.CV_64F, 0, 1, ksize=7)
        
        #abs_grad_x = cv2.convertScaleAbs(grad_x)
        #abs_grad_y = cv2.convertScaleAbs(grad_y)
        
        #grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        
        # Create an empty canvas with the same shape as Can
        #WhiteCanvas = np.ones_like(Can) * 255
        
        # Find contours using RETR_TREE mode
        #contours, hierarchy = cv2.findContours(Can, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create an empty canvas with the same shape as Can
        #WhiteCanvas = np.ones_like(Can) * 255
        
        # Find the outer contours by checking the hierarchy
        #outer_contours = []
        #for i in range(len(contours)):
        #    if hierarchy[0][i][3] != -1:  # Contour has a parent (no enclosed regions)
        #        outer_contours.append(contours[i])
        
        #print(hierarchy[0], contours[0])
        
        #hierarchy = hierarchy[0]
        #colors = [75,150,225]
        #count = 0
        
        #for contour in contours:
          #cv2.fillPoly(WhiteCanvas, pts=[contour], color=0)
        
        
        #for i, contour in enumerate(contours):
        #    color = colors[count] 
        #    cv2.drawContours(WhiteCanvas, [contour], -1, color, thickness=cv2.FILLED, hierarchy=hierarchy[i])
        #    count += 1
        
        # Fill the outer contours on the canvas
        Can = cv2.Canny(blurred*255,300,400, L2gradient = False)
        #test = np.invert(Can)
        # Flood fill bakcground (white + black):
        cv2.floodFill(Can, mask=None, seedPoint=(int(0), int(0)), newVal=(255))

        cv2.floodFill(Can, mask=None, seedPoint=(int(0), int(0)), newVal=(0))
        # Find contours and hierarchy
        contours, hierarchy = cv2.findContours(Can, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create an empty canvas with the same shape as Can
        WhiteCanvas = np.ones_like(Can) * 255
        
        # Find the outer contours by checking the hierarchy
        outer_contours = []
        for i in range(len(contours)):
            if hierarchy[0][i][3] != -1:  # Contour has a parent (no enclosed regions)
                outer_contours.append(contours[i])
        
        # Fill the outer contours on the canvas
        Filled = cv2.drawContours(WhiteCanvas, outer_contours, contourIdx=-1, color=(0), thickness=cv2.FILLED)
        ### As spaces in letters are filled in we now want the recreate these spaces
        #contours_fill, hierarchy = cv2.findContours(Can, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        plt.clf()
        plt.imshow(Filled, cmap = "gray")
        plt.show()
        
        list_of_crops.append(cropped_img)
      else: 
        print(4)
        list_of_crops.append(Image_names[x])

  return(list_of_crops)


#### No we need to get the Number above the line
# def getUnit(cropped_Image) ### For py Tesseract we need sudo install
### white only needs mask
### We need to check if the scale is white
### Therefore we check if we have more 0 pixels in the mask than 255

def get_Scale(path_to_preprocessed_images, psm_mode=7):
  import pytesseract as pytesseract
  import matplotlib.pyplot as plt
  import cv2 as cv2
  
  ScaleUnit = []
  pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
  for x in range(len(path_to_preprocessed_images)):
    
    img = cv2.imread(path_to_preprocessed_images[x])
    #try:
      #gaussian_3 = cv2.GaussianBlur(crop_img, (0, 0), 2.0)
      #unsharp_image = cv2.addWeighted(crop_img, 2.0, gaussian_3, -1.0, 0)
    #except:
      #gaussian_3 = crop_img

    ## tesseract detect
    
    tesseract_config = r'--oem 3 --psm ' + str(psm_mode) +" -c tessedit_char_whitelist=0123456789."
  
    ## as we dont no which kind of scale was put in we use all 3 (inverse, contrast for white and MasCorrNormBare) and then compare.
    ## We could find out the colour of the scale when we use the lines, but is it necessary?
  
    number = pytesseract.image_to_string(img, config= tesseract_config).strip()
    ScaleUnit.append(number)
  return(ScaleUnit)

########### We also should detect mm or µm this can be achieved with pytesseract. but we also can guess as 
########### high values should always be µm and low ones mm ## psm 7 is good by far not perfect... Run multiple modes?

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
### The new approach is split differently to increase the accuracy
# 1. Make image gray
# 2. Detect the line, find color and length of it
# 3. Cut out sourrinding area 
# 3a. Optionally delete only the line
# 4. Normalize DPI
# 5. Binarise, Invert and Threshold the image based
# 6. Detect with Tesseract
# 7. Save in Df/Aig Cellulose 5000 84.jpg

a,b = Images_list("/home/philipp/JPG")
#Image_repository/SimonaAig21d
##/JPG
c,d,q,z = getLineLength(a)
l = CutAndColour(d,q) 
k = NormalizeDPI(l,a)
h = get_Scale(k,7) ## Thats right now the now transformation mode
z[19]
len(q)
len(c)
len(d)
### Two unlikely values: 1 due to missing scale in image and one due to false length
from collections import Counter
Counter(h)
Counter(sorted(c))
import cv2
import numpy as np
sorted(c)
set(c)
import cv2
import numpy as np
import matplotlib.pyplot as plt 
for i in q:
  if sum(i) != 0:
    print(i)
    print(i.shape)
  
len(q)
len(d)  
def CutAndColour_old(Line_coordinates, Image_names):
  # Input: Coordinates of the shortest line (calcualted in getLineLength)
  # Output: Two lists, one containing all colour data in format white (0/1)
  # and a list of the scale cut out based on coordinates
  import cv2 as cv2
  import numpy as np
  import matplotlib.pyplot as plt
  
  list_of_crops = []
  for x in range(len(Image_names)):
      img = cv2.imread(Image_names[x])
      img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      print(1)
      TempCoor = Line_coordinates[x] # Assign the Coordinates to a value
      print(2, TempCoor)
      if TempCoor == 0:
        list_of_crops.append(img_gray)
      
      else:
        ## Cut the image we give 5% in x coordinate as buffer to detect the number
        
        height, width = img_gray.shape
        buffer = int(width*0.05)
  
        ## Check if the y value still lies within the range of the image
        if int(TempCoor[0][1] - buffer) > 0 and int(TempCoor[0][1] + buffer) < height:
          
          cropped_img = img_gray[TempCoor[0][1]-buffer:TempCoor[0][1]+buffer,TempCoor[0][0]:TempCoor[0][2]]
          
        else: ## If the image is not big enough/the scale too low we pad the size of the buffer
              ## In the unlikely case we also add it on the upper part
          padded_img = np.pad(img_gray, ((buffer, buffer), (0, 0)), mode='constant', constant_values= int(np.average(img_gray)))
          cropped_img = padded_img[TempCoor[0][1]:TempCoor[0][1] + 2*buffer,TempCoor[0][0]:TempCoor[0][2]]
          
        
        # Recalculate the position of the scale which is at position buffer as buffer*2 is the new height
        ## Now we want the colour of the line
        ## So we make the image gray and extract the pixel values of of it
        
        # How about resizing the image. Thos would allow cv2.Canny
        # on small pixel numbers? 
        # Now that we do not need exact values for the scale anymore we can process the image
        #Empty = np.zeros(cropped_img)
        img_normalized = cv2.normalize(cropped_img, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(img_normalized, (3,3), sigmaX=0, sigmaY=0) ## Blur important
        upscaled_image = cv2.resize(blurred, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        # Scale and number should normally be the same colour so Scale_colour = Value_colour
        #350,400
        Can = cv2.Canny(upscaled_image,0,1, L2gradient = True)
        #test = np.invert(Can)
        print(3)
        # Find contours and hierarchy
        contours, hierarchy = cv2.findContours(Can, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create an empty canvas with the same shape as Can
        WhiteCanvas = np.ones_like(Can) * 255
        
        # Find the outer contours by checking the hierarchy
        outer_contours = []
        for i in range(len(contours)):
            if hierarchy[0][i][3] != -1:  # Contour has a parent (no enclosed regions)
                outer_contours.append(contours[i])
        
        # Fill the outer contours on the canvas
        Filled = cv2.drawContours(WhiteCanvas, outer_contours, contourIdx=-1, color=(0), thickness=cv2.FILLED)
        ### As spaces in letters are filled in we now want the recreate these spaces
        #contours_fill, hierarchy = cv2.findContours(Can, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(4)
        plt.clf()
        plt.imshow(Filled, cmap = "gray")
        plt.show()
        
        list_of_crops.append(Filled)
        
  return(list_of_crops)
l = CutAndColour_old(d,a)


dasd = CutAndColour_grays(d,a)
k = NormalizeDPI(dasd,a)
h = get_Scale(k,7) 

### MEthods not really used
def NormalizeDPI(images, image_names):
  # Resize Cutouts of the scale and enforce an uniform DPI scale ##
  # This helps tesseract ocr-to detect scales more confidently  ###
  # according to documentation
  # Input: Folder_path
  # Output: Saved Scale Images 512x512, 600 DPI at the folder path called
  # Converted Scales as Image_name_ConvScale.jpg
  
  import os
  from PIL import Image
  import re
  import matplotlib.pyplot as plt
  
  list_of_standardized_images = []
  directory = "Converted_Scales"
  save_path =  os.path.dirname(image_names[0])
  convert_path = os.path.join(save_path,directory)
  
  if os.path.exists(convert_path) == False:
    os.mkdir(convert_path)
  i = 1
  
  for image in range(len(images)):

    result = str(image_names[image]).split('/')[-1]
    
    try:
        new_file = convert_path + "/" + result[:-4] + "_ConvScale.jpg"
        
        i = i+1
        im = Image.fromarray(images[image])
        im.thumbnail((512,512))
        im.save(new_file,"JPEG",dpi=(600,600))
        list_of_standardized_images.append(new_file)
    except:
        continue
  return(list_of_standardized_images)

def Split(Image_list):
  ## Function that splits images
  ## to reduce noise in tesseract
  ## as we want to avoid canny detection
  ## Input lis of precut iamges 
  ## Output List of lists with  further cut images
  
  list_of_image_cuts = []
  for x in Image_list:
      
    Idx = Get_Split_Indices(x)
    if len(Idx) != 0:
      Seq = Split_Into_sequences(Idx)
      list_of_image_cuts.append(Split_Images(Seq,x))
      
    else: 
      list_of_image_cuts.append(x)
  return list_of_image_cuts

def Get_Split_Indices(Image):
  # Initialize variables
  # Input Image
  # Find index that are non max or 0
  # and return them
  
  segment_indices = []
  current_index = []
  sum_rows = np.sum(Image, axis = 1)

  # Iterate through the data
  for i, value in enumerate(sum_rows):
      if value == 0 or value == (255*Image.shape[1]):
          if current_index != -1:
              segment_indices.append(current_index)
          current_index = i + 1
  
  # Add the last segment index if it exists
  if current_index != -1:
      segment_indices.append(current_index)
  segment_indices = segment_indices[1:]
  return segment_indices

def Split_Into_sequences(lst):
    ## Build a sequence out of 
    ## the Idx list
    ## Input Indexes
    ## Output Index as (min,max) list
    
    sequences = []

    current_sequence = [lst[0]]
    
    for i in range(1, len(lst)):
        if lst[i] == current_sequence[-1] + 1:
            current_sequence.append(lst[i])
        else:
            sequences.append(current_sequence)
            current_sequence = [lst[i]]

    sequences.append(current_sequence)
    return sequences

def Split_Images(seq, Image):
  ### We now split the cutout int othe segments how are not unifrom e.g. contain our
  ### Information if all went well
  ### Input min max indexes
  ### Returns images split
  
  min_max_values = [(min(seq), max(seq)) for seq in list_of_splits]

  split_images = []
  for x in range(1,len(min_max_values)):
    
    split_images.append(Image[min_max_values[x-1][1]:min_max_values[x][0],:])
    
  return split_images
  
def CutAndColour_grays(Line_coordinates, Image_names):
  # Input: Coordinates of the shortest line (calcualted in getLineLength)
  # Output: Two lists, one containing all colour data in format white (0/1)
  # and a list of the scale cut out based on coordinates
  import cv2 as cv2
  import numpy as np
  import matplotlib.pyplot as plt
  
  list_of_crops = []
  for x in range(len(Image_names)):
      img = cv2.imread(Image_names[x])
      img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      print(1)
      TempCoor = Line_coordinates[x] # Assign the Coordinates to a value
      print(2, TempCoor)
      if TempCoor == 0:
        list_of_crops.append(img_gray)
      
      else:
        ## Cut the image we give 5% in x coordinate as buffer to detect the number
        
        height, width = img_gray.shape
        buffer = int(width*0.05)
  
        ## Check if the y value still lies within the range of the image
        if int(TempCoor[0][1] - buffer) > 0 and int(TempCoor[0][1] + buffer) < height:
          
          cropped_img = img_gray[TempCoor[0][1]-buffer:TempCoor[0][1]+buffer,TempCoor[0][0]:TempCoor[0][2]]
          
        else: ## If the image is not big enough/the scale too low we pad the size of the buffer
              ## In the unlikely case we also add it on the upper part
          padded_img = np.pad(img_gray, ((buffer, buffer), (0, 0)), mode='constant', constant_values= int(np.average(img_gray)))
          cropped_img = padded_img[TempCoor[0][1]:TempCoor[0][1] + 2*buffer,TempCoor[0][0]:TempCoor[0][2]]
          
        
        # Recalculate the position of the scale which is at position buffer as buffer*2 is the new height
        ## Now we want the colour of the line
        ## So we make the image gray and extract the pixel values of of it
        
        # How about resizing the image. Thos would allow cv2.Canny
        # on small pixel numbers? 
        # Now that we do not need exact values for the scale anymore we can process the image
        #Empty = np.zeros(cropped_img)
        img_normalized = cv2.normalize(cropped_img, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(img_normalized, (3,3), sigmaX=0, sigmaY=0) ## Blur important
        upscaled_image = cv2.resize(blurred, (600,600),interpolation=cv2.INTER_LANCZOS4) # INTER_CUBIC INTER_LANCZOS4
        # Scale and number should normally be the same colour so Scale_colour = Value_colour
        #350,400
        
        # Threshold the grayscale image
        _, thresholded1 = cv2.threshold(upscaled_image, thresh= int(np.average(upscaled_image)), maxval=255, type=cv2.THRESH_BINARY_INV)
        
        # Other way round
        _, thresholded2 = cv2.threshold(upscaled_image, thresh= int(np.average(upscaled_image)), maxval=255, type=cv2.THRESH_BINARY)
        # Set white pixels to black and non-white pixels to zero
        result = cv2.bitwise_and(upscaled_image, upscaled_image, mask=thresholded1)
        result = cv2.bitwise_and(upscaled_image, upscaled_image, mask=thresholded2)
        
        
        
        
        Can = cv2.Canny(upscaled_image,150,200, L2gradient = True)
        
        # Perform morphological operations to refine the edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(Can, kernel, iterations=2)
        #eroded = cv2.erode(dilated, kernel, iterations=1)
        
        #test = np.invert(Can)
        print(3)
        # Find contours and hierarchy
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Create a blank canvas with the same shape as the grayscale image
        canvas = np.ones_like(Can) * 255
        
        # Find the outer contours by checking the hierarchy
        outer_contours = []
        for i in range(len(contours)):
            if hierarchy[0][i][3] == -1:  # Contour has no parent (outer contour)
                outer_contours.append(contours[i])
        
        # Fill each level of the contour hierarchy with a different shade of gray
        for level, contour in enumerate(outer_contours):
            # Generate a random shade of gray for each level
            black_level = int(255 - (level + 1) * (255 / len(outer_contours)))
            # Fill the contour with the corresponding shade of black
            cv2.drawContours(canvas, [contour], contourIdx=-1, color=black_level, thickness=cv2.FILLED)

            ### As spaces in letters are filled in we now want the recreate these spaces
            #contours_fill, hierarchy = cv2.findContours(Can, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(4)
        plt.clf()
        plt.imshow(canvas, cmap = "gray")
        plt.show()
        
        list_of_crops.append(thresholded2)
        
  return(list_of_crops)

def CutAndColour(Line_coordinates, Image_names):
  # Input: Coordinates of the shortest line (calcualted in getLineLength)
  # Output: Two lists, one containing all colour data in format white (0/1)
  # and a list of the scale cut out based on coordinates
  import cv2 as cv2
  import numpy as np
  import matplotlib.pyplot as plt
  
  list_of_crops = []
  for x in range(len(Image_names)):
      img = cv2.imread(Image_names[x])
    
      img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      
      TempCoor = Line_coordinates[x] # Assign the Coordinates to a value
      
      if TempCoor == 0:
        list_of_crops.append(img_gray)
      
      else:
        ## Cut the image we give 5% in x coordinate as buffer to detect the number
        
        height, width = img_gray.shape
        buffer = int(width*0.05)
  
        ## Check if the y value still lies within the range of the image
        if int(TempCoor[1] - buffer) > 0 and int(TempCoor[1] + buffer) < height:
          
          cropped_img = img_gray[TempCoor[1]-buffer:TempCoor[1]+buffer,TempCoor[0]:TempCoor[2]]
          
        else: ## If the image is not big enough/the scale too low we pad the size of the buffer
              ## In the unlikely case we also add it on the upper part
          padded_img = np.pad(img_gray, ((buffer, buffer), (0, 0)), mode='constant', constant_values= int(np.average(img_gray)))
          cropped_img = padded_img[TempCoor[1]:TempCoor[1] + 2*buffer,TempCoor[0]:TempCoor[2]]
          
        
        # Recalculate the position of the scale which is at position buffer as buffer*2 is the new height
        ## Now we want the colour of the line
        ## So we make the image gray and extract the pixel values of of it
        
        # How about resizing the image. Thos would allow cv2.Canny
        # on small pixel numbers? 
        # Now that we do not need exact values for the scale anymore we can process the image
        #Empty = np.zeros(cropped_img)
        img_normalized = cv2.normalize(cropped_img, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(img_normalized, (3,3), sigmaX=0, sigmaY=0) ## Blur important
        # Scale and number should normally be the same colour so Scale_colour = Value_colour
        
        Can = cv2.Canny(blurred,350,400, L2gradient = True)
        #test = np.invert(Can)
        
        # Find contours and hierarchy
        contours, hierarchy = cv2.findContours(Can, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create an empty canvas with the same shape as Can
        WhiteCanvas = np.ones_like(Can) * 255
        
        # Find the outer contours by checking the hierarchy
        outer_contours = []
        for i in range(len(contours)):
            if hierarchy[0][i][3] != -1:  # Contour has a parent (no enclosed regions)
                outer_contours.append(contours[i])
        
        # Fill the outer contours on the canvas
        Filled = cv2.drawContours(WhiteCanvas, outer_contours, contourIdx=-1, color=(0), thickness=cv2.FILLED)
        ### As spaces in letters are filled in we now want the recreate these spaces
        #contours_fill, hierarchy = cv2.findContours(Can, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        
        plt.clf()
        plt.imshow(Can, cmap = "gray")
        plt.show()
        
        list_of_crops.append(Filled)
        
  return(list_of_crops)


###  Test 
List_of_all_detections = []
for y in range(len(Splitted)):
  Temp_detect = []
  if len(Splitted[y]) < 30:
    for x in Splitted[y]:
    
      import pytesseract as pytesseract
      import matplotlib.pyplot as plt
      import cv2 as cv2
      
      ScaleUnit = []
      pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
      img = x
      mean_value = gray.mean()
      
      
      # Check if the mean value is closer to 0 or 255
      if mean_value < 128:
        img = abs(img-255)
  
      ### Check if its white on black or black on white
  
      tesseract_config = r'--oem 3 --psm ' + str(7) +" -c tessedit_char_whitelist=0123456789."
    
      ## as we dont no which kind of scale was put in we use all 3 (inverse, contrast for white and MasCorrNormBare) and then compare.
      ## We could find out the colour of the scale when we use the lines, but is it necessary?
    
      number = pytesseract.image_to_string(img, config= tesseract_config).strip()
      Temp_detect.append(number)
      print(f"I spy with m little eye a {number}. In run number {y}")
  else:
    print("Image not splitted skippin")
    List_of_all_detections.append(0) 
  List_of_all_detections.append(Temp_detect) 
    
    len(Splitted[12])

for x in l:
  plt.clf()
  plt.imshow(q[-1], cmap = "gray")
  plt.show()

number = pytesseract.image_to_string(cv2.GaussianBlur(l[-1],(1,1),0), config= tesseract_config).strip()
List_of_all_detections
### Wrong cut why?



threshold_values = [0,10,20,30,235,245,255]

# Apply thresholding for each threshold value
thresholded_images = []
for threshold in threshold_values:
    _, thresholded = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    thresholded_images.append(thresholded)

# Show the thresholded images
for i, thresholded in enumerate(thresholded_images):
    plt.clf()
    plt.imshow(l[6])
    plt.show()

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
grayimg = cv2.cvtColor(cv2.imread(a[32]), cv2.COLOR_BGR2GRAY)
cl1 = clahe.apply(grayimg)
cl1.shape
np.max(cl1)
cl1[cl1 < 250] = 0
plt.clf()
plt.imshow(cl1, cmap = "gray")
plt.show()


###### Evaluate whats best

#Thresholds -> No
# Canny good if with big numbers
# White/black detect
#
