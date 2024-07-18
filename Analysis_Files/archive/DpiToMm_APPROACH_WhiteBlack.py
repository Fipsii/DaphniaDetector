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
      
      
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      #thresh, gray_thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
      #thresh = cv2.threshold()

      ## why should we blur? Image lines should be sharp -> thresholding should be the most effective
      ## In thresholding we need to consider colors but should be managable if we make an OTSU and 
      ## hard cut off for white backgrounds with white scales
      kernel_size = (5,1) # We do not wnt to blur the x axis as we need the length information 
      blur_gray = cv2.GaussianBlur(gray,(kernel_size),0)
    
      low_threshold = 200
      high_threshold = 300
      edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
      
      #### Set HoughLinesP Parameters
      
      rho = 1 # distance resolution in pixels of the Hough grid
      theta = np.pi / 180  # angular resolution in radians of the Hough grid
      threshold = 10  # minimum number of votes (intersections in Hough grid cell)
      min_line_length = 100 # minimum number of pixels making up a line
      max_line_gap = 0  # maximum gap in pixels between connectable line segments
      line_image = np.copy(img) * 0  # creating a blank to draw lines on
      # Run Hough on edge detected image
      # Output "lines" is an array containing endpoints of detected line segments
      
      lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)
      
      if lines is None: ## If no lines are found
        list_of_lengths.append(0) 
        line_Coors.append(0)
      else: # if there are lines
        print(lines)
        for line in lines:
          x1, y1, x2, y2 = line[0]
          cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        combined, Coordinates = group_lines(lines) ## combine close lines
        
        plt.clf()
        plt.imshow(line_image, cmap = "gray")
        plt.show()
        
        # Find the index of the min value so we can sort the line coordinates
        Idx = combined.index(min(combined))
        Correct_Coor = Coordinates[0][Idx] ## Get the coordinates of the shortest line
        PixelPerUnit = min(combined) ## take the min value of length as scale
        
        # Append values to lists
        line_Coors.append(Correct_Coor)
        list_of_lengths.append(PixelPerUnit) 
        list_of_images.append(line_image)
        list_of_lines.append(lines)
  return list_of_lengths, line_Coors, list_of_images, list_of_lines ## Works right now with 5% Error reimplement giving coordinates

def group_lines(lines):
    # Combines lines and outputs the new coordiantes
    # Lines are grouped by y and then added to each other
    # with a new mean y 
    # Input: Line coordinates [[[...],[...],...]]
    # Output: Line coordinates of combined lines [[[...],[...],...]]
    # Length of the combined lines
    import numpy as np
    
    groups = []
    
    # Convert lines to a list if it is a NumPy array
    if isinstance(lines, np.ndarray):
        lines = lines.tolist()
        lines_flat = [line[0] for line in lines]
        
    # Sort lines based on y-coordinate
    lines_flat.sort(key=lambda line: line[1])
    
    current_group = [lines_flat[0]]
    
    ### group them according if their difference is smaller than 5 pixels
    for line in lines_flat[1:]:
        
        if line[1] - current_group[-1][3] <= 3:
            current_group.append(line)
        
        else:
            groups.append(current_group)
            current_group = [line]
    
    groups.append(current_group)
    ### We also want to build the lines and output them too
    
    fused_lines = [] ## This is gonna be the list containing the lists of lines
    temp_lines = [] # These are the lines per image
    extracted_length = [] # This gonna be the list containing the lists of lenghts
    
    for group in groups:
        
        xmin = min(line[0] for line in group)
        xmax = max(line[2] for line in group)
        ymean = int(sum(line[1] for line in group)/len(group))
        extracted_length.append(abs(xmax - xmin)) # Calculate length
        temp_lines.append([xmin, ymean, xmax, ymean]) ## Coordinates of our line
    
    fused_lines.append(temp_lines)
      
    return extracted_length, fused_lines

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

a,b = Images_list("/home/philipp/JPG/")

c,d,q,z = getLineLength(a)
l= CutAndColour(d,a) ### Here we need another approach if we want to use new scales IDK iterate over 5 
k = NormalizeDPI(l,a)
h = get_Scale(k,7)

### Two unlikely values: 1 due to missing scale in image and one due to false length
from collections import Counter
Counter(c)
import cv2
import numpy as np
original_image = cv2.imread(a[-6])
plt.imshow(original_image)
plt.show()
# Load the original RGB image
original_image = original_image.astype(np.float32)
kernel_size = (5,1) # We do not wnt to blur the x axis as we need the length information 
blur_gray = cv2.GaussianBlur(original_image,(kernel_size),0)
# Extract the R, G, B channels ## Weird the image does show True values an the mask too
b, g, r = cv2.split(original_image)

#b =np.around(b/5, decimals=0)*5
#g =np.around(g/5, decimals=0)*5
#r =np.around(r/5, decimals=0)*5
# Create a mask for pixels where the absolute differences are within the tolerance

# Create a mask for pixels where the absolute differences are within the tolerance
red_mask = np.logical_or(original_image[:, :, 0] > 150, original_image[:, :, 0] < 100)
green_mask = np.logical_or(original_image[:, :, 1] > 150, original_image[:, :, 1] < 100)
blue_mask = np.logical_or(original_image[:, :, 2] > 150, original_image[:, :, 2] < 100)
mask = np.logical_and.reduce((red_mask, green_mask, blue_mask))

len(np.where(mask == True)[1])
# Create a new grayscale image with the same size as the original image
gray_image = np.zeros_like(r, dtype=np.uint8) + 255

# Set the pixels in the grayscale image where the mask is True to black (0)
gray_image[mask] = 0

# Set the pixels in the grayscale image where the mask is False to white (255)
gray_image[np.logical_not(mask)] = 255


plt.clf()
plt.imshow(gray_image, cmap = "gray")
plt.show()

plt.imsave("Waddup.jpg",arr = gray_image, dpi = 600, cmap = "gray")
a[1]

f
import cv2 as cv2
import matplotlib.pyplot as plt
img_normalized = cv2.normalize(l[2], None, 0, 255, cv2.NORM_MINMAX)
blurred = cv2.GaussianBlur(img_normalized, (3,3), sigmaX=0, sigmaY=0) ## Blur important
# Scale and number should normally be the same colour so Scale_colour = Value_colour
for x in range(50,500,50):        
  Can = cv2.Canny(blurred,0,x, L2gradient = True)
  plt.clf()
  plt.imshow(Can, cmap = "gray")
  plt.show()
  
h[1]

l
