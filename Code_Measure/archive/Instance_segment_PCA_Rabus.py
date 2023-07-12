##### Body width #####
###################### OLD DONT USE 19/05/23

### Based on segment output of daphnid_instances_0.1
### Measure a defined point: Halfway between eye and spina base and not only the longest
### Read in output from Object detection

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

def getOrientation(pts, img, visualize=False):
  
  import cv2 as cv
  from math import atan2, cos, sin, sqrt, pi
  import numpy as np
  ## [pca]
  # Construct a buffer used by the pca analysis
  sz = len(pts)
  data_pts = np.empty((sz, 2), dtype=np.float64)
  for i in range(data_pts.shape[0]):
    data_pts[i,0] = pts[i,0,0]
    data_pts[i,1] = pts[i,0,1]
 
  # Perform PCA analysis
  mean = np.empty((0))
  mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
  
  # Store the center of the object
  cntr = (int(mean[0,0]), int(mean[0,1]))
  ## [pca]
 
  angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
  #print(angle, cntr, mean, "Vectors:",eigenvectors)
  
  if visualize == True:
    ###### Show what happens
    ## [visualization]
    # Draw the principal components
    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (255, 255, 0), 1)
    drawAxis(img, cntr, p2, (0, 0, 255), 5)
   
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    ## [visualization]
   
    # Label with the rotation angle
    label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
    textbox = cv.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
    cv.putText(img, label, (cntr[0], cntr[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
  return angle

def Image_Rotation(masks, angles, visualize=False):
  
  import matplotlib.pyplot as plt
  import cv2 as cv
  import numpy as np
  import imutils
  # Preprocess and show the image
  # Calculate the rotation
  # Rotate the image 
  angles = []
  rotated_images = []
  
  for x in range(len(masks)):
    # Load the image
    # Convert image to grayscale
    #gray = cv.cvtColor(masks[x], cv.COLOR_BGR2GRAY)
    # Find all the contours in the thresholded image
    contours, _ = cv.findContours(masks[x], cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    
    for i, c in enumerate(contours):
      
      # Calculate the area of each contour
      area = cv.contourArea(c)
      
      # Ignore contours that are too small or too large
      if area < 3700:
        continue
      
      # Draw each contour only for visualisation purposes
      cv.drawContours(masks[x], contours, i, (0, 0, 255), 2)
      # Find the orientation of each shape
      temp_angles = getOrientation(c, masks[x])
      angle_deg = -int(np.rad2deg(temp_angles)) - 90
      angles.append(angle_deg)
      rotated = imutils.rotate_bound(masks[x], angle_deg)
      
      
      if visualize == True:
        ### Plot the original
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.text(0.5, 0.5, angle_deg)
        plt.title(names + " original")
        
        ## Plot the rotation 
        
        plt.subplot(1, 2, 2)
        plt.imshow(rotated)
        plt.title(names + " rotated")
        plt.show()
        plt.imsave("/home/philipp/Output_instance_segment/Marvin_val/" + names + f"_{angle_deg}.jpg",arr = rotated, dpi = 600)
    rotated_images.append(rotated)
    
  return(angles, rotated_images)

def drawAxis(img, p_, q_, color, scale):
  
  from math import atan2, cos, sin, sqrt, pi
  import cv2 as cv
  p = list(p_)
  q = list(q_)
 
  ## [visualization1]
  angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
  hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
 
  # Here we lengthen the arrow by a factor of scale
  q[0] = p[0] - scale * hypotenuse * cos(angle)
  q[1] = p[1] - scale * hypotenuse * sin(angle)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
 
  # create the arrow hooks
  p[0] = q[0] + 9 * cos(angle + pi / 4)
  p[1] = q[1] + 9 * sin(angle + pi / 4)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
 
  p[0] = q[0] + 9 * cos(angle - pi / 4)
  p[1] = q[1] + 9 * sin(angle - pi / 4)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
  ## [visualization1]

def Create_Mask(anno_save_loc, parent_dir):
  import json
  import cv2 as cv
  import numpy as np
  import pandas as pd
  # Load the COCO formatted JSON file
  with open(anno_save_loc, 'r') as f:
      coco_data = json.load(f)
  
  # We have to match the image_paths and the image id's beforehand
  image_id_to_file_name = {}
  for image in coco_data['images']:
    image_id_to_file_name[image['id']] = image['file_name']
  
  Order_of_file_names = []
  # Loop through the annotations and print the file names
  for annotation in coco_data['annotations']:
    image_id = annotation['image_id']
    file_name = image_id_to_file_name[image_id]
    Order_of_file_names.append(file_name)
  
  # Loop through the annotations and extract the polygon data
  list_of_polygons = []
  for annotation in coco_data['annotations']:
          segmentation = annotation['segmentation']
          
          ### segmentation contains more than one value. Which one is correct?
          ### Is Refinement already applied?
          
          longest_polygon = max(segmentation, key=len)
          polygon = np.array(longest_polygon, np.int32).reshape((-1, 2))
          # Polygon is a list of (x,y) coordinates
          list_of_polygons.append(polygon)
          
  List_of_Images = []      

  for x in range(len(Order_of_file_names)):
      
    # Load the image
    
    image = cv.imread(parent_dir + Order_of_file_names[x])
    	
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Create a mask image with the same size as the input image
    mask = np.zeros(gray.shape, dtype=np.uint8)
    
    # Loop through the annotations and draw the polygons onto the mask
    
    cv.fillPoly(mask, [list_of_polygons[x]], (255, 255, 255))
    
    List_of_Images.append(mask)
    
    # Display the mask image
    #import matplotlib.pyplot as plt
    #plt.clf()
    #plt.subplot(1,2,1)
    #plt.imshow(mask, cmap = "gray")
    #plt.subplot(1, 2, 2)
    #plt.imshow(image, cmap = "gray")
    #plt.show()
    
  return Order_of_file_names, List_of_Images

def point_trans(ori_point, angle, ori_shape, new_shape):
    
    # Transfrom the point from original to rotated image.
    # Args:
    #    ori_point: Point coordinates in original image.
    #    angle: Rotate angle in radians.
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

def Detect_Midpoint(Annotations_object_detect, angles, images_list, rot_masks, org_mask):
  
  # Turn img upright based on the eye position.
  # Args:
  #    AnnotationFrame: Point coordinates of object detection on original image
  #    angles: List of the turning angles in degrees
  #    rot_masks_: Rotated Image masks craeted by Image_Rotation
  #    images_list: names of images in annotation order
  #    org_mask: Original mask created by CreateMask
  # Returns:
  #    Halfway point between eye and spina centre for the cropped images
  
  import pandas as pd
  import numpy as np
  import cv2 as cv2
  import matplotlib.pyplot as plt


  data = Annotations_object_detect.copy()

  list_of_Midpoints = []
  for item in range(len(images_list)):
    
    img = rot_masks[item]
    Row_of_Image = data.loc[data['Name'] == images_list[item]] ### get the data for the image
  
    #### Now we want to rotate the image and y coordinates 180 degrees in both diretions
    #### if y gets higher by rotation we want interrupt the while loop and rotate into
    #### the other direction
  
    try:
      CoorEye = int(Row_of_Image["Center_X_Eye"].iloc[0]),int(Row_of_Image["Center_Y_Eye"].iloc[0])
      
      CoorSb = int(Row_of_Image["Center_X_Sb"].iloc[0]),int(Row_of_Image["Center_Y_Sb"].iloc[0])
    
    
      ### Now find the new coordinates ## Points (X,Y). Shape (Height (Y), Width (X))
    
      Eye_trans = point_trans(CoorEye, np.deg2rad(angles[item]), org_mask[item].shape, img.shape)
      Sb_trans = point_trans(CoorSb, np.deg2rad(angles[item]), org_mask[item].shape, img.shape)
      MidX = (Eye_trans[0] + Sb_trans[0])/2
      MidY = (Eye_trans[1] + Sb_trans[1])/2
    
      #### These coordinates are not correct as the masks are cropped.
      #### To get the cropping positions of every images we calculate the
      #### offset between the segmentation coordinates of annotations.josn
      #### and annotations_cropped.josn and add it to the coordinates MidY/MidX
    
      Midpoint = (MidX,MidY)
      list_of_Midpoints.append(Midpoint)
      
    except:
      Midpoint = 0
      list_of_Midpoints.append(Midpoint)

  return list_of_Midpoints

def Measure_Width(Rotated_Images, ListofMidpoints):
  ## Now calculate the width at the midpoint variable
  import numpy as np
  Widths = []
  
  for item in range(len(Rotated_Images)):
    try:
      MidRow = ListofMidpoints[item][1] ## Y coordinate for image of midpoint
      Width = np.sum(Rotated_Images[item][:,int(MidRow)]) /255
      Widths.append(Width)
    except:
      Widths.append(np.nan)
  
  return Widths

### Non -c gives us no masks but we want the masks, Use the polygons directly?
### How do we proceed?
### We can calculate the crop factor of every image with comparing annotations.json

# Step 1: Make full image Masks # The resulting values are sorted according to the annotations

Image_sort, Mask = Create_Mask("/home/philipp/Output_instance_segment/Marvin_val/annotations.json", "/home/philipp/Image_repository/Marvin_fish_bile/Images/")

# Step 2 Rotate the mask using eigenvectors

Rotation_angles, Rotated_masks = Image_Rotation(Mask, Image_sort) 

import pandas as pd
Eye_Spina_df  = pd.read_csv("/home/philipp/Master_thesis_data/Val_data_MA_output/Master_file_output_Marvin/Unique_Width/Completed_data.csv",decimal = ".")

### Calculate the middle of the eye spin base axis as most Evaluation in papers do
### Right now we have the approach by Rabus & Laforsch
### To keep the rotation correct we have to make sure to keep the images sorted how they are in the annotations

Midpoints = Detect_Midpoint(Eye_Spina_df,Rotation_angles,Image_sort,Rotated_masks,Mask)

Body_width = Measure_Width(Rotated_masks,Midpoints)



