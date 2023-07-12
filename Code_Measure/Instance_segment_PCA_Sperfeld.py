##### Body width after Sperfeld et al 2020#####
######################

### Based on segment output of daphnid_instances_0.1
### Measure a defined point: Halfway between eye and spina base and not only the longest
### Read in output from Object detection
### Needs functions in Instance_sgment Imhof, and is used in Body_width_chooser

def PerpendicularLine_Eye_Sb(Annotations_object_detect, images_list,org_mask, visualize=False):
  
  # Turn img upright based on the eye position.
  # Args:
  #    AnnotationFrame: Point coordinates of object detection on original image
  #    angles: List of the turning angles in degrees
  #    rot_masks_: Rotated Image masks
  #    images_list: names of images in annotation order
  # Returns:
  #    Halfway point between eye and spina centre for the cropped images
  #    Images turned so that the the axis that is perpendicular to the body axis
  #    is straight
  
  import pandas as pd
  import numpy as np
  import cv2 as cv2
  import matplotlib.pyplot as plt

  data = Annotations_object_detect.copy()
  Perp_to_Image = []
  list_of_Midpoints = []
  Rot_Angles = []
  for item in range(len(images_list)):
    
    img = org_mask[item]
    Row_of_Image = data.loc[data['image_id'] == images_list[item]] ### get the data for the image
  
    #### Now we want to rotate the image and y coordinates 180 degrees in both diretions
    #### if y gets higher by rotation we want interrupt the while loop and rotate into
    #### the other direction
  
    try:
      CoorEye = int(Row_of_Image["Center_X_Eye"].iloc[0]),int(Row_of_Image["Center_Y_Eye"].iloc[0])
      
      CoorSb = int(Row_of_Image["Center_X_Sb"].iloc[0]),int(Row_of_Image["Center_Y_Sb"].iloc[0])
      
      
      ### Now find the new coordinates ## Points (X,Y). Shape (Height (Y), Width (X))
      
      MidX = (CoorEye[0] + CoorSb[0])/2
      MidY = (CoorEye[1] + CoorSb[1])/2
  
      #plt.savefig("/home/philipp/Data_New_Workflow/Images_Body_axis_Sperfeld/" + images_list[item] +".jpg")
  
      #### These coordinates are not correct as the masks are cropped.
      #### To get the cropping positions of every images we calculate the
      #### offset between the segmentation coordinates of annotations.josn
      #### and annotations_cropped.josn and add it to the coordinates MidY/MidX
    
      Midpoint = (MidX,MidY)
  
      
      # Now we create a line perpendicular through the midpoint like in Sperfeld et al. 2020
      
      slope = (CoorSb[1] - CoorEye[1]) / (CoorSb[0] - CoorEye[0])
      perpendicular_slope = -1 / slope
      
      line_length = min(img.shape[1], img.shape[0]) 
      
      x_end = Midpoint[0] + line_length / (2 * np.sqrt(1 + perpendicular_slope**2))
      y_end = Midpoint[1] + perpendicular_slope * (x_end - Midpoint[0])
      
      x_range = np.linspace(0, img.shape[1], 100)
      y_vals = perpendicular_slope * (x_range - Midpoint[0]) + Midpoint[1]
      
      ### No we successfully have an orthogonal line to the body axis
      ### Defined by and spina base. For easy measurment we turn the nwe line so 
      ### it is orthogonal to the image extract the numpy row and sum()/256 it
      
      angle = np.arctan(slope) 
      
      ### TO DO Constructions ##
      import imutils
      
      rotated_image = imutils.rotate_bound(img, 270 - np.rad2deg(angle))
      
      ### Translate all points to new image
  
      #### In a rigthly transformed image y and x beginning should be (0/y_end_trans)
      #### as the line is perpendicular and allowed to span the whole image
      #### To trun correctly we have to consider that we trun clockwise = 2pi - angle
      
      angle_c = 1.5*np.pi - angle
      TransSb = point_trans(CoorSb, angle_c,img.shape,rotated_image.shape)
      TransEye = point_trans(CoorEye, angle_c ,img.shape,rotated_image.shape)
      TransMid = point_trans(Midpoint, angle_c ,img.shape,rotated_image.shape)
      Line_endpoint = point_trans((x_end,y_end), angle_c ,img.shape,rotated_image.shape)
      Line_begin = (0, Line_endpoint[1])
      
      ### The line should now be perpendicular so
      y_vals_trans = 0 * (x_range - TransMid[0]) + TransMid[1]
      
      ## This option shows the images in comparison
      if visualize == True:
        plt.clf()
        plt.subplot(1, 2, 2)
        plt.imshow(rotated_image)
        plt.plot(TransMid[0], TransMid[1], 'bo')
        plt.plot(TransEye[0],TransEye[1], 'ro')
        plt.plot(TransSb[0],TransSb[1], 'go')
        plt.plot(x_range, y_vals_trans, 'r-', label='Perpendicular Line')
        
        plt.xlim(0, rotated_image.shape[1])
        plt.ylim(0, rotated_image.shape[0])
        
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.plot(Midpoint[0], Midpoint[1], 'bo')
        plt.plot(CoorEye[0],CoorEye[1], 'ro')
        plt.plot(CoorSb[0],CoorSb[1], 'go')
        plt.plot(x_range, y_vals, 'r-', label='Perpendicular Line')
        
        plt.xlim(0, img.shape[1])
        plt.ylim(0, img.shape[0])
        plt.show()
      
      Perp_to_Image.append(rotated_image)
      list_of_Midpoints.append(TransMid)
      Rot_Angles.append(270 - np.rad2deg(angle))
    except:
      list_of_Midpoints.append(0)
      Perp_to_Image.append(0)
      Rot_Angles.append(0)

  return list_of_Midpoints, Perp_to_Image, Rot_Angles

def Measure_Width_Sperfeld(Rotated_Images, ListofMidpoints):
  ## Now calculate the width at the midpoint variable
  import numpy as np
  Widths = []
  X_start = []
  X_end = []
  for item in range(len(Rotated_Images)):
    try:
      MidRow = ListofMidpoints[item][1] ## Y coordinate for image of midpoint
      
      Width = np.sum(Rotated_Images[item][int(MidRow), :]) /255
      Widths.append(Width)
      
      reshaped_row = Rotated_Images[item][int(MidRow), :].reshape(-1, 1)

      X_start.append(np.argmax(reshaped_row))
      X_end.append(len(reshaped_row) - np.argmax(np.fliplr(reshaped_row)))

    except:
      Widths.append(0)
      X_start.append(0)
      X_end.append(0)
    
      ### To get the points so that we can build a visualization we need the coordinates of the points 
      ### and translate the back
  
  
  return Widths, X_start, X_end


  
