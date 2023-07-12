### Test scales with easy OCR
def detect_Number(List_with_Images):
  # Input Images, best case closely cropped to image
  # Output List of Numbers and letters 
  
  import easyocr
  reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory

  Results = []
  for x in List_with_Images:
    try:
      result = reader.readtext(x, detail = 0) ## Predict
      Results.append(result)
    except:
      Results.append(0)

def CropImage(Images):
  # takes as list of images
  # outputs a closley cropped list of images
  # based on CvCanny structure
  
  counter = 0
  List_of_crops = []
  for x in l:
    try:
      img = x
      
      Can = cv2.Canny(img,300,400, L2gradient = False)
      # The whiskers can confuse the process we have which is we already crop the far sides
      
      height, width = Can.shape
      Cropped_Can = Can[:, 10:width-10]
      Cropped_img_temp = img[:, 10:width-10]
      
      row_scan = np.sum(Cropped_Can, axis = 0)/255 ### Scans the image in rows
      coloumn_scan = np.sum(Cropped_Can, axis = 1)/255 ### Scans it in coloumns
      
      # Get the standard value for the scan
      reduced_column = coloumn_scan - coloumn_scan[0]
      
      # Find the index of the first number greater than 3 
      # This means we exclude all artifacts caused by jittery lines
      # smaller than 2 + 3 (5) Pixels
      first_index_y = np.argmax(reduced_column > 3)
    
      # Find the index of the last number greater than 3
      last_index_y = np.max(np.argwhere(reduced_column > 3))
      
      reduced_row = row_scan - row_scan[0]
      
      # Find the index of the first number greater than 3 
      # This means we exclude all artifacts caused by jittery lines
      # smaller than 2 + 3 (5) Pixels
      first_index_x = np.argmax(reduced_row > 3)
      
      # Find the index of the last number greater than 3
      last_index_x = np.max(np.argwhere(reduced_row > 3))
      
      if first_index_y-5 < 0: 
        first_index_y = 5
      if first_index_x-5 < 0: 
        first_index_x = 5
      
      print(counter, first_index_x-5,last_index_x+5,first_index_y-5,last_index_y+5, img.shape)
      img_crop = Cropped_img_temp[first_index_y-5:last_index_y+5,first_index_x-5:last_index_x+5]
      counter += 1
      
      ## Very small and pixelated values have to be resized
      ## If the image is less than 100 pixels in any dimension
      ## We resize it with Inter cubic
      ## We resize according to the smaller axis
      
      height, width = img_crop.shape
      print(img_crop.shape)
      ## Check size
      if (height < 100) or (width < 100):
        # Calculate scaling factors
        scale_height = 100/height
        scale_width = 100/width
        
        # Choose the bigger factor
        if scale_height > scale_width:
          scaling_factor = scale_height
        else:
          scaling_factor = scale_width
        print(scaling_factor,height*scaling_factor)
        # Scale up
        img_crop = cv2.resize(img_crop, (int(width*scaling_factor),int(height*scaling_factor)), interpolation = cv2.INTER_CUBIC)
      
      List_of_crops.append(img_crop)
    except:
      List_of_crops.append(img)
      counter += 1
      
  return Krops

def Sortlist(String_of_numbers):
  ### Drop entrys except numbers can cope with ['2','mm'] or ['2mm'] or ['2'], [].
  ### Input list of lists. 
  ### Output float values found in string
  import re
  
  numbers_only = [float(re.findall(r'\d+(?:\.\d+)?', entry[0])[0]) if entry and entry[0] and re.findall(r'\d+(?:\.\d+)?', entry[0]) else 0 for entry in String_of_numbers]

  return numbers_only

  
