## Classification module for species


def Classify_Species(Folder_With_Images, Classifier_Location):
  
  import tensorflow as tf
  import os
  import shutil
  import keras
  import numpy as np
  ### Input Folder_With_Images: This is quite confusing but flow_from_directory
  ### Wants the directory above the folder with images so: The structure should be
  ### Folder -> Folder with Images -> Images.
  ### So We create a tempfolder for that we delete afterwards
  ### Folder_With_Images -> Subfolder called temp_directory -> images
  ### Load the created model
  
  ## Supress warnings
  
  tf.compat.v1.logging.set_verbosity(30) # WARN
  
  # Create a temporary directory to organize the images
  temp_dir = Folder_With_Images + '/temp_directory'

  os.makedirs(temp_dir, exist_ok=True)
  
  # Move images to the label subdirectory
  for filename in os.listdir(Folder_With_Images):
      if filename.endswith('.jpg') or filename.endswith('.png'):  # JPGs or PNGs
          src = os.path.join(Folder_With_Images, filename)
          dst = os.path.join(temp_dir, filename)
          shutil.copyfile(src, dst)
  
  
  Classifier = tf.keras.models.load_model(Classifier_Location)
  predictions = []
  
  ### The settings we have for our model
  # Image size and Channels

  img_width = 600 
  img_height = 600
  channels = 3
  target_size = (img_width, img_height)
  
  # Number of classes
  output_n = 5 
  
  ### Now read in data 
  data = keras.utils.image_dataset_from_directory(Folder_With_Images,  
         image_size = target_size,
         color_mode = "rgb",
         label_mode = 'categorical', 
         shuffle = False,
         
         )
                                           
  ### After this  we define our classes
  
  label_mapping = ["magna", "longicephala", "pulex","cucullata", "longispina"]
  
  ### Now we can perform the prediction
  prediction = Classifier.predict(data)
  
  # Predicitions outputs values we want the
  # likeliest prediction 
  
  predicted_classes = np.argmax(prediction, axis=1)
  
  # Convert predictions into names
  species = [label_mapping[class_index] for class_index in predicted_classes]
  
  # Remove the temporary directory
  shutil.rmtree(temp_dir)
  
  return species

