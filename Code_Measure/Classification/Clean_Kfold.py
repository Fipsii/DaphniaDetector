##### Works but very slowly, CUDA does not work as well as other parts while loop bug #####
import sys
print(sys.prefix)
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import (Conv2D,
                          Dense,
                          LeakyReLU,
                          BatchNormalization, 
                          MaxPooling2D, 
                          Dropout,
                          Flatten)
#from keras.optimizers import RMSprops
from keras.callbacks import ModelCheckpoint, TensorBoard
from datetime import datetime as dt
import tensorflow as tf
import importlib
importlib.reload(tf)
tf.compat.v1.Session()
import os as os
from PIL import Image
from tensorflow.keras.applications import EfficientNetB7
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
#from tf_keras_vis.activation_maximization import ActivationMaximization
#from tf_keras_vis.activation_maximization.callbacks import Progress
#from tf_keras_vis.activation_maximization.input_modifiers import Jitter, Rotate2D
#from tf_keras_vis.activation_maximization.regularizers import TotalVariation2D, Norm
#from tf_keras_vis.utils.model_modifiers import ExtractIntermediateLayer, ReplaceToLinear
#from tf_keras_vis.utils.scores import CategoricalScore
#from tensorflow.keras.preprocessing.image import load_img
import cv2 as cv2

keras.__version__
tf.__version__
#### Functions
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7

# Load the base model with pre-trained weights



img_width = 600 
img_height = 600
channels = 3
target_size = (img_width, img_height)
output_n = 5 ## Number of classes
path = "/home/philipp/Daphnia_Images_Classifier_all"
  
#### Create train and vaildation set val split amount val data, seed has to be the same
#### Image Size set to 600 (Requirement of EffiecentNETB7)
#### Labels = "inferred" extracts labels based on the folders in Data_DaphnaisPNG_entpackt
#### color_mode, EfficientNET requires 3 channels RGB images, so RGB is chosen
#### Labels are categories, each categorie corresponds to a species
#### class_names declares class names additionally
#### validation split 10% of images used as validation (Later split into 5% for test and 5% for validation)
#### batch_size amount of batches for read in (29 batches used)
#### seed is set randomly and synchronizes train_data with val_data aka needs to be the same for both

train_data = keras.utils.image_dataset_from_directory(path,  
                                           image_size = target_size,
                                           labels = "inferred",
                                           color_mode = "rgb",
                                           label_mode = 'categorical', 
                                           subset = "training",
                                           class_names = ("magna", "longicephala", "pulex","cucullata", "longispina"),
                                           batch_size = 32,
                                           seed = 646,
                                           validation_split = 0.1
                                           )
                                           
val_data = keras.utils.image_dataset_from_directory(path,  
                                           image_size = target_size,
                                           labels = "inferred",
                                           color_mode = "rgb",
                                           label_mode = 'categorical', 
                                           subset = "validation",
                                           class_names = ("magna", "longicephala", "pulex","cucullata", "longispina"),
                                           batch_size = 32,
                                           seed = 646,
                                           validation_split = 0.1
                                           )

### Divide the data 90/10
import tensorflow as tf


image_data_train, target_train = zip(*train_data.as_numpy_iterator())
inputs = np.concatenate( image_data_train, axis=0 )
targets = np.concatenate( target_train, axis=0 )


print(inputs.shape,targets.shape)


##### val_batches is the amount of batches used for validation
##### We then split the data. Every second batch is for testing every other for validation

### Data Augmentation 
### We define im augmentation as a layer that randomly Rotates transforms, flips and changes Contrast ####
  
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
  
img_augmentation = Sequential(
      [
          layers.RandomRotation(factor=0.15),
          layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
          layers.RandomFlip(),
          layers.RandomContrast(factor=0.1),
      ],
      name="img_augmentation",
  )

###### Build model
##### Here Augment our Images using our self defined augemtation values FLip, transform, rotate

# inputs = layers.Input(shape=(img_width, img_width, 3))
#### Here Augment our Images using our self defined augemtation values FLip, transform, rotate
def build_model(num_classes):
      #### Building a Model for EfficientNET
      #### Inputs are the expected shapes: Here 600,600,3 and get resized
      sh_input = layers.Input(shape=(img_width, img_width, 3))
      #### Here Augment our Images using our self defined augemtation values FLip, transform, rotate
      x = img_augmentation(sh_input)
      ### initalize EfficientNETB7 for transfer learning, exclude Top, 
      ### Augmented Data as input_tensor, shape as in inputs, use imagenet weights
      model = EfficientNetB7(include_top=False, input_tensor=x, input_shape = (600,600,3), weights="imagenet")
  
      # Freeze the pretrained weights
      model.trainable = False
      
      # Rebuild top and use our data as top
      x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
      x = layers.BatchNormalization()(x)
      
      ### Dropout tries to prevent overfitting
      top_dropout_rate = 0.2
      x = layers.Dropout(top_dropout_rate, name="top_dropout")(x) 
      outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)
  
      # Compile
      model = tf.keras.Model(sh_input, outputs, name="EfficientNetB7")
      
      for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
          layer.trainable = True
      
      optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
      model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
      )
      return model



# Initiate kFold cross validation

from sklearn.model_selection import KFold
kfold = KFold(n_splits = 5, shuffle=True)
fold_no = 1

acc_per_fold = []
loss_per_fold = []

for train, test in kfold.split(inputs, targets):
  
  model = build_model(5)
  epochs = 15  # Number of Epochs in unfrozen mode
  ### Augment images
  
  hist = model.fit(inputs[train], targets[train], epochs= epochs, validation_data= val_data, verbose=1)
  
  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])

  # Increase fold number
  fold_no = fold_no + 1

print("loss: ", loss_per_fold, "acc: ", acc_per_fold)


model.save('Kfold_validated_model/') 




