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

  
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
keras.__version__
tf.__version__


##### Convert all tifs to png. This step should be in a different document#####




#### Initilaize data 224x224 is the biggest possible

img_width = 224
img_height = 224
target_size = (img_width, img_height)
output_n = 5 ## Number of classes
path = "/home/philipp/Daphnia_Images_Classifier_all"

#### Create train and vaildation set val split amount val data, seed has to be the same

train_data = keras.utils.image_dataset_from_directory(path,  
                                           image_size = target_size,
                                           labels = "inferred",
                                           color_mode = "rgb",
                                           label_mode = 'categorical', 
                                           subset = "training",
                                           class_names = ("magna", "longicephala", "pulex","cucullata", "longispina"),
                                           batch_size = 32,
                                           seed = 745,
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
                                           seed = 745,
                                           validation_split = 0.1
                                           )

image_data_train, target_train = zip(*train_data.as_numpy_iterator())
inputs = np.concatenate( image_data_train, axis=0 )
targets = np.concatenate( target_train, axis=0 )


def build_model(num_classes):
  
  #### Building a Model for EfficientNET
  #### Inputs are the expected shapes: Here 600,600,3 and get resized
  inputs = layers.Input(shape=(img_width, img_width, 3))
  #### Here Augment our Images using our self defined augmentation values FLip, transform, rotate
  x = inputs
  x = img_augmentation(inputs)
  
  ### initalize EfficientNETB7 for transfer learning, exclude Top, 
  ### Augmented Data as input_tensor, shape as in inputs, use imagenet weights
  model = tf.keras.applications.MobileNetV2(include_top=False, input_tensor=x, input_shape = (224,224,3), weights="imagenet")
  
  # Freeze the pretrained weights
  model.trainable = False
  
  # Add 2 layers on frozen model
  x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
  x = layers.BatchNormalization()(x)
  
  ### Dropout tries to prevent overfitting
  top_dropout_rate = 0.2
  x = layers.Dropout(top_dropout_rate, name="top_dropout")(x) 
  
  x = layers.Dense(128, activation="relu")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dense(64, activation="relu")(x)
  outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

  model = tf.keras.Model(inputs, outputs, name="VGG16")
  
  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
  model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
  )

  return model

def unfreeze_model(model):
      # We unfreeze the top 20 layers while leaving BatchNorm layers frozen to prevent 
      # overfitting due to batch statistics liekly changing
      for layer in model.layers[-20:]:
          if not isinstance(layer, layers.BatchNormalization):
              layer.trainable = True
      
      
      # Changing learning rate
      optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
      model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

### Data Augmentation 
### We define im augmentation as a layer that randomly Rotates transforms, flips and changes Contrast ####
img_augmentation = Sequential(
      [
          layers.RandomRotation(factor=0.15),
          layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
          layers.RandomFlip(),
          layers.RandomContrast(factor=0.1),
      ],
      name="img_augmentation",
  )


from sklearn.model_selection import KFold
kfold = KFold(n_splits = 5, shuffle=True)
fold_no = 1

acc_per_fold = []
loss_per_fold = []

for train, test in kfold.split(inputs, targets):
  model = build_model(num_classes= 5)
  
  
  ### 5 epochs learning with the frozen model and the 2 layers we built on top
  hist = model.fit(inputs[train], targets[train], epochs= 10 , validation_data= val_data, verbose=1)
  
  ############ Fine tune and update weights unfreeze and all
  
  unfreeze_model(model)
  
  epochs = 5  # Unfreeze some layers to allow fint tuning
  hist = model.fit(inputs[train], targets[train], epochs=epochs, validation_data= val_data, verbose=1)
  
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

new_list = [f"Loss: {x}, Acc: {y}" for x, y in zip(loss_per_fold, acc_per_fold)]
import pickle

with open("Scores_Kfold/Results_Kfold_MobileNetV2.txt", "w") as f:
    for s in new_list:
        f.write(str(s) +"\n")









