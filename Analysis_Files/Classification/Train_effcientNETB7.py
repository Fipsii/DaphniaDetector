#### After KFold validation we see that EfficientNETB7 is the best model
#### So now we build and save a model an use the test data to verfiy the accuracy

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

  
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
keras.__version__
tf.__version__

#### Functions
def plot_hist(hist):
      #### Funtions plots accuracy of train and val data against each other after every run
      plt.clf()
      plt.plot(hist.history["accuracy"])
      plt.plot(hist.history["val_accuracy"])
      plt.title("Training and validation accuracy; Epochs: " + str(5))
      plt.ylabel("accuracy")
      plt.xlabel("epoch")
      plt.legend(["train", "validation"], loc="upper left")
      plt.show()

def build_model(num_classes):
      #### Building a Model for EfficientNET
      #### Inputs are the expected shapes: Here 600,600,3 and get resized
      inputs = layers.Input(shape=(img_width, img_width, 3))
      #### Here Augment our Images using our self defined augmentation values FLip, transform, rotate
      x = inputs
      x = img_augmentation(inputs)
      ### initalize EfficientNETB7 for transfer learning, exclude Top, 
      ### Augmented Data as input_tensor, shape as in inputs, use imagenet weights
      model = EfficientNetB7(include_top=False, input_tensor=x, input_shape = (600,600,3), weights="imagenet")
  
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
      
      # Compile
      model = tf.keras.Model(inputs, outputs, name="EfficientNetB7")
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
      optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
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


img_width = 600 
img_height = 600
channels = 3
target_size = (img_width, img_height)
output_n = 5 ## Number of classes
path = "/home/philipp/Crops_Class"
  
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
                                           seed = 745,
                                           validation_split = 0.1
                                           )

train_batches = tf.data.experimental.cardinality(train_data)
val_ds = train_data.take(train_batches // 4)
train_ds = train_data.skip(train_batches // 4)

test_data = keras.utils.image_dataset_from_directory(path,  
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
model = build_model(num_classes= 5)

### Implement balanced weights to acc for data imbalance
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold, StratifiedKFold

## Take the train data and build the class_labels
image_data_train, target_train = zip(*train_data.as_numpy_iterator())
inputs = np.concatenate( image_data_train, axis=0 )
targets = np.concatenate( target_train, axis=0 )
unique_labels = np.unique(targets, axis=0)
label_mapping = {tuple(label): i for i, label in enumerate(unique_labels)}
y_multiclass = np.array([label_mapping[tuple(label)] for label in targets]) ## make it mulitclass not multilabel

class_labels = np.unique(y_multiclass)  # Replace y with your target labels
class_weights = compute_class_weight(class_weight = 'balanced', classes = class_labels, y = y_multiclass)  # Compute class weights
class_weights_dict = dict(enumerate(class_weights))


### 10 epochs learning with the frozen model and the 2 layers we built on top
hist = model.fit(train_ds, epochs= 10 ,validation_data = val_ds, verbose=1, class_weight=class_weights_dict)

#Unfreeze some layers to allow fine tuning
unfreeze_model(model) 

# Fine tune
hist = model.fit(train_ds, epochs=9, validation_data = val_ds, verbose=1, class_weight=class_weights_dict)

#### Test model on test set and save

model.save("/home/philipp/Scores_Kfold/EfficientNETB7_model_KFold_Balanced_Weight")

##### Predict
loaded_model = tf.keras.models.load_model("/home/philipp/Scores_Kfold/EfficientNETB7_model_KFold_Balanced_Weight")

## Build confusion matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
###
y_pred = []  # store predicted labels
y_true = []  # store true labels
dataset = test_data
test_data.class_names
# iterate over the dataset
for image_batch, label_batch in dataset:   # use dataset.unbatch() with repeat
   # append true labels
   y_true.append(label_batch)
   # compute predictions
   preds = loaded_model.predict(image_batch)
   # append predicted labels
   y_pred.append(np.argmax(preds, axis = - 1))

# convert the true and predicted labels into tensors
correct_labels = tf.concat([item for item in y_true], axis = 0)
predicted_labels = tf.concat([item for item in y_pred], axis = 0)
from sklearn.metrics import confusion_matrix

absolute_cm = confusion_matrix(np.argmax(correct_labels,axis = 1), predicted_labels)

# Calculate row sums
row_sums = absolute_cm.sum(axis=1)

# Convert absolute values to relative values
relative_cm = absolute_cm / row_sums[:, np.newaxis]

disp = ConfusionMatrixDisplay(confusion_matrix=relative_cm,display_labels= test_data.class_names)

disp.plot(cmap='Purples', colorbar = False)
plt.show()
plt.imsave("Conf_Matrix_Crops.png", disp)
len(predicted_labels)
