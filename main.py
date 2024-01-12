from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def create_model(w, h, c):
  model = Sequential()
  model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape %= (w, h, c)))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
  model.add(Flatten())
  model.add(Dense(64, activation = 'relu'))
  model.add(Dense(1, activation = 'sigmoid'))
  model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
  return model
from PIL import Image
import os, os.path
import numpy as np

train_path = "/content/dataset/training_set/cats"
test_path = "/content/dog vs cat/dataset/test_set/cats"

def f(path):
  total_width = 0
  total_height = 0
  n = 0
  deleted = 0

  for f in os.listdir(path):
    try:
      img = Image.open(os.path.join(path, f))
      exif_data = img._getexif()

      if((np.asarray(img).shape[2] != 3)):
        print("Not RGB: " + f)
        os.remove(os.path.join(path, f))
        deleted += 1

      width, height = img.size
      total_width += width
      total_height += height
      n += 1

    except Exception as err:
      print("Error: " + str(err))
      print("Exif Data: " + str(exif_data))
      print("Path: " + os.path.join(path,f))
      os.remove(os.path.join(path, f))
      deleted += 1

  return (total_width, total_height, n, deleted)

width, height, n, f = f("/content/imagenette2-320/train/n02102040")

print(n)

width = 350
height = 350  
seed = 4245
train_path = "/content/dataset/training_set"
test_path = "/content/dataset/test_set"

import tensorflow as tf

train_rgb = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    label_mode="binary",
    color_mode = "rgb",
    batch_size = 32,
    image_size = (width, height),
    seed = seed,
    #subset = "training",
    #validation_split = 0.2
)

test_rgb = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    label_mode="binary",
    color_mode = "rgb",
    batch_size = 32,
    image_size = (width, height),
    seed = seed,
    #subset = "validation",
    #validation_split = 0.2
) 

train_gray = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    label_mode="binary",
    color_mode = "grayscale",
    batch_size = 32,
    image_size = (width, height),
    seed = seed,
    #subset = "training",
    #validation_split = 0.2
) 

test_gray = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    label_mode="binary",
    color_mode = "grayscale",
    batch_size = 32,
    image_size = (width, height),
    seed = seed,
    #subset = "validation",
    #validation_split = 0.2
)

import pandas as pd

model_1 = create_model(width, height, 3)
print("*****Starting first model*****")
history_color = model_1.fit(train_rgb, epochs = 50, validation_data = test_rgb)
hist_color_df = pd.DataFrame(history_color.history) 
print("*****Finished first model*****")
print("*****Starting to save*****")
with open("hist_color", mode='w') as f:
    hist_color_df.to_csv(f)
print("*****Saved*****")
model_2 = create_model(width, height, 1)
print("*****Starting second model*****")
history_gray = model_2.fit(train_gray, epochs = 50, validation_data = test_gray)
hist_gray_df = pd.DataFrame(history_gray.history) 
with open("hist_gray", mode='w') as g:
    hist_gray_df.to_csv(g)