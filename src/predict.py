from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
keras = tf.keras

AUTOTUNE = tf.data.experimental.AUTOTUNE

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import sys
import os


def main():
  IMG_WIDTH = 440
  IMG_HEIGHT = 248

  data_dir = pathlib.Path('../images/predict')
  CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
  print(CLASS_NAMES)
  image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
  predict_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                       batch_size=32,
                                                       shuffle=True,
                                                       target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                       class_mode='binary')


  predict_image_batch, predict_label_batch = next(predict_data_gen)

  # for img in predict_image_batch:
  #   plt.imshow(img, cmap=plt.cm.binary)
  #   plt.show()


  model = tf.keras.models.load_model('./dope.h5')
  predictions = model.predict(predict_image_batch)

  i = 0
  for prediction in predictions:
    print(prediction)
    plt.imshow(predict_image_batch[i], cmap=plt.cm.binary)
    plt.show()

    print(np.argmax(prediction))
    print(predict_label_batch[i])
    i = i+1


main()
