from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
keras = tf.keras

AUTOTUNE = tf.data.experimental.AUTOTUNE

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tensorflow.keras import datasets, layers, models
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import sys
import os

EPOCHS = 2
BATCH_SIZE = 32
STEPS_PER_EPOCH
IMG_WIDTH = 160
IMG_HEIGHT = 160

class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()


def load():
  data_dir = pathlib.Path('../images')
  image_count = len(list(data_dir.glob('*/*.jpg')))
  # print("total images: ", image_count)
  CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])

  # print(CLASS_NAMES)

  # lol = list(data_dir.glob('World_of_Warcraft/*'))
  # (ACTUAL_IMG_WIDTH, ACTUAL_IMG_HEIGHT) = Image.open(str(lol[0])).size
  # IMG_WIDTH = int(ACTUAL_IMG_WIDTH / 8)
  # IMG_HEIGHT = int(ACTUAL_IMG_HEIGHT / 8)

  image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,  validation_split=0.2)
  train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                       batch_size=BATCH_SIZE,
                                                       shuffle=True,
                                                       target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                       subset='training',
                                                       class_mode='binary',
                                                       classes = list(CLASS_NAMES))

  validate_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     subset='validation',
                                                     class_mode='binary',
                                                     classes = list(CLASS_NAMES))

  return (train_data_gen, validate_data_gen)

def verify(image_batch, label_batch):
  plt.figure(figsize=(IMG_WIDTH, IMG_HEIGHT))
  plt.subplots_adjust(hspace=0.5)
  for n in range(30):
    plt.subplot(6,5,n+1)
    plt.imshow(image_batch[n])
    plt.title(CLASS_NAMES[int(label_batch[n])])
    plt.axis('off')
  plt.show()


def train(train_data_gen, validate_data_gen):

  IMG_SHAPE = (160, 160, 3)

  # Create the base model from the pre-trained model MobileNet V2
  base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                 include_top=False,
                                                 weights='imagenet')

  image_batch, label_batch = next(train_data_gen)
  feature_batch = base_model(image_batch)
  # print(feature_batch.shape)

  base_model.trainable = False

  # base_model.summary()


  global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
  feature_batch_average = global_average_layer(feature_batch)
  print(feature_batch_average.shape)


  prediction_layer = keras.layers.Dense(1)
  prediction_batch = prediction_layer(feature_batch_average)
  print(prediction_batch.shape)


  model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
  ])


  base_learning_rate = 0.0001
  model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy'])

  model.summary()

  print(len(model.trainable_variables))

  # loss0,accuracy0 = model.evaluate(validate_data_gen, steps = 10)

  history = model.fit(train_data_gen,
                      epochs=10,
                      validation_data=validate_data_gen)


def main():
  (train_data_gen, validate_data_gen) = load()

  # image_batch, label_batch = next(train_data_gen)
  # print(image_batch.shape)
  # verify(image_batch, label_batch)

  train(train_data_gen, validate_data_gen)


main()
