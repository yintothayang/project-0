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

EPOCHS = 10
BATCH_SIZE = 32
STEPS_PER_EPOCH = 32
IMG_WIDTH = 440
IMG_HEIGHT = 248
CLASS_NAMES = []

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
  print(CLASS_NAMES)

  # lol = list(data_dir.glob('World_of_Warcraft/*'))
  # (ACTUAL_IMG_WIDTH, ACTUAL_IMG_HEIGHT) = Image.open(str(lol[0])).size
  # IMG_WIDTH = int(ACTUAL_IMG_WIDTH / 8)
  # IMG_HEIGHT = int(ACTUAL_IMG_HEIGHT / 8)

  image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,  validation_split=0.3)
  train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                       batch_size=BATCH_SIZE,
                                                       shuffle=True,
                                                       target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                       subset='training',
                                                       class_mode='binary')


  validate_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                          batch_size=BATCH_SIZE,
                                                          shuffle=True,
                                                          target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                          subset='validation',
                                                          class_mode='binary')


  return (train_data_gen, validate_data_gen)

def verify(image_batch, label_batch, labels_by_key):
  plt.figure(figsize=(IMG_WIDTH, IMG_HEIGHT))
  plt.subplots_adjust(hspace=0.5)
  for n in range(30):
    plt.subplot(6,5,n+1)
    plt.imshow(image_batch[n])
    plt.title(labels_by_key[int(label_batch[n])])
    plt.axis('off')
  plt.show()


def train(train_data_gen, validate_data_gen):
  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(len(CLASS_NAMES), activation='softmax'))

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  history = model.fit(train_data_gen,
                      epochs=5,
                      validation_data=validate_data_gen)

  plt.plot(history.history['accuracy'], label='accuracy')
  plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.ylim([0.5, 1])
  plt.legend(loc='lower right')
  plt.show()

  # test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)


def main():
  (train_data_gen, validate_data_gen) = load()

  image_batch, label_batch = next(train_data_gen)

  labels_by_key = {v: k for k, v in train_data_gen.class_indices.items()}

  # print(image_batch.shape)
  verify(image_batch, label_batch, labels_by_key)
  # print(len(train_data_gen.filepaths))
  train(train_data_gen, validate_data_gen)


main()
