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
BATCH_SIZE = 128
STEPS_PER_EPOCH = 33
IMG_WIDTH = 150
IMG_HEIGHT = IMG_WIDTH
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
  data_dir = pathlib.Path('../images/test')
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
  model = models.Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
  ])


  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  model.summary()


  history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=len(train_data_gen.filepaths) / BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validate_data_gen,
    validation_steps=len(validate_data_gen.filepaths) / BATCH_SIZE
  )



  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(EPOCHS)

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()


  # class_names = sorted(train_data_gen.class_indices.items(), key=lambda pair:pair[1])
  # class_names = np.array([key.title() for key, value in class_names])
  # print(class_names)

  # image_batch, label_batch = next(validate_data_gen)
  # predicted_batch = model.predict(image_batch)
  # predicted_id = np.argmax(predicted_batch, axis=-1)
  # predicted_label_batch = class_names[predicted_id]
  # label_id = np.argmax(label_batch, axis=-1)

  # plt.figure(figsize=(10,9))
  # plt.subplots_adjust(hspace=0.5)

  # print(predicted_label_batch)
  # print(label_id)

  # for n in range(30):
  #   plt.subplot(6,5,n+1)
  #   plt.imshow(image_batch[n])
  #   color = "green" if predicted_id[n] == label_id[n] else "red"
  #   plt.title(predicted_label_batch[n].title(), color=color)
  #   plt.axis('off')
  # _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")

  # plt.show()


def main():
  (train_data_gen, validate_data_gen) = load()

  image_batch, label_batch = next(train_data_gen)

  labels_by_key = {v: k for k, v in train_data_gen.class_indices.items()}
  # print(image_batch.shape)
  # verify(image_batch, label_batch, labels_by_key)
  # print(len(train_data_gen.filepaths))
  train(train_data_gen, validate_data_gen)


main()
