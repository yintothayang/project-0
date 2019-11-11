from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

from PIL import Image
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
import pathlib


data_dir = pathlib.Path('../images/twitch')
image_count = len(list(data_dir.glob('*/*.jpg')))

print(image_count)

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])

print(CLASS_NAMES)

lol = list(data_dir.glob('lol/*'))
(ACTUAL_IMG_WIDTH, ACTUAL_IMG_HEIGHT) = Image.open(str(lol[0])).size

print(ACTUAL_IMG_WIDTH, ACTUAL_IMG_HEIGHT)

IMG_WIDTH = int(ACTUAL_IMG_WIDTH / 8)
IMG_HEIGHT = int(ACTUAL_IMG_HEIGHT / 8)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,  validation_split=0.2)

EPOCHS = 10
BATCH_SIZE = 32
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     subset='training',
                                                     class_mode='binary',
                                                     classes = list(CLASS_NAMES))

val_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     subset='validation',
                                                     class_mode='binary',
                                                     classes = list(CLASS_NAMES))


def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')
  plt.show()

image_batch, label_batch = next(train_data_gen)
# print(image_batch)
# print(label_batch)
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# plotImages(image_batch[:5])
# show_batch(image_batch, label_batch)

model = models.Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH,3)),
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


total_train = 9312
total_val = 2327

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train,
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=total_val
)


# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(EPOCHS)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()


# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')

# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# plt.show()
