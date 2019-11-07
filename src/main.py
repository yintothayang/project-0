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
(IMG_WIDTH, IMG_HEIGHT) = Image.open(str(lol[0])).size

print(IMG_WIDTH, IMG_HEIGHT)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,  validation_split=0.2)

EPOCHS = 5
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

image_batch, label_batch = next(val_data_gen)
# show_batch(image_batch, label_batch)

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



# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))


model.compile(optimizer='adam',
              # loss='sparse_categorical_crossentropy',
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


# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')

# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
