import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

train = r"C:\Users\lenovo\Documents\python practise\archive\train"
test = r"C:\Users\lenovo\Documents\python practise\archive\test"

img_width, img_height = 256,256
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale = 1.0/255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

train_generator = train_datagen.flow_from_directory(
    train,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical'
)

test_datagen = ImageDataGenerator(rescale = 1.0/255)

test_generator = test_datagen.flow_from_directory(
    test,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical'
)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.compile(optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']    
)

checkpoint_path = 'best_weights.h5'

checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor = 'val_accuracy',
    mode = 'max',
    save_best_only = True,
    save_weights_only = True,
    verbose = 1
)

history = model.fit(
    train_generator,
    epochs = 30,
    validation_data = test_generator,
    callbacks = [checkpoint]
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

model.save('best_weights.h5') 