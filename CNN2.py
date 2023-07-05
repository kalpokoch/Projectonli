import tensorflow as tf
import os
import numpy as numpy
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras import models,layers
from tensorflow.keras.layers import Activation, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization

train_dir = r"C:\Users\lenovo\Documents\python practise\archive\train"
test = r"C:\Users\lenovo\Documents\python practise\archive\test"

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    batch_size=64,
    color_mode = "grayscale",
    shuffle=True,
    class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    test,
    target_size=(48,48),
    batch_size=64,
    color_mode="grayscale",
    shuffle=True,
    class_mode='categorical'
)

model = tf.keras.Sequential()
model.add(Conv2D(32, (3,3),padding='same',kernel_initializer='he_normal',input_shape = (48, 48, 1)))
model.add(Activation(('relu')))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3),padding='same',kernel_initializer='he_normal',input_shape = (48, 48, 1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(7, kernel_initializer='he_normal'))
model.add(Activation('softmax'))


model.summary()


checkpoint = tf.keras.callbacks.ModelCheckpoint('./Emotion_recog.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)
callbacks = [earlystop,checkpoint, reduce_lr]


opt = tf.keras.optimizers.Adam(lr=0.0005)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(train_generator,epochs=7,validation_data=validation_generator,callbacks=callbacks)

fig ,ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
fig.set_size_inches(12,4)

ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('Training Accuracy vs Validation Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Validation'], loc='upper left')

ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Training Loss vs Validation Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Validation'], loc='upper left')

plt.show()

model.save('Emotion_recog.h5')

img=image.load_img(r"C:\Users\lenovo\Documents\python practise\archive\test\happy\im3.png",target_size=(48,48),color_mode = "grayscale")

img = np.array(img)
plt.imshow(img)
print(img.shape)

label_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

img = np.expand_dims(img,axis=0)
img = img.reshape(1,48,48,1)
result = model.predict(img)
result = list(result[0])
print(result)

img_index = result.index(max(result))
print(label_dict[img_index])
plt.show()

train_loss, train_acc = model.evaluate(train_generator)
test_loss, test_acc = model.evaluate(validation_generator)
print("final accuracy = {:2.2f}, validation accuracy = {:.2f}".format(train_acc*100, test_acc*100))

model.save_weights('model_weights.h5')