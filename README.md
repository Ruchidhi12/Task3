# Task3
 Program to develop an image classification model to distinguish between images of cats and dogs using data science technique in Python.
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# Load the Cats vs. Dogs dataset
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
testrain_generator = train_datagen.flow_from_directory(
t_datagen = ImageDataGenerator(rescale=1./255)
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
validation_generator= 'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
        # Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
# Train the model
model.fit_generator(
        train_generator,
        steps_per_epoch=2000,

            epochs=10,
validation_data=validation_generator,
        validation_steps=800)
# Save the model
model.save('cats_and_dogs.h5')








 

