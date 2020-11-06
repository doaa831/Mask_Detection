#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 00:04:15 2020

@author: Doaa
"""
#%% ========================== Importing Libraries ============================
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.applications.resnet50 import ResNet50
from keras.models import Model

#%% ================ Part 2 - Loading & Preprocessing Dataset =================

# Loading Dataset from pickle file
with open('Dataset.pickle', 'rb') as f:
    data, labels = pickle.load(f)  
    
data = np.array(data,dtype="float32")
labels = np.array(labels)

# Catogrize labels to OneHotEncoder
labels = LabelBinarizer().fit_transform(labels)  
labels = to_categorical(labels)   

X_train,X_test,y_train,y_test = train_test_split(data,labels,test_size=0.2,
                                                   stratify=labels,
                                                   random_state=42)

train_datagen = ImageDataGenerator(shear_range = 0.2,
                                   width_shift_range=0.2,
                                   rotation_range=20,
                                   height_shift_range=0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator()

train_set = train_datagen.flow(X_train,y_train)
test_set = test_datagen.flow(X_test,y_test)

#%% ====================== Part 2 - Building the model ========================
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Creating model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

input_tensor = Input(shape=(224, 224, 3))
# loading the resnet50 model without classifier layers
# create the base pre-trained model
base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
# add new classifier layers
pool1 = GlobalAveragePooling2D()(base_model.output)
# let's add a fully-connected layer
class1 = Dense(1024, activation='relu')(pool1)
# and a logistic layer -- let's say we have 2 classes
output = Dense(2, activation='softmax')(class1)
# this is the model we will train
model = Model(input_tensor, outputs=output)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional ResNet50 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# summarize
model.summary()

#%%%%%%%%%%%%%%%%%%%%%%%%%% Model Training & Testing %%%%%%%%%%%%%%%%%%%%%%%%%%

# train the model on the new data for a few epochs
history = model.fit(
  train_set,
  validation_data= test_set,
  epochs=100,
  steps_per_epoch=len(train_set)//32,
  validation_steps=len(test_set)//32
)

score = model.evaluate(X_test, y_test, verbose=0) 
accuracy = 100*score[1] 
print(accuracy )

model.save('Mask_Detection_Model_ResNet50.h5')

# loss
plt.plot(history.history['loss'], label='train loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(history.history['accuracy'], label='train acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

