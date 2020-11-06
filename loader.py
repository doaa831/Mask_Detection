#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 15:19:26 2020

@author: Doaa
"""
#%% ============================= Importing Libraries =========================
import os
import pickle
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

#%% ===================== Part 1 - Create Dataset ====================
"""
Input: root of dataset
Output: Digital Images and it's label

"""
def load_Dataset(root):
    
    image_Paths = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if not name.startswith('.'):
                image_Paths.append(os.path.join(path, name))
                
    data = []
    labels =  []
    for image_Path in image_Paths:
        image = load_img(image_Path, target_size=(224,224))
        label = image_Path.split(os.sep)[-2]
        image = img_to_array(image)
        image = preprocess_input(image) # normalization
        
        data.append(image)  
        labels.append(label)
    
    return data,labels
        
    with open('Dataset.pickle', 'wb') as f:
        pickle.dump([data, labels], f)  

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if __name__ == "__main__":
