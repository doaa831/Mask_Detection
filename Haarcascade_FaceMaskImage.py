#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 16:45:39 2020

@author: Doaa
"""
#%% ========================== Importing Libraries ============================
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from keras.models import load_model

#%% ============================= Detect Images ==============================

def Mask_Image(image_path):
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    model = load_model('model_ResNet50_100_99.74.h5')
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_title('Image with Mask Detection')  
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.06,minNeighbors=5,minSize=(1,1), flags=cv2.CASCADE_SCALE_IMAGE)
    print('Number of faces detected:', len(faces))
        
    image_with_detections = np.copy(image)
    for(x,y,w,h) in faces:
        cv2.rectangle(image_with_detections, (x,y), (x+w,y+h), (0,255,0), 2)
        face_img = image_with_detections[y:y+h, x:x+w]
        resize_face_img = np.array(cv2.resize(face_img, (224,244)))
        test_image= np.expand_dims(resize_face_img,axis=0)
        pred= np.argmax(model.predict(test_image), axis=-1)
                
        if pred==1:
            cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(image_with_detections,'NO MASK',((x),y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
              
        else:
            cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(image_with_detections,'MASK',((x),y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    # show the output image
    ax.imshow(cv2.cvtColor(image_with_detections, cv2.COLOR_BGR2RGB))
    ax.savefig('Detected_Image')	
    cv2.waitKey(0)
	
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to input image")
	args = vars(ap.parse_args())
	Mask_Image(args["image"])
