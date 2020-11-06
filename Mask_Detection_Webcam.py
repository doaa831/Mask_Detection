#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 18:59:40 2020

@author: kinect
"""
#%% ========================== Importing Libraries ============================
import numpy as np
import cv2
import datetime
from keras.models import load_model
#%% =========================== Real-time Detection ==========================


def Mask_Camera():
    cap=cv2.VideoCapture(0)
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    model = load_model('model_ResNet50_100_99.74.h5')
    while cap.isOpened():
        ret, frame =cap.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)
            print('Number of faces detected:', len(faces))
            
            frame_with_detections = np.copy(frame)
            for(x,y,w,h) in faces:
                cv2.rectangle(frame_with_detections, (x,y), (x+w,y+h), (0,255,0), 3)
                face_img = frame_with_detections[y:y+h, x:x+w]
                resize_face_img = cv2.resize(face_img, (224,244))
                test_image=np.expand_dims(resize_face_img,axis=0)
                pred= np.argmax(model.predict(test_image), axis=-1)
                
                if pred==1:
                    cv2.rectangle(frame_with_detections,(x,y),(x+w,y+h),(0,0,255),3)
                    cv2.putText(frame_with_detections,'NO MASK',((x),y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
              
                else:
                    cv2.rectangle(frame_with_detections,(x,y),(x+w,y+h),(0,255,0),3)
                    cv2.putText(frame_with_detections,'MASK',((x),y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
               
                datet=str(datetime.datetime.now())
                cv2.putText(frame_with_detections,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
              
            cv2.imshow('img',frame_with_detections)
        
            if cv2.waitKey(1)==ord('q'):
                break
            
        # Break the loop
        else: 
            break
        
    cap.release()
    
    cv2.destroyAllWindows()
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
	Mask_Camera()