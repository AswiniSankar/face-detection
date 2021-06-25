# -*- coding: utf-8 -*-
"""facedetection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ztwjiZlDVSktOsn4x_Ylvw3hrNP6z97m
"""

!apt-get -qq install -y libsm6 libxext6
!pip install -q -U opencv-python

import cv2
from matplotlib import pyplot as plt
import numpy as np
face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

img1 = cv2.imread('sample.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
faces1 = face.detectMultiScale(gray1, 1.3, 5)
print(faces1)

for (x,y,w,h) in faces1:
  cv2.rectangle(img1,(x,y),(x+w,y+h),(0,400,0),10)
  #roigray=gray[y:y+h,x:x+w]
  roicolor=img1[y:y+h,x:x+w]

plt.grid(None)
plt.xticks([])
plt.yticks([])
imgplot=plt.imshow(img1)