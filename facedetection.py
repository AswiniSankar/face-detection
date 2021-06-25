

import cv2
from matplotlib import pyplot as plt
import numpy as np
face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#to detect the face
img1 = cv2.imread('sample.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
faces1 = face.detectMultiScale(gray1, 1.3, 5)
#print(faces1) -->prints the coordinates of each faces

#to draw the rectangle box on the detected face
for (x,y,w,h) in faces1:
  cv2.rectangle(img1,(x,y),(x+w,y+h),(0,400,0),10)
  roigray=gray1[y:y+h,x:x+w]
  roicolor=img1[y:y+h,x:x+w]

plt.grid(None)
plt.xticks([])
plt.yticks([])
imgplot=plt.imshow(img1)#give the resulten image
