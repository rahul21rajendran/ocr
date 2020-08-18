
import cv2 # for capturing videos
import math # for mathematical operations
import matplotlib.pyplot as plt #3.3.0
import pandas as pd
import tensorflow as tf #2.3.0
from keras.preprocessing import image # 2.4.3
import numpy as np
from keras.utils import np_utils
from skimage.transform import resize #0.17.2
from numpy import asarray
from PIL import ImageTk,Image
import skimage.io
import csv
from skimage.io import imread,imshow,imsave
import glob
import os
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import preprocess_input
import shutil, sys  
import imutils
import argparse




#os.mkdir('data')
file =  r'ltdata/60.png'
im1 = cv2.imread(file,0)
im = cv2.imread(file)
ret,thresh1 = cv2.threshold(im1,180,278,cv2.THRESH_BINARY_INV)
#contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),1)    
    #if w>4 and h>4:
     #   cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),1)    
i=1
for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        #imgResized=cv2.imread(thresh1[y:y+h,x:x+w])
        #cv2.imwrite(imgResized,thresh1[y:y+h,x:x+w])
        #imgResized=cv2.resize(thresh1[y:y+h,x:x+w], (224,224))
        #imgResized = plt.imread(imgResized)
        #imageResized=plt.figure(figsize=(224,224))
        #if w>4 and h>4:
         #   cv2.imwrite(str(i)+".jpg",thresh1[y:y+h,x:x+w])
            
        #save individual images
        #cv2.imwrite(str(i)+".jpg",imgResized)
        cv2.imwrite("data/char%d.jpg" %i, thresh1[y:y+h,x:x+w])
        img = cv2.imread("data/char%d.jpg" %i)
        img=cv2.resize(img, (224,224))
        cv2.imwrite("data/char%d.jpg" %i, img)
        i=i+1
#cv2.imwrite('BindingBox3.jpg',im)
#cv2.imshow("Image", im)
#cv2.waitKey(0)



a1=0
a=1
b=i
a=a-1
b=b
with open('data/test.csv', 'w') as csvoutput:
    writer = csv.writer(csvoutput)
    for i in range(a,b):
        if(a1==0):
            writer.writerow(['Image_ID'])
        else:
            writer.writerow(['data/char'+str(i)+'.jpg'])
        a1=a1+1
df = pd.read_csv('data/test.csv')
df.to_csv('data/test.csv', index=False)

test = pd.read_csv('data/test.csv')

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

test_image = []
for img_name in test.Image_ID:
    img = plt.imread('' + img_name)
    test_image.append(img)
test_img = np.array(test_image)
test_image = []
for i in range(0,test_img.shape[0]):
    a = resize(test_img[i], preserve_range=True, output_shape=(224,224)).astype(int)
    test_image.append(a)

test_image = np.array(test_image)
tf.keras.applications.vgg16.preprocess_input(test_image, data_format=None)


model = tf.keras.models.load_model('combined_model.h5')


model.layers[0].input_shape#(None, 224, 224, 3)

test_image = base_model.predict(test_image)

print(i+1)

test_image = test_image.reshape(i+1, 7*7*512)

# zero centered images
test_image = test_image/test_image.max()

predictions = model.predict_classes(test_image)

print("Number of combined", predictions[predictions==0].shape[0], "")
print("Number of single", predictions[predictions==1].shape[0], "")


print("predicted outputs")
rounded_predictions = model.predict_classes(test_image)

#print(rounded_predictions[3])
print(rounded_predictions)

a=-1


im1 = cv2.imread(file,0)
im = cv2.imread(file)

ret,thresh1 = cv2.threshold(im1,180,278,cv2.THRESH_BINARY_INV)
#contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# find contours in thresholded image, then grab the largest
# one
contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    #cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),1) 
 
    #print(x,y,w,h)   
 
#c = max(cnts, key=cv2.contourArea)
# determine the most extreme points along the contour
 
    #if w>4 and h>4:
     #   cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),1)    

i=0
for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        #if w>4 and h>4:
         #   cv2.imwrite(str(i)+".jpg",thresh1[y:y+h,x:x+w])
            
        #save individual images
        #cv2.imwrite(str(i)+".jpg",thresh1[y:y+h,x:x+w])
        if(rounded_predictions[i] == 0):
            cv2.line(im,(x+int(w/2),y-5),(x+int(w/2),y+h+5),(0,0,255),2)
        i=i+1

cv2.imwrite('BindingBox3.png',im)
cv2.imshow("Image", im)
cv2.waitKey(0)
















