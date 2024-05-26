import cv2 
import numpy as np 
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt 

TF_ENABLE_ONEDNN_OPTS=0

image = cv2.imread('C:/Users/ADMIN/Pictures/Camera Roll/B.jpg')

cv2.imshow('color image',image)
print(image.shape)
cv2.waitKey(0) 


# image = image[100:700, 150:700]
cv2.imshow('color image',image)
print(image.shape)
cv2.waitKey(0) 



image = cv2.resize(image,(28,28))
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(img.shape)


fig, axe = plt.subplots(1,2, figsize = (12,12)) 
axe[0].imshow(image, )
axe[1].imshow(img, cmap='gray')  
plt.show() 



img = img.reshape(1,28,28,1)
img = img / 255 

model = tf.keras.models.load_model('model.h5')
prediction = model.predict(img)
predicted_class = np.argmax(prediction)
print(prediction)
print(predicted_class)