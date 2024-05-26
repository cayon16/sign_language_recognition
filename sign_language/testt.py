import cv2 
import numpy as np 
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt 

test = pd.read_csv('sign_language/sign_mnist_train.csv')
count = 0
for i in range(test.shape[0]):
    if test['label'][i] == 0:
        count = i
        break
print(count)

img = test.loc[count]
label = img['label']
img = img.drop(['label'])

image = img.values.reshape(28,28)

plt.title('IMAGE')
plt.imshow(image, cmap = 'gray')
plt.show() 


img = img.values.reshape(1,28,28,1)
img = img / 255 

model = tf.keras.models.load_model('model.h5')
prediction = model.predict(img)
predict_label = np.argmax(prediction)
print(predict_label )
print(label)