import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from keras.datasets import mnist # là tập dữ liệu chữ viết tay từ 0 =>9. 1 dữ liệu gồm 1 ảnh đen trắng 28x28
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import sys
import seaborn as sns 
TF_ENABLE_ONEDNN_OPTS=0

train_df=pd.read_csv("sign_language\sign_mnist_train.csv")
test_df=pd.read_csv("sign_language\sign_mnist_test.csv")


train_label = train_df['label']
test_label = test_df['label']
trainset = train_df.drop(['label'],axis=1)
testset = test_df.drop(['label'],axis=1)


for i in range (len(train_label)):
    if train_label[i]>9:
        train_label[i] -=1
        

for i in range (len(test_label)):
    if test_label[i]>9:
        test_label[i] -=1


y_train = tf.keras.utils.to_categorical(train_label, num_classes = 24)
y_test = tf.keras.utils.to_categorical(test_label, num_classes = 24)

x_train = trainset.values.reshape(-1,28,28,1)
x_test = testset.values.reshape(-1,28,28,1)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

fig,axe=plt.subplots(2,2)
fig.suptitle('Preview of dataset')
axe[0,0].imshow(x_train[0].reshape(28,28),cmap='gray')
axe[0,0].set_title('label: 3  letter: C')
axe[0,1].imshow(x_train[1].reshape(28,28),cmap='gray')
axe[0,1].set_title('label: 6  letter: F')
axe[1,0].imshow(x_train[2].reshape(28,28),cmap='gray')
axe[1,0].set_title('label: 2  letter: B')
axe[1,1].imshow(x_train[4].reshape(28,28),cmap='gray')
axe[1,1].set_title('label: 13  letter: M')
plt.show() 


# countplot 
plt.figure(figsize=(10, 6))
sns.countplot(x=train_label)
plt.title('Frequency of Each Label in Training Set')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x=test_label)
plt.title('Frequency of Each Label in Training Set')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.show()



datagen = ImageDataGenerator(
    rotation_range=20, # độ xoay
    width_shift_range=0.2, # 20% dịch chuyển ngang 
    height_shift_range=0.2, # 20% dọc 
    shear_range=0, # độ cắt góc 
    zoom_range=0, # phóng to, thu nhỏ 20%
    horizontal_flip=True, # lật ngang ảnh ngẫu nhiên 
    fill_mode='nearest' # điền các pixel bị mất do xoay, phóng to, ... bằng pixel gần nhất 
)

datagen.fit(x_train)

x_train = x_train / 255
x_test = x_test/255   # giúp normalization dữ liệu
TF_ENABLE_ONEDNN_OPTS=0

model=Sequential()
model.add(Conv2D(128,kernel_size=(5,5),
                 strides=1,padding='same',activation='relu',input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(3,3),strides=2,padding='same'))
model.add(Conv2D(64,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))
model.add(Conv2D(32,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))
          
model.add(Flatten())

model.add(Dense(units=512,activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=24,activation='softmax'))
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#xây dựng mô hình
model.build()

#tổng kết lại mô hình trước khi train
model.summary()
model.save



model.fit(datagen.flow(x_train, y_train, batch_size=256), epochs=35, validation_data=(x_test, y_test), shuffle = 1) # tráo thứ tự data để batch ngẫu nhiên 
# predictions = cnn.predict(x_test)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('MODEL ACCURACY = {}%'.format(test_accuracy*100))
print('MODEL LOSS = {}%'.format(test_loss))
model.save('model.h5')

