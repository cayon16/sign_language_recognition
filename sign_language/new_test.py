from skimage import transform
from skimage import data
import os
import cv2
import numpy as np
from skimage.color import rgb2gray
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) for f in os.listdir(label_directory)]
        for f in file_names:
            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (28, 28))
            images.append(img)
            labels.append(ord(d) - ord('A'))  # Chuyển đổi ký tự thành số từ 0-25
    return np.array(images), np.array(labels)

ROOT_PATH = "C:/Users/ADMIN/Desktop/python code/sign_language/archive"
train_data_directory = os.path.join(ROOT_PATH, "train")

images, labels = load_data(train_data_directory)

x_train = np.array(images)
y_train = np.array(labels)

# Kiểm tra số lượng nhãn khác nhau
print("Total number of classes:", len(set(y_train)))
print("Label Array:", [chr(X + ord('A')) for X in set(y_train)])

# Hiển thị một số ảnh
plt.imshow(x_train[0], cmap="gray")
plt.show()

plt.imshow(x_train[250], cmap="gray")
plt.show()

plt.imshow(x_train[900], cmap="gray")
plt.show()




test_data_directory = os.path.join(ROOT_PATH, "test")
images, labels = load_data(test_data_directory)

x_test = np.array(images)
y_test = np.array(labels)

# Hiển thị một số ảnh
plt.imshow(x_test[0], cmap="gray")
plt.show()

plt.imshow(x_test[250], cmap="gray")
plt.show()

print("Total number of classes:", len(set(y_test)))
print("Label Array:", [chr(X + ord('A')) for X in set(y_test)])

# Chuẩn hóa dữ liệu
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encoding
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.transform(y_test)

# Định hình lại dữ liệu
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Data augmentation
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

# MODEL BUILDING 
model = Sequential()
model.add(Conv2D(128, kernel_size=(5, 5),
                 strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(3, 3), strides=2, padding='same'))
model.add(Conv2D(64, kernel_size=(2, 2),
                 strides=1, activation='relu', padding='same'))
model.add(MaxPool2D((2, 2), 2, padding='same'))
model.add(Conv2D(32, kernel_size=(2, 2),
                 strides=1, activation='relu', padding='same'))
model.add(MaxPool2D((2, 2), 2, padding='same'))
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(rate=0.25))  # 25% set value = 0 
model.add(Dense(units=24, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(datagen.flow(x_train, y_train, batch_size=128), epochs=35, validation_data=(x_test, y_test), shuffle=True)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('MODEL ACCURACY = {}%'.format(test_accuracy * 100))
print('MODEL LOSS = {}%'.format(test_loss))
model.save('model.h6')

# Analysis results 
epochs = [i for i in range(35)]
fig, ax = plt.subplots(1, 2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(16, 9)

ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
ax[0].plot(epochs, val_acc, 'ro-', label='Testing Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs, train_loss, 'g-o', label='Training Loss')
ax[1].plot(epochs, val_loss, 'r-o', label='Testing Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()


predictions = model.predict(x_test)
aaa = [] 
for i in range(len(predictions)):
    aaa.append(np.argmax(predictions[i]))

print(predictions)
print(predictions.shape)
print(aaa)
for i in range(len(aaa)):
    if(aaa[i] >= 9):
        aaa[i] += 1
aaa[:5]     


classes = ["Class " + str(i) for i in range(25) if i != 9]
print(classification_report(y, aaa, target_names = classes))

predictions = aaa 

# ma trận nhầm lẫn, các hàng là giá trị thực tế, các cột là giá trị dự đoán 
cm = confusion_matrix(y,predictions)
cm = pd.DataFrame(cm , index = [i for i in range(25) if i != 9] , columns = [i for i in range(25) if i != 9])
plt.figure(figsize = (15,15))
sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='')
plt.show() 

correct = np.nonzero(predictions == y)[0]

i = 0
for c in correct[:6]:
    plt.subplot(3,2,i+1)
    plt.imshow(x_test[c].reshape(28,28), cmap="gray", interpolation='none')
    plt.title("Predicted Class {},Actual Class {}".format(predictions[c], y[c]))
    plt.tight_layout()
    i += 1
plt.show() 

