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
    directories = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) for f in os.listdir(label_directory)]
        for f in file_names:
            img = cv2.imread(f)
            img = cv2.resize(img, (64, 64))
            images.append(img)
            labels.append(ord(d) - ord('A'))  # Chuyển đổi ký tự thành số từ 0-25
    return np.array(images), np.array(labels)

ROOT_PATH_1 = "C:/Users/ADMIN/Desktop/python code/sign_language/new_data/asl_alphabet_train"
train_data_directory_1 = os.path.join(ROOT_PATH_1, "asl_alphabet_train")
images_1, labels_1 = load_data(train_data_directory_1)

ROOT_PATH_2 = "C:/Users/ADMIN/Desktop/python code/sign_language/archive"
train_data_directory_2 = os.path.join(ROOT_PATH_2, "train")
images_2, labels_2 = load_data(train_data_directory_2)

ROOT_PATH_TEST = "C:/Users/ADMIN/Desktop/python code/sign_language/testt"
test_data_directory = os.path.join(ROOT_PATH_TEST, "test")
images_test, labels_test = load_data(test_data_directory)

# Hợp nhất các mảng hình ảnh và nhãn
x_train = np.concatenate((images_1, images_2), axis=0)
y_train = np.concatenate((labels_1, labels_2), axis=0)
x_test = np.array(images_test)
y_test = np.array(labels_test)

# Kiểm tra số lượng nhãn khác nhau
print("Total number of classes:", len(set(y_train)))
print("Label Array:", [chr(X + ord('A')) for X in set(y_train)])

# kiểm tra số phần tử tập train và test
print("The shape of train set: ", x_train.shape, y_train.shape)
print("The shape of test set: ", x_test.shape, y_test.shape)


# Hiển thị một số ảnh
plt.imshow(x_train[0], interpolation='none')
plt.title(f'label: {y_train[0]}')
plt.show()

plt.imshow(x_train[32145], interpolation='none')
plt.title(f'label: {y_train[32145]}')
plt.show()

plt.imshow(x_train[3245], interpolation='none')
plt.title(f'label: {y_train[3245]}')
plt.show()

plt.imshow(x_train[3215], interpolation='none')
plt.title(f'label: {y_train[3215]}')
plt.show()

plt.imshow(x_train[1500], interpolation='none')
plt.title(f'label: {y_train[1500]}')
plt.show()

plt.imshow(x_test[0], interpolation='none')
plt.title(f'label: {y_test[0]}')
plt.show()

plt.imshow(x_test[315], interpolation='none')
plt.title(f'label: {y_test[315]}')
plt.show()

plt.imshow(x_test[1500], interpolation='none')
plt.title(f'label: {y_test[1500]}')
plt.show()

# Chuẩn hóa dữ liệu
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encoding
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.transform(y_test)

# Định hình lại dữ liệu
x_train = x_train.reshape(-1, 64, 64, 3)
x_test = x_test.reshape(-1, 64, 64, 3)

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

print(x_train.size, y_train.size)
print(x_test.size, y_test.size)

# MODEL BUILDING 
my_model = Sequential()
my_model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=(64,64,3)))
my_model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
my_model.add(Dropout(0.5))
my_model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))
my_model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))
my_model.add(Dropout(0.5))
my_model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu'))
my_model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu'))
my_model.add(Flatten())
my_model.add(Dropout(0.5))
my_model.add(Dense(512, activation='relu'))
my_model.add(Dense(24, activation='softmax'))

my_model.summary()

my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
my_model.summary()
my_model.save

history = my_model.fit(datagen.flow(x_train, y_train, batch_size=512), epochs=20, validation_data=(x_test, y_test), shuffle=True)

test_loss, test_accuracy = my_model.evaluate(x_test, y_test)
print('MODEL ACCURACY = {}%'.format(test_accuracy * 100))
print('MODEL LOSS = {}%'.format(test_loss))
my_model.save('my_model.h5')

# Analysis results 
epochs = [i for i in range(20)]
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

predictions = my_model.predict(x_test)
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
print(classification_report(y_test, aaa, target_names = classes))

# Ma trận nhầm lẫn, các hàng là giá trị thực tế, các cột là giá trị dự đoán 
cm = confusion_matrix(y_test, aaa)
cm = pd.DataFrame(cm , index = [i for i in range(25) if i != 9] , columns = [i for i in range(25) if i != 9])
plt.figure(figsize = (15,15))
sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='')
plt.show() 

correct = np.nonzero(aaa == y_test)[0]

i = 0
for c in correct[:6]:
    plt.subplot(3,2,i+1)
    plt.imshow(x_test[c], interpolation='none')
    plt.title("Predicted Class {},Actual Class {}".format(aaa[c], y_test[c]))
    plt.tight_layout()
    i += 1
plt.show()
