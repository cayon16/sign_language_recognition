
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
TF_ENABLE_ONEDNN_OPTS=0




df=pd.read_csv("sign_language\sign_mnist_train.csv")
train_df = df 
label = train_df['label']
print(train_df.shape)



plt.figure(figsize=(10, 6))
sns.countplot(x = label )
plt.title('Frequency of Each Label in Training Set')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.show()



train_df, test_df = train_test_split(train_df, test_size = 0.2,stratify=train_df['label'], random_state=42)
y_train = train_df['label']
y_train.reset_index(inplace=True, drop=True)
y_test = test_df['label']

y_test.reset_index(inplace=True, drop=True)
y = y_test 
train_df = train_df.drop(['label'],axis=1)
train_df.reset_index(inplace=True, drop=True)
test_df = test_df.drop(['label'],axis=1)
test_df.reset_index(inplace=True, drop=True)

fig, axe = plt.subplots(1,2)
axe[0] = sns.countplot(x = y_train)
axe[1] = sns.countplot(x = y_test)
plt.title('distribution of train and test set')
plt.show()



from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)



print(train_df.shape)
print(test_df.shape)


x_train = train_df.values
x_test = test_df.values
x_train = x_train/255
x_test = x_test/255 

# Reshaping the data from 1-D to 3-D as required through input by CNN's
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

print(x_train.shape , x_test.shape)



# visualize images 
f, ax = plt.subplots(2,5) 
f.set_size_inches(10, 10)
k = 0
for i in range(2):
    for j in range(5):
        ax[i,j].imshow(x_train[k].reshape(28, 28) , cmap = "gray")
        k += 1
    plt.tight_layout()    
plt.show() 




# With data augmentation to prevent overfitting
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


history = model.fit(datagen.flow(x_train, y_train, batch_size=256), epochs=35, validation_data=(x_test, y_test), shuffle = 1) # tráo thứ tự data để batch ngẫu nhiên 
# predictions = cnn.predict(x_test)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('MODEL ACCURACY = {}%'.format(test_accuracy*100))
print('MODEL LOSS = {}%'.format(test_loss))
model.save('model.h5')


# analysis results 
epochs = [i for i in range(35)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(16,9)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'r-o' , label = 'Testing Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()



predictions = model.predict_classes(x_test)
for i in range(len(predictions)):
    if(predictions[i] >= 9):
        predictions[i] += 1
predictions[:5]     


classes = ["Class " + str(i) for i in range(25) if i != 9]
print(classification_report(y, predictions, target_names = classes))

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