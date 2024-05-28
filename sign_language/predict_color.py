import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

TF_ENABLE_ONEDNN_OPTS=0

# Đọc hình ảnh
image_path = 'C:/Users/ADMIN/Desktop/python code/sign_language/testt/test/L/8.jpg'
image = cv2.imread(image_path)

# Hiển thị hình ảnh gốc
cv2.imshow('Color Image', image)
print('Original image shape:', image.shape)
cv2.waitKey(0)

# (Nếu cần, cắt hình ảnh) Ví dụ:
# image = image[100:700, 150:600]

# Hiển thị hình ảnh đã cắt (nếu có)
cv2.imshow('Cropped Image', image)
print('Cropped image shape:', image.shape)
cv2.waitKey(0)

# Thay đổi kích thước hình ảnh thành 28x28
img_resized = cv2.resize(image, (64, 64))
print('Resized image shape:', img_resized.shape)

# Hiển thị hình ảnh gốc và hình ảnh đã thay đổi kích thước
fig, axes = plt.subplots(1, 2, figsize=(12, 12))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[1].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), interpolation='none')
axes[1].set_title('Resized Image (28x28)')
plt.show()

# Chuẩn bị hình ảnh để dự đoán
img_resized = img_resized.reshape(1, 64, 64, 3)  # Thêm chiều batch
img_resized = img_resized / 255.0  # Chuẩn hóa hình ảnh

# Tải mô hình đã lưu
my_model = tf.keras.models.load_model('my_model.h5')

# Dự đoán lớp của hình ảnh
prediction = my_model.predict(img_resized)
predicted_class = np.argmax(prediction)

print('Prediction probabilities:', prediction)
print('Predicted class:', predicted_class)
