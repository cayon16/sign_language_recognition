import pandas as pd
import numpy as np

# Đọc dataset gốc
df = pd.read_csv('sign_language\sign_mnist_train.csv')


# Tạo danh sách để lưu các DataFrame phản chiếu
flipped_images = []

# Duyệt qua từng ảnh trong dataset gốc
for index, row in df.iterrows():
    label = row['label']
    image = row.drop('label').values.reshape(28, 28)
    
    # Tạo ảnh phản chiếu
    flipped_image = np.fliplr(image)
    
    # Chuyển ảnh phản chiếu thành 1D array và thêm label
    flipped_image_flat = flipped_image.flatten()
    flipped_image_series = pd.Series(flipped_image_flat)
    flipped_image_series['label'] = label
    
    # Thêm ảnh phản chiếu vào danh sách dưới dạng DataFrame
    flipped_images.append(flipped_image_series.to_frame().T)

# Kết hợp các DataFrame trong danh sách thành một DataFrame duy nhất
flipped_df = pd.concat(flipped_images, ignore_index=True)

# Đổi tên các cột cho phù hợp
columns = list(range(784))  # Tạo danh sách từ 0 đến 783
columns.append('label')  # Thêm cột 'label'
flipped_df.columns = columns

# Kết hợp DataFrame gốc và DataFrame ảnh phản chiếu
combined_df = pd.concat([df, flipped_df], ignore_index=True)

# Lưu DataFrame kết hợp thành file CSV mới
combined_df.to_csv('combined_sign_mnist.csv', index=False)

print("Dataset kết hợp đã được lưu thành công!")
