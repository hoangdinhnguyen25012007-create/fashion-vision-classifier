import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN VÀ THAM SỐ
# ==========================================
# Sửa từ DATA_DIR = "data" thành:
DATA_DIR = "NHẬN DẠNG TRANG PHỤC/data"
# Thứ tự các file tương ứng với Label từ 0 đến 9
CLASSES = [
    "ao_thun.npy", "quan_dai.npy", "vay.npy", "ao_khoac.npy", "giay.npy",
    "dep.npy", "vo.npy", "hat.npy", "tui.npy", "quan_ngan.npy"
]
NUM_CLASSES = len(CLASSES)
SAMPLES_PER_CLASS = 15000  # Giới hạn số lượng ảnh mỗi class để tránh tràn RAM

X_data = []
y_data = []

# ==========================================
# 2. TIỀN XỬ LÝ DỮ LIỆU (PREPROCESSING)
# ==========================================
print("--- Đang tải và tiền xử lý dữ liệu ---")
for label, file_name in enumerate(CLASSES):
    file_path = os.path.join(DATA_DIR, file_name)
    
    # Tải dữ liệu từ file .npy
    data = np.load(file_path)
    
    # Chỉ lấy một lượng dữ liệu vừa đủ để train nhanh và mượt
    data = data[:SAMPLES_PER_CLASS]
    
    # Thêm vào danh sách dữ liệu tổng
    X_data.append(data)
    y_data.append(np.full(data.shape[0], label)) # Tạo mảng nhãn tương ứng

# Gộp tất cả các class lại thành mảng numpy lớn
X_data = np.concatenate(X_data, axis=0)
y_data = np.concatenate(y_data, axis=0)

# Reshape dữ liệu về dạng ảnh (28x28x1) và Normalize (chia cho 255.0)
# Dataset Quick Draw lưu dạng mảng phẳng 784 phần tử, cần đưa về 28x28
X_data = X_data.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Chia dữ liệu thành 2 tập: Train (80%) và Validation/Test (20%)
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

print(f"Tổng số ảnh dùng để Train: {X_train.shape[0]}")
print(f"Tổng số ảnh dùng để Validation: {X_val.shape[0]}")

# ==========================================
# 3. XÂY DỰNG MÔ HÌNH CNN (SEQUENTIAL)
# ==========================================
model = Sequential([
    # Lớp Convolution 1: Trích xuất các nét vẽ cơ bản (ngang, dọc, cong)
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    
    # Lớp Convolution 2: Trích xuất các đặc trưng phức tạp hơn của trang phục
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Lớp Convolution 3: Làm mịn và bắt trọn cấu trúc hình vẽ
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Phẳng hóa ma trận ảnh thành mảng 1 chiều để đưa vào lớp Dense
    Flatten(),
    
    # Lớp Fully Connected (Dense) và Dropout chống Overfitting (học vẹt)
    Dense(128, activation='relu'),
    Dropout(0.5), # Tự động ngắt ngẫu nhiên 50% neuron để mô hình không bị phụ thuộc
    
    # Lớp Output: Trả về xác suất của 10 class trang phục
    Dense(NUM_CLASSES, activation='softmax')
])

# Xem cấu trúc tổng quan của mô hình
model.summary()

# ==========================================
# 4. COMPLILE VÀ TRAIN MODEL
# ==========================================
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

print("\n--- Bắt đầu huấn luyện mô hình AI ---")
history = model.fit(X_train, y_train,epochs=10,batch_size=64,validation_data=(X_val, y_val))

# ==========================================
# 5. LƯU MÔ HÌNH VÀ VẼ ĐỒ THỊ KẾT QUẢ
# ==========================================
# Tạo thư mục models nếu chưa có
os.makedirs("models", exist_ok=True)
model_path = "models/fashion_quickdraw.keras"
model.save(model_path)
print(f"\n Đã train xong! Mô hình được lưu thành công tại: {model_path}")

# Vẽ đồ thị độ chính xác (Accuracy) để kiểm tra mô hình
plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Độ chính xác của mô hình qua các Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()