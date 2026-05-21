# fashion-vision-classifier

## CÁCH CHẠY CHƯƠNG TRÌNH

### Bước 1: Cài thư viện
pip install -r requirements.txt

### Bước 2: Tải data từ Google Drive
Tải folder data tại: [https://drive.google.com/YOUR_FOLDER_LINK](https://drive.google.com/drive/folders/1ZdUuFDod7-2fUhyMJWVmbRo74PSoKjhe?usp=sharing)

Giải nén và đặt vào thư mục project sao cho đúng cấu trúc:
NHẬN DANG TRANG PHỤC/
├── data/
│   ├── ao_khoac.npy
│   ├── ao_thun.npy
│   ├── dep.npy
│   ├── giay.npy
│   ├── hat.npy
│   ├── quan_dai.npy
│   ├── quan_ngan.npy
│   ├── tui.npy
│   ├── vay.npy
│   └── vo.npy
├── models/
├── epp.py
├── train.py
└── requirements.txt

### Bước 3: Train model
sử dụng luôn models đã có sẵn ở drive trên , nếu lỗi thì chạy file train.py một lần nữa

### Bước 4: Chạy ứng dụng
python epp.py

VÌ FILE DATA VÀ MODEL QUÁ NẶNG NÊN EM KHÔNG THỂ UP TRỰC TIẾP LÊN GITHUB ĐƯỢC, PHIỀN THẦY VÀO DRIVE RỒI TẢI VỀ MỚI CHẠY ĐƯỢC Ạ .
EM CẢM ƠN THẦY NHIỀU !!!
