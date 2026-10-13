# TỪ XÁC SUẤT DỰ BÁO ĐẾN QUYẾT ĐỊNH (HCMUTE)
> **Mô hình Hồi quy Logistic & Dashboard Tương tác Điều chỉnh Ngưỡng Quyết định**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lyhiahan-mohinh-app-fy7kmr.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/lyhiahan/MoHinh)

---

## 1. Giới thiệu dự án
Dự án nghiên cứu bài toán phân loại nhị phân dự báo khả năng qua môn của học viên dựa trên các chỉ số rèn luyện và học tập. Dự án tập trung giải quyết bài toán:
- Xây dựng mô hình **Hồi quy Logistic (Logistic Regression)** có khả năng diễn giải cao (thông qua hệ số chuẩn hóa $\beta$ và tỷ số Odds Ratio - OR).
- Nghiên cứu tác động của việc **điều chỉnh ngưỡng quyết định (Decision Threshold $\tau$)** từ xác suất dự báo $P(Y=1|X)$ sang quyết định nhị phân.
- Đánh giá sự đánh đổi giữa **Precision**, **Recall lớp Rớt (lớp thiểu số)**, **Macro F1** và **Balanced Accuracy** phục vụ nghiệp vụ cảnh báo sớm sinh viên có nguy cơ rớt môn.

---

## 2. Liên kết trực tuyến
- **Dashboard Online (Streamlit Cloud):** [https://lyhiahan-mohinh-app-fy7kmr.streamlit.app/](https://lyhiahan-mohinh-app-fy7kmr.streamlit.app/)
- **Mã nguồn GitHub:** [https://github.com/lyhiahan/MoHinh](https://github.com/lyhiahan/MoHinh)

---

## 3. Cấu trúc thư mục dự án
```text
MoHinh/
├── app.py                   # Ứng dụng Dashboard Streamlit tương tác thời gian thực
├── notebook.ipynb           # Jupyter Notebook nghiên cứu, huấn luyện & khảo sát ngưỡng
├── du_lieu_sinh_vien.csv    # Bộ dữ liệu 600 sinh viên phục vụ mô phỏng
├── requirements.txt         # Danh sách thư viện Python và phiên bản tối thiểu
├── link_dashboard.txt       # Thông tin đường dẫn truy cập và hướng dẫn chạy cho thầy cô
├── README.md                # Tài liệu hướng dẫn và giới thiệu tổng quan dự án
└── .gitignore               # Cấu hình bỏ qua tệp tạm và bộ nhớ đệm
```

---

## 4. Dữ liệu và Phương pháp nghiên cứu

### 4.1. Đặc trưng dữ liệu (`du_lieu_sinh_vien.csv`)
Tập dữ liệu gồm 600 mẫu quan sát với 4 biến độc lập và 1 biến mục tiêu:
- `gio_hoc_tuan`: Số giờ tự học mỗi tuần ($0.0 - 20.0$ giờ).
- `ty_le_chuyen_can`: Tỷ lệ tham gia lớp học ($30.0 - 100.0\%$).
- `so_bai_tap`: Số bài tập đã hoàn thành ($0 - 10$ bài).
- `diem_giua_ky`: Điểm thi thường kỳ / giữa kỳ ($0.0 - 10.0$ điểm).
- `qua_mon`: Biến mục tiêu nhị phân ($1$: Đạt/Qua môn - 538 sinh viên; $0$: Rớt môn - 62 sinh viên).

> **Lưu ý:** Dữ liệu được mô phỏng toán học phục vụ minh họa phương pháp và nghiên cứu hành vi của ngưỡng phân loại; không đại diện cho hiệu quả dự kiến trên dữ liệu thực tế.

### 4.2. Huấn luyện mô hình
- Dữ liệu được chia theo tỷ lệ **80% Train (480 sinh viên)** và **20% Test (120 sinh viên)** có phân tầng (`stratify=y`).
- Tập Test gồm **108 ca Đạt** và **12 ca Rớt**.
- Chuẩn hóa đặc trưng bằng `StandardScaler` (chỉ `fit` trên tập Train và `transform` sang tập Test).
- Huấn luyện với thuật toán **Logistic Regression** (`random_state=42`).

### 4.3. Quy ước phân loại và đánh giá
- **Lớp dương:** Đạt ($Y=1$).
- **Lớp âm:** Rớt môn / Nguy cơ ($Y=0$).
- Trường hợp thực tế Rớt nhưng dự báo Đạt là **False Positive (FP)** theo quy ước lớp Đạt là lớp dương, tương ứng một ca rớt bị bỏ sót trong nghiệp vụ cảnh báo.
- **Tập test chỉ dùng để đánh giá kiểm chứng cuối cùng**, không dùng để chọn ngưỡng tối ưu (ngưỡng tối ưu thực tế cần dựa trên cross-validation hoặc tập validation riêng).

---

## 5. Hướng dẫn cài đặt và khởi chạy cục bộ (Local)

### Yêu cầu hệ thống
- Python 3.9+ (đã kiểm thử trên Python 3.10 - 3.12).

### Bước 1: Cài đặt thư viện
Mở Terminal / Command Prompt tại thư mục dự án và chạy:
```bash
pip install -r requirements.txt
```

### Bước 2: Khởi chạy Dashboard Streamlit
```bash
streamlit run app.py
```
Hệ thống sẽ tự động mở trình duyệt web tại địa chỉ:
```text
http://localhost:8501
```

### Bước 3: Khởi chạy Notebook (tùy chọn)
```bash
jupyter notebook notebook.ipynb
```
hoặc mở trực tiếp trên VS Code / Jupyter Lab để xem từng bước phân tích và kết quả kiểm nghiệm.

---

## 6. Tác giả & Thông tin bài nộp
- **Sinh viên thực hiện:** lyhiahan
- **Đơn vị:** Trường Đại học Sư phạm Kỹ thuật TP.HCM (HCMUTE)
- **Môn học:** Tương tác dữ liệu / Nhập môn Học máy
