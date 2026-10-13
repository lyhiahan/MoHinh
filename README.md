# Bài tập: Từ xác suất dự báo đến quyết định

Bài tập xây dựng mô hình **Hồi quy Logistic (Logistic Regression)** và **Dashboard Streamlit tương tác** nhằm khảo sát ảnh hưởng của ngưỡng phân loại (Decision Threshold) đối với bài toán dự báo sinh viên qua môn / cảnh báo nguy cơ rớt môn.

---

## 1. Liên kết Dashboard
- **Dashboard Online:** [https://lyhiahan-mohinh-app-fy7kmr.streamlit.app/](https://lyhiahan-mohinh-app-fy7kmr.streamlit.app/)
- **GitHub Repository:** [https://github.com/lyhiahan/MoHinh](https://github.com/lyhiahan/MoHinh)

---

## 2. Cấu trúc thư mục
- `app.py`: Ứng dụng Dashboard Streamlit tương tác thời gian thực.
- `notebook.ipynb`: Huấn luyện mô hình, giải thích hệ số và khảo sát ngưỡng.
- `du_lieu_sinh_vien.csv`: Dữ liệu mô phỏng 600 sinh viên (538 Đạt, 62 Rớt).
- `requirements.txt`: Danh sách thư viện cần thiết.
- `link_dashboard.txt`: Đường dẫn truy cập dashboard.

---

## 3. Cách chạy trên máy (Local)

1. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Khởi chạy Dashboard:**
   ```bash
   streamlit run app.py
   ```
   Sau khi chạy, mở trình duyệt tại địa chỉ: `http://localhost:8501`.
