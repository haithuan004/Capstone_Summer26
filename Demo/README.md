# Action Recognition Demo (ST-GCN)

Thư mục này chứa toàn bộ các file tối thiểu cần thiết để teammate của bạn ráp đường ống (pipeline) và chạy Demo khả năng dự đoán hành động thông qua skeleton.

## Thành phần của thư mục (Files)

1. `best_model.pth`
   - File trọng số (weights) của mô hình tốt nhất đã được huấn luyện với các kỹ thuật giảm overfitting và lấy cân bằng lớp (oversampling + focal loss).

2. `stgcn_model.py`
   - Chứa cấu trúc mạng nơ-ron cục bộ (Architecture) của thuật toán ST-GCN. Đây là file định nghĩa cấu trúc code chính yếu để load được model. (KHÔNG ĐƯỢC XÓA)

3. `best_metrics_report.txt`
   - Báo cáo chi tiết về F1-score, Precision, Recall và Matrix nhầm lẫn ở fold hiệu năng tốt nhất để show ra độ chuẩn xác cho người xem Demo có thể đánh giá tham khảo.

4. `inference_demo.py`
   - **Tài liệu chạy nhanh (Quick Start Demo!)**: Chứa mã nguồn ngắn gọn và dễ hiểu nhất chỉ rõ cách nạp model và cách đưa 1 input skeleton vào để dự đoán hành động.

## Input của mô hình là gì?

Teammate của bạn khi chạy Demo thực tế (với MediaPipe hay YOLO-Pose) cần lưu ý cấp đúng format Input cho model:
- Mô hình yêu cầu Tensor chứa toạ độ dạng: `(Batch, C, T, V)`
  - **Batch**: Có thể là `1` mẫu mỗi lần lấy inference time.
  - **C (Channels) = 3**: `[Toạ_độ_X, Toạ_độ_Y, Độ_tự_tin_Confidence]`.
  - **T (Thời gian) = 100**: Nghĩa là clip hành động bao quát 100 frames (sliding window clip len 100). (Nếu clip hơi ngắn hơn, hãy pad bằng 0 vào phần bù hoặc clone frame cuối).
  - **V (Khớp nối) = 17**: Mã hoá của chuẩn 17 keypoints theo COCO. (ví dụ như Mũi, Vai trái, Khuỷu trái, v.v).

Classes nhận diện gồm:
0: `Standing`
1: `Walking`
2: `Sitting`
3: `Falling`

## Cài đặt thư viện yêu cầu

```bash
pip install torch numpy
```

## Chạy thử Demo (Mock Data)

Lệnh sau sẽ chạy script demo với dữ liệu dummy để chứng minh việc nạp weights trơn tru:
```bash
python inference_demo.py
```
