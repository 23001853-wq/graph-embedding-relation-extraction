# 💊 Trích xuất Tương tác Thuốc (Drug-Drug Interaction) với Seg-GCRN

Dự án này ứng dụng mô hình Deep Learning **Seg-GCRN** (Segment Graph Convolutional Relational Network) để tự động phát hiện và cảnh báo tương tác thuốc (DDI) từ các văn bản y khoa, bệnh án hoặc tài liệu dược lý.

---

## 🏗️ Sơ đồ Kiến trúc Hệ thống (Pipeline Architecture)

Quá trình từ lúc nhập câu bệnh án thô đến khi AI đưa ra quyết định cảnh báo được mô tả qua sơ đồ luồng dữ liệu dưới đây:

```mermaid
graph TD
    A[Văn bản Y khoa & 2 Thuốc] -->|Input| B(SpaCy NLP Engine)
    
    subgraph preprocessing [Tiền Xử Lý NLP - Preprocessing]
    B --> C1[Trích xuất Đồ thị Phụ thuộc <br/> Dependency Parsing]
    B --> C2[Căn chỉnh Thực thể <br/> Entity Alignment]
    B --> C3[Gán nhãn Từ vựng <br/> Tokenization & Vocab]
    end

    C1 -->|Adjacency Matrix| D[Dataloader]
    C2 -->|Pos1, Pos2, Segments| D
    C3 -->|Token IDs| D

    subgraph seg_model [Cỗ máy Seg-GCRN]
    D -->|Shift +120 & Clamp| E[Word & Position Embeddings]
    E --> F[Mạng tích chập Đồ thị <br/> GCN Layers]
    F --> G[Mạng chuỗi thời gian <br/> Bi-LSTM]
    G --> H[Logits Output]
    end

    H -->|Hiệu chuẩn xác suất <br/> Temperature Scaling T=4.0| I(Softmax)
    I --> J{Kết Luận: TRUE / FALSE}
```

## 🚀 Các Điểm Nhấn Công Nghệ & Tối Ưu Hóa (Key Features)

Dự án này được xây dựng theo chuẩn mực đánh giá mô hình của các bài báo khoa học (Paper) quốc tế, khắc phục triệt để các lỗi rò rỉ dữ liệu (Data Leakage) và Overfitting:

- **Stratified Split (Phân tầng dữ liệu):** Dữ liệu được chia Train (80%) và Validation (20%) bằng sklearn đảm bảo tỷ lệ nhãn True/False đồng đều ở mọi tập. Tuyệt đối cô lập tập Test (Unseen Data).
- **Cơ chế Dời trục Vị trí (Position Shift & Clamp):** Vị trí tương đối của từ ngữ được dịch chuyển +120 đơn vị và giới hạn chặt chẽ trong khoảng [1, 249] ngay tại DataLoader, đảm bảo sự đồng nhất tuyệt đối giữa Train và Inference.
- **Hiệu chuẩn Xác suất (Temperature Scaling):** Áp dụng hằng số nhiệt độ $T = 4.0$ vào hàm Softmax ở bước Inference để khắc phục hiện tượng "Tự tin thái quá" (Overconfidence) kinh điển của mạng Nơ-ron, giúp AI đưa ra % tự tin phản ánh đúng độ khó của câu hỏi.
- **Xử lý Mất cân bằng Class (Class Weights):** Sử dụng trọng số phạt [1.0, 6.3] trong hàm Loss để ép mô hình tập trung học các ca có tương tác (TRUE), hạn chế tối đa việc bỏ lọt cảnh báo nguy hiểm.

## 📂 Quy Trình Thực Hiện (Workflow)

**1. Chuẩn bị Dữ liệu (Data Preparation)**
Dữ liệu gốc dạng XML được chuyển đổi sang định dạng JSON gọn nhẹ. Sau đó đi qua pipeline SpaCy để trích xuất ma trận kề (Adjacency Matrix), vị trí tương đối (Pos) và Phân đoạn (Segments).

**2. Huấn Luyện (Training) - train.py**
- Mô hình học trên tập `train_80_split.json`.
- Sau mỗi vòng (Epoch), mô hình thi thử trên tập `val_20_split.json`.
- Cơ chế Checkpoint: Chỉ lưu lại bộ trọng số (Weights) có điểm Validation F1-Score cao nhất vào thư mục `models/best_seg_gcrn.pth`.

**3. Đánh giá Chung cuộc (Evaluation)**
Mô hình tốt nhất được load lại và chấm điểm duy nhất 1 lần trên tập đề thi đại học `test_processed.json`. Hệ thống tự động xuất ra:
- **Classification Report:** Precision, Recall, F1-Score cho từng Class.
- **Confusion Matrix:** Được vẽ và lưu thành ảnh `.png` trực quan hóa các ca đoán sai.

**4. Suy luận Thực tế (Inference) - predict.py**
Môi trường giả lập phòng khám. Cho phép người dùng nhập một câu tiếng Anh bất kỳ chứa tên 2 loại thuốc. AI sẽ quét qua toàn bộ Pipeline một lần duy nhất (Single Forward Pass) trong vài mili-giây và đưa ra cảnh báo True/False kèm độ tự tin đã hiệu chuẩn.

## 💻 Hướng Dẫn Chạy Cục Bộ (How to run)

**Bước 1: Huấn luyện mô hình**
*(Lưu ý: Dữ liệu đã được chia sẵn nằm trong thư mục `data/processed`)*
```bash
python train.py
```

**Bước 2: Chạy Demo dự đoán thực tế**
```bash
python predict.py
```