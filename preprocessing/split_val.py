import os
import json
import re
from sklearn.model_selection import train_test_split

# ==========================================
# 1. ĐỊNH NGHĨA HÀM LƯU FILE ĐẸP (GIỮ MẢNG NẰM NGANG)
# ==========================================
def save_beautiful_json(data, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        # Bước 1: Xuất JSON với indent=4 để thụt lề cấu trúc ngoài
        json_str = json.dumps(data, ensure_ascii=False, indent=4)
        
        # Bước 2: Dùng Regex "bóp" các mảng số nguyên lại cho nằm trên 1 dòng
        json_str = re.sub(
            r'\[\s*([-\d,\s]+)\s*\]', 
            lambda m: '[' + re.sub(r'\s+', '', m.group(1)) + ']', 
            json_str
        )
        f.write(json_str)

if __name__ == '__main__':
    # Lùi lại 1 cấp thư mục từ vị trí của file split_val.py
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # ==========================================
    # 2. KHAI BÁO ĐƯỜNG DẪN (AN TOÀN TUYỆT ĐỐI)
    # ==========================================
    # File gốc (Chỉ đọc)
    train_file = os.path.join(project_root, "data", "processed", "train_processed.json")
    
    # Hai file sinh ra mới (Ghi)
    train_80_file = os.path.join(project_root, "data", "processed", "train_80_split.json")
    val_20_file = os.path.join(project_root, "data", "processed", "val_20_split.json")

    print(f"Đang đọc file gốc từ: {train_file} ...")
    
    # ==========================================
    # 3. ĐỌC FILE GỐC (Chỉ dùng lệnh 'r' -> Không bao giờ làm hỏng file gốc)
    # ==========================================
    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            full_train_data = json.load(f)
    except FileNotFoundError:
        print(f" LỖI: Không tìm thấy file {train_file}. Bạn kiểm tra lại đường dẫn nhé!")
        exit(1)

    # ==========================================
    # 4. CHIA STRATIFIED 80/20 (Đảm bảo tỷ lệ True/False)
    # ==========================================
    print("Đang tiến hành cắt dữ liệu 80/20...")
    
    # Lấy danh sách nhãn để thư viện sklearn biết đường chia đều
    labels = [item.get('label', item.get('relation')) for item in full_train_data]

    train_data, val_data = train_test_split(
        full_train_data, 
        test_size=0.2, 
        random_state=42, # Chốt seed để chạy 100 lần vẫn ra đúng kết quả này
        stratify=labels
    )

    # ==========================================
    # 5. LƯU 2 FILE MỚI XUỐNG Ổ CỨNG
    # ==========================================
    print("Đang lưu dữ liệu ra 2 file mới với định dạng chuẩn...")
    save_beautiful_json(train_data, train_80_file)
    save_beautiful_json(val_data, val_20_file)

    # ==========================================
    # 6. IN BÁO CÁO NGHIỆM THU
    # ==========================================
    print("\n✅ HOÀN TẤT! BÁO CÁO PHÂN BỔ DỮ LIỆU:")
    print("-" * 50)
    for name, data in [("Train (80%)", train_data), ("Validation (20%)", val_data)]:
        total = len(data)
        trues = sum(1 for item in data if item.get('label', item.get('relation')) == 1)
        falses = total - trues
        print(f"Tập {name:<18}: {total} câu (True: {trues} / False: {falses})")
    print("-" * 50)
    print("Bạn có thể vào thư mục data/processed/ để kiểm tra 3 file đang nằm cạnh nhau!")