import os
import json

def check_distribution(file_path, file_name):
    if not os.path.exists(file_path):
        print(f"[-] Bỏ qua {file_name:<20}: Không tìm thấy file.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    total = len(data)
    if total == 0:
        print(f"[-] {file_name:<20}: File trống!")
        return
        
    # Đếm số lượng nhãn (hỗ trợ cả key 'label' và 'relation')
    trues = sum(1 for item in data if item.get('label', item.get('relation')) == 1)
    falses = total - trues
    
    # Tính tỷ lệ (False / True)
    ratio = falses / trues if trues > 0 else 0
    percent_true = (trues / total) * 100
    
    print(f"[+] {file_name:<20} | Tổng: {total:<5} | True: {trues:<4} ({percent_true:>4.1f}%) | False: {falses:<4} | Tỷ lệ F/T: {ratio:.2f}")

if __name__ == '__main__':
    # Lấy đường dẫn thư mục gốc một cách an toàn
    project_root = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(project_root, "data", "processed")
    
    print("\n" + "="*80)
    print(" 📊 BÁO CÁO KIỂM TOÁN TỶ LỆ NHÃN (TRUE/FALSE) TRONG CÁC TẬP DỮ LIỆU")
    print("="*80)
    
    # Danh sách các file cần kiểm tra (Dựa theo ảnh bạn gửi)
    files_to_check = [
        "train_processed.json",  # File gốc 100%
        "train_80_split.json",   # File 80%
        "val_20_split.json",     # File 20%
        "test_processed.json"    # File thi đại học
    ]
    
    for filename in files_to_check:
        filepath = os.path.join(processed_dir, filename)
        check_distribution(filepath, filename)
        
    print("="*80 + "\n")