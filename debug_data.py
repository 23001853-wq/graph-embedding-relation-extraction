import json
import os
import random

def debug_sample(file_path, sample_index=None):
    # Đọc file dữ liệu đã xử lý
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Nếu không chỉ định, bốc ngẫu nhiên 1 câu để soi
    if sample_index is None:
        sample_index = random.randint(0, len(data) - 1)
        
    sample = data[sample_index]
    
    print(f"\n========== X-RAY KIỂM TRA DỮ LIỆU (Mẫu số {sample_index}) ==========")
    print(f"ID Câu: {sample['id']}")
    print(f"Nhãn thực tế: {sample['label_name'].upper()} ({sample['label']})")
    print("=" * 68)
    
    tokens = sample['tokens']
    h_pos = sample['h_pos']
    t_pos = sample['t_pos']
    segments = sample['segments']
    pos1 = sample['pos1']
    pos2 = sample['pos2']
    
    print(f"🟢 Thực thể 1 (Head) : '{h_pos['text']}'")
    print(f"   -> Code tìm thấy ở Token Index: [{h_pos['token_start']} đến {h_pos['token_end']}]")
    print(f"🔴 Thực thể 2 (Tail) : '{t_pos['text']}'")
    print(f"   -> Code tìm thấy ở Token Index: [{t_pos['token_start']} đến {t_pos['token_end']}]")
    print("-" * 68)
    
    # In ra bảng đối chiếu Token với các Ma trận Đặc trưng
    print(f"{'IDX':<4} | {'TOKEN (Từ vựng)':<20} | {'SEG':<3} | {'POS1':<4} | {'POS2':<4} | {'GHI CHÚ':<15}")
    print("-" * 68)
    
    for i, token in enumerate(tokens):
        marker = ""
        # Đánh dấu nếu token thuộc về Head
        if h_pos['token_start'] <= i <= h_pos['token_end']:
            marker = "🟢 [HEAD]"
        # Đánh dấu nếu token thuộc về Tail
        elif t_pos['token_start'] <= i <= t_pos['token_end']:
            marker = "🔴 [TAIL]"
            
        print(f"{i:<4} | {token:<20} | {segments[i]:<3} | {pos1[i]:<4} | {pos2[i]:<4} | {marker}")
        
    print("=" * 68)

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(project_root, "data", "processed", "train_processed.json")
    
    if os.path.exists(train_file):
        # Bạn có thể đổi số vào hàm debug_sample(train_file, 10) để xem câu số 10
        debug_sample(train_file)
    else:
        print("Không tìm thấy file train_processed.json!")