# preprocessing/nlp_pipeline.py
import json
import os
import re
import spacy
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Import các module đã được bóc tách
try:
    from parser import align_tokens_to_chars
    from segmenter import get_positions, get_segments
    from graph_builder import build_adjacency_matrix
except ImportError:
    from .parser import align_tokens_to_chars
    from .segmenter import get_positions, get_segments
    from .graph_builder import build_adjacency_matrix

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Mô hình spaCy 'en_core_web_sm' chưa được tải. \nVui lòng chạy lệnh: python -m spacy download en_core_web_sm")
    exit(1)

def process_data(input_file):
    print(f"Bắt đầu xử lý file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    processed_data = []
    valid_count = 0
    invalid_count = 0
    
    for sample in tqdm(data, desc=os.path.basename(input_file)):
        text = sample['text']
        doc = nlp(text)
        
        tokens = [token.text for token in doc]
        seq_len = len(tokens)
        if seq_len == 0:
            invalid_count += 1
            continue
            
        # 1. Map Char -> Token index (Dùng file parser)
        h_start, h_end = align_tokens_to_chars(doc, sample['h_entity']['char_start'], sample['h_entity']['char_end'], sample['h_entity']['text'])
        t_start, t_end = align_tokens_to_chars(doc, sample['t_entity']['char_start'], sample['t_entity']['char_end'], sample['t_entity']['text'])
        
        if h_start == -1 or t_start == -1:
            invalid_count += 1
            continue
            
        # 2 & 3. Features và Segments (Dùng file segmenter)
        pos1 = get_positions(seq_len, h_start, h_end)
        pos2 = get_positions(seq_len, t_start, t_end)
        segments = get_segments(seq_len, h_start, h_end, t_start, t_end)
        
        # 4. Dependency Graph (Dùng file graph_builder)
        adj = build_adjacency_matrix(doc, seq_len)
        
        # 5. Label
        relation = sample.get('relation', 'false').lower().strip()
            
        if relation == 'false' or not relation:
            label_val = 0
            label_name = "false"
        else:
            # Các nhãn tương tác thật (effect, mechanism, advise, int...)
            label_val = 1
            label_name = "true"
            
        processed_data.append({
            "id": sample['id'],
            "sentence_id": sample['sentence_id'],
            "document_id": sample.get('document_id', ''),
            "tokens": tokens,
            "token_ids": [],
            "attention_mask": [1] * seq_len,
            "h_pos": {
                "text": sample['h_entity']['text'],
                "token_start": h_start,
                "token_end": h_end
            },
            "t_pos": {
                "text": sample['t_entity']['text'],
                "token_start": t_start,
                "token_end": t_end
            },
            "pos1": pos1,
            "pos2": pos2,
            "segments": segments,
            "adj_matrix": adj.tolist(),
            "label": label_val,
            "label_name": label_name
        })
        valid_count += 1
        
    print(f"Xử lý xong {os.path.basename(input_file)}! Hợp lệ: {valid_count}, Lỗi index: {invalid_count}")
    return processed_data

def save_json(data, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        # Regex để làm gọn mảng số nguyên trên 1 dòng cho file JSON dễ đọc
        json_str = re.sub(r'\[\s*([-\d,\s]+)\s*\]', lambda m: '[' + re.sub(r'\s+', '', m.group(1)) + ']', json_str)
        f.write(json_str)
    print(f"Đã lưu vào: {output_file}")

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Chỉ đọc đúng 1 file dataset tổng
    dataset_in = os.path.join(project_root, "data", "dataset.json")
    
    train_out = os.path.join(project_root, "data", "processed", "train_processed.json")
    test_out = os.path.join(project_root, "data", "processed", "test_processed.json")
    
    if not os.path.exists(dataset_in):
        print(f"[LỖI] Không tìm thấy {dataset_in}. Hãy chạy file chuyển đổi XML sang JSON trước!")
        exit(1)
        
    # 1. Đọc và xử lý dữ liệu
    all_processed_data = process_data(dataset_in)
        
    print(f"\nTổng số lượng câu dữ liệu thu thập được: {len(all_processed_data)}")
    
    if len(all_processed_data) == 0:
        print("Không có dữ liệu hợp lệ để chia. Kết thúc.")
        exit(1)
    
    # 2. Rút trích danh sách nhãn để làm cơ sở chia phân tầng
    labels = [item['label'] for item in all_processed_data]
    
    # 3. Chia Data (80% Train, 20% Test) đảm bảo tỷ lệ True/False không đổi
    print("\nĐang tiến hành trộn và chia phân tầng (Stratified Split)...")
    train_data, test_data = train_test_split(
        all_processed_data, 
        test_size=0.2, 
        random_state=42, # Đóng băng seed
        stratify=labels
    )
    
    # 4. Lưu ra file
    save_json(train_data, train_out)
    save_json(test_data, test_out)
    
    # 5. In báo cáo thống kê để nghiệm thu
    print("\n========== BÁO CÁO PHÂN BỔ DỮ LIỆU ==========")
    for name, data in [("Train", train_data), ("Test", test_data)]:
        total = len(data)
        trues = sum(1 for item in data if item['label'] == 1)
        falses = total - trues
        ratio = falses / trues if trues > 0 else 0
        print(f"Tập {name}: {total} câu")
        print(f"  -> False (0): {falses} | True (1): {trues}")
        print(f"  -> Tỷ lệ False/True: {ratio:.2f}")
    print("=============================================\n")