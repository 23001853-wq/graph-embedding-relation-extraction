import os
import json
import time
import torch
import spacy
import torch.nn.functional as F

# Import các hàm tiền xử lý của bạn
from preprocessing.parser import align_tokens_to_chars
from preprocessing.segmenter import get_positions, get_segments
from preprocessing.graph_builder import build_adjacency_matrix
from models.seg_gcrn import SegGCRN

# Thêm tham số temperature=4.0 làm mặc định
def predict_ddi(text, drug1, drug2, temperature=4.0):
    print(f"\n[AI ĐANG PHÂN TÍCH (T={temperature})]...")
    print(f"Câu bệnh án: '{text}'")
    print(f"Thuốc 1: {drug1} | Thuốc 2: {drug2}")
    
    # 1. Tìm vị trí ký tự (char_start, char_end) tự động
    h_start = text.find(drug1)
    t_start = text.find(drug2)
    
    if h_start == -1 or t_start == -1:
        print("[LỖI] Không tìm thấy tên thuốc trong câu!")
        return
        
    h_end = h_start + len(drug1)
    t_end = t_start + len(drug2)

    # 2. Tiền xử lý NLP
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    tokens = [token.text for token in doc]
    seq_len = len(tokens)
    
    h_t_start, h_t_end = align_tokens_to_chars(doc, h_start, h_end, drug1)
    t_t_start, t_t_end = align_tokens_to_chars(doc, t_start, t_end, drug2)
    
    if h_t_start == -1 or t_t_start == -1:
        print("[LỖI] SpaCy không thể cắt từ chuẩn xác. Hãy thử câu khác.")
        return

    # Lấy vị trí thô (có số âm)
    pos1_raw = get_positions(seq_len, h_t_start, h_t_end)
    pos2_raw = get_positions(seq_len, t_t_start, t_t_end)
    segments = get_segments(seq_len, h_t_start, h_t_end, t_t_start, t_t_end)
    adj_matrix = build_adjacency_matrix(doc, seq_len)
    
    # 3. Load Vocab và chuyển sang ID
    project_root = os.path.dirname(os.path.abspath(__file__))
    vocab_file = os.path.join(project_root, "data", "processed", "vocab.json")
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        
    token_ids = [vocab.get(t.lower(), 1) for t in tokens] # 1 là <UNK>
    
    # 4. Tiền xử lý vị trí (Giả lập quy trình của DataLoader)
    POSITION_SHIFT = 120
    pos1_shifted = [p + POSITION_SHIFT for p in pos1_raw]
    pos2_shifted = [p + POSITION_SHIFT for p in pos2_raw]

    # Ép kiểu Tensor và Thêm chiều Batch (unsqueeze)
    token_ids_t = torch.tensor([token_ids], dtype=torch.long)
    segments_t = torch.tensor([segments], dtype=torch.long)
    adj_matrix_t = torch.from_numpy(adj_matrix).float().unsqueeze(0)
    masks_t = torch.ones(1, seq_len, dtype=torch.float)
    
    # Clamp cứng giống collate_fn
    pos1_t = torch.clamp(torch.tensor([pos1_shifted]), min=1, max=249)
    pos2_t = torch.clamp(torch.tensor([pos2_shifted]), min=1, max=249)
    
    # 5. KHỞI TẠO VÀ LOAD MODEL (ĐÃ SỬA CHUẨN THAM SỐ)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegGCRN(
        vocab_size=len(vocab),
        word_emb_dim=100,
        pos_emb_dim=25,
        max_pos=250, 
        gcn_hidden=150,
        lstm_hidden=150,
        num_classes=2
    ).to(device)
    
    model_path = os.path.join(project_root, "models", "best_seg_gcrn.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 6. Dự đoán và đo thời gian
    start_infer = time.time()
    with torch.no_grad():
        logits = model(
            token_ids_t.to(device), pos1_t.to(device), pos2_t.to(device), 
            adj_matrix_t.to(device), segments_t.to(device), masks_t.to(device)
        )
        
        # ÁP DỤNG TEMPERATURE SCALING
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=1)[0]
        prediction = torch.argmax(logits, dim=1).item()
        
    end_infer = time.time() 
    process_time_ms = (end_infer - start_infer) * 1000
        
    # 7. In kết quả chuẩn UI/UX
    print("-" * 55)
    if prediction == 1:
        print(f"🚨 KẾT LUẬN: CÓ TƯƠNG TÁC THUỐC (TRUE)")
    else:
        print(f"✅ KẾT LUẬN: KHÔNG CÓ TƯƠNG TÁC (FALSE)")
        
    print(f"   • Độ tự tin AI : {probs[prediction]*100:.1f}%")
    print(f"   • Thời gian xử lý : {process_time_ms:.2f} ms")
    print("-" * 55)

if __name__ == '__main__':
    # THỬ NGHIỆM 1: Câu CÓ tương tác (Aspirin và Warfarin làm tăng nguy cơ chảy máu)
    predict_ddi(
        text="The doctor prescribed aspirin for topical application and amoxicillin to be taken orally in the evening..",
        drug1="aspirin",
        drug2="amoxicillin"
    )
    predict_ddi(
        text="The doctor said that mixing Argatroban and warfarin together would cause them to interact, and taking them together would reduce memory loss.",
        drug1="Argatroban",
        drug2="warfarin"
    )
    # THỬ NGHIỆM 2: Câu KHÔNG CÓ tương tác (Chỉ liệt kê)
    predict_ddi(
        text="The patient was prescribed ibuprofen to relieve headaches and amoxicillin to relieve nausea.",
        drug1="ibuprofen",
        drug2="amoxicillin"
    )