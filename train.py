import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from dataset.data_loader import get_dataloader
from models.seg_gcrn import SegGCRN

def train_and_evaluate():
    # ==========================================
    # 1. CẤU HÌNH "ÉP XUNG" (HYPERPARAMETERS)
    # ==========================================
    BATCH_SIZE = 32
    EPOCHS = 30           
    LEARNING_RATE = 0.0005 
    WEIGHT_DECAY = 1e-5    
    
    # ==========================================
    # 2. ĐƯỜNG DẪN 3 TẬP DỮ LIỆU ĐỘC LẬP
    # ==========================================
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Thay file train gốc bằng 2 file đã cắt (80/20)
    train_80_file = os.path.join(project_root, "data", "processed", "train_80_split.json")
    val_20_file = os.path.join(project_root, "data", "processed", "val_20_split.json")
    
    # File Test (Tuyệt đối không đụng vào trong lúc Train)
    test_file = os.path.join(project_root, "data", "processed", "test_processed.json")
    vocab_file = os.path.join(project_root, "data", "processed", "vocab.json")
    
    # Đọc vocab_size
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
        vocab_size = len(vocab_data)
        
    print(f"Kích thước từ vựng (Vocab Size): {vocab_size}")

    # ==========================================
    # 3. KHỞI TẠO DATALOADER
    # ==========================================
    print("Đang nạp 3 tập dữ liệu chuẩn Quốc tế (Train/Val/Test)...")
    train_loader = get_dataloader(train_80_file, vocab_file, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(val_20_file, vocab_file, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = get_dataloader(test_file, vocab_file, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang huấn luyện trên thiết bị: {device}")

    model = SegGCRN(
        vocab_size=vocab_size,
        word_emb_dim=100,
        pos_emb_dim=25,
        max_pos=250, 
        gcn_hidden=150,
        lstm_hidden=150,
        num_classes=2
    ).to(device)

    # Khắc phục mất cân bằng dữ liệu
    class_weights = torch.tensor([1.0, 6.3]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_f1 = 0.0
    model_save_path = os.path.join(project_root, "models", "best_seg_gcrn.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    print("\n========== BẮT ĐẦU HUẤN LUYỆN (30 EPOCHS) ==========")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        
        # --- HỌC TRÊN TẬP TRAIN ---
        for batch in train_loader:
            token_ids = batch['token_ids'].to(device)
            pos1 = batch['pos1'].to(device)
            pos2 = batch['pos2'].to(device)
            segments = batch['segments'].to(device)
            adj_matrix = batch['adj_matrix'].to(device)
            masks = batch['masks'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(token_ids, pos1, pos2, adj_matrix, segments, masks)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # --- THI THỬ TRÊN TẬP VALIDATION (Chọn Checkpoint) ---
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader: # <--- ĐÃ SỬA THÀNH VAL_LOADER
                token_ids = batch['token_ids'].to(device)
                pos1 = batch['pos1'].to(device)
                pos2 = batch['pos2'].to(device)
                segments = batch['segments'].to(device)
                adj_matrix = batch['adj_matrix'].to(device)
                masks = batch['masks'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(token_ids, pos1, pos2, adj_matrix, segments, masks)
                preds = torch.argmax(logits, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds, zero_division=0)
        val_rec = recall_score(val_labels, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        print(f"[Epoch {epoch}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Prec: {val_prec:.4f} | Val Rec: {val_rec:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            print(f"  [!] => Đã lưu kỷ lục Val F1 mới: {best_val_f1:.4f}")

    # ==========================================
    # 4. CHẤM ĐIỂM CHUNG CUỘC TRÊN TẬP TEST
    # ==========================================
    print("\n" + "="*60)
    print(" 🚀 ĐÁNH GIÁ CHUNG CUỘC TRÊN TẬP TEST (UNSEEN DATA)")
    print("="*60)
    
    # Load lại não bộ xịn nhất vừa lưu
    best_model = SegGCRN(
        vocab_size=vocab_size, word_emb_dim=100, pos_emb_dim=25,
        max_pos=250, gcn_hidden=150, lstm_hidden=150, num_classes=2
    ).to(device)
    best_model.load_state_dict(torch.load(model_save_path))
    best_model.eval()

    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader: # <--- GIỜ MỚI ĐƯỢC LẤY ĐỀ THI ĐẠI HỌC RA LÀM
            token_ids = batch['token_ids'].to(device)
            pos1 = batch['pos1'].to(device)
            pos2 = batch['pos2'].to(device)
            segments = batch['segments'].to(device)
            adj_matrix = batch['adj_matrix'].to(device)
            masks = batch['masks'].to(device)
            labels = batch['labels'].to(device)
            
            logits = best_model(token_ids, pos1, pos2, adj_matrix, segments, masks)
            preds = torch.argmax(logits, dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_prec = precision_score(test_labels, test_preds, zero_division=0)
    test_rec = recall_score(test_labels, test_preds, zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)

    print(f" FINAL TEST ACCURACY  : {test_acc*100:.2f}%")
    print(f" FINAL TEST PRECISION : {test_prec*100:.2f}%")
    print(f" FINAL TEST RECALL    : {test_rec*100:.2f}%")
    print(f" FINAL TEST F1-SCORE  : {test_f1*100:.2f}%")
    print("="*60)

if __name__ == '__main__':
    train_and_evaluate()