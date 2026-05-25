import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn_layer import GCNLayer
from .bilstm_layer import BiLSTMLayer

class SegGCRN(nn.Module):
    def __init__(self, vocab_size, word_emb_dim=100, pos_emb_dim=25, max_pos=250, gcn_hidden=150, lstm_hidden=150, num_classes=2):
        super(SegGCRN, self).__init__()
        
        # ==========================================
        # BƯỚC 1: Tầng Embedding (Từ vựng + 2 Vị trí)
        # ==========================================
        self.word_emb = nn.Embedding(vocab_size, word_emb_dim, padding_idx=0)
        # max_pos định nghĩa biên giới lớn nhất có thể của khoảng cách sau khi đã Dời trục (POSITION_SHIFT)
        self.pos1_emb = nn.Embedding(max_pos, pos_emb_dim, padding_idx=0)
        self.pos2_emb = nn.Embedding(max_pos, pos_emb_dim, padding_idx=0)
        
        # Tổng số kênh của 1 Token = Word + Pos1 + Pos2
        total_emb_dim = word_emb_dim + 2 * pos_emb_dim
        
        # ==========================================
        # BƯỚC 2: Tầng GCN (Graph Convolutional Network)
        # ==========================================
        # Nhúng thông tin quan hệ ngữ pháp (Dependency Tree) vào các từ
        self.gcn1 = GCNLayer(total_emb_dim, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_hidden) # Xếp 2 lớp để hấp thụ ngữ cảnh sâu hơn
        
        # ==========================================
        # BƯỚC 3: Tầng BiLSTM (Truyền vế tuần tự)
        # ==========================================
        # Nhập output của GCN vào LSTM để thu thập đặc trưng chuỗi theo thời gian
        self.bilstm = BiLSTMLayer(
            input_size=gcn_hidden, 
            hidden_size=lstm_hidden, 
            num_layers=1
        )
        
        # BiLSTM đi 2 chiều nên output sẽ gấp đôi số kênh
        lstm_out_dim = lstm_hidden * 2
        
        # ==========================================
        # BƯỚC 4: Tầng Segmented Pooling (Khởi tạo kích thước cho FFN)
        # ==========================================
        # 5 đoạn: Left, E1, Middle, E2, Right
        pooled_dim = lstm_out_dim * 5
        
        # ==========================================
        # BƯỚC 5 & 6: Tầng Linear + Softmax (Phân loại)
        # ==========================================
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(pooled_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
            # Hàm Softmax sẽ được kích hoạt ngầm nếu dùng chung Loss là CrossEntropyLoss
        )

    def forward(self, token_ids, pos1, pos2, adj_matrix, segments, masks=None):
        """
        Flow chạy dữ liệu (Forward Pass)
        """
        # ------------------------------------------
        # 1. EMBEDDING
        # ------------------------------------------
        w_emb = self.word_emb(token_ids)    # (batch, seq_len, word_dim)
        p1_emb = self.pos1_emb(pos1)        # (batch, seq_len, pos_dim)
        p2_emb = self.pos2_emb(pos2)        # (batch, seq_len, pos_dim)
        
        # Nối (Concat) cả 3 embedding vào chung 1 trục (dim=-1)
        x = torch.cat([w_emb, p1_emb, p2_emb], dim=-1)
        
        if masks is not None:
            x = x * masks.unsqueeze(-1)
            
        # ------------------------------------------
        # 2. GCN
        # ------------------------------------------
        x = self.gcn1(x, adj_matrix)
        x = self.gcn2(x, adj_matrix)
        
        if masks is not None:
            x = x * masks.unsqueeze(-1)
            
        # ------------------------------------------
        # 3. BiLSTM
        # ------------------------------------------
        lstm_out = self.bilstm(x) # (batch, seq_len, lstm_out_dim)
        
        # ------------------------------------------
        # 4. SEGMENTED MAX-POOLING
        # ------------------------------------------
        # Lấy đặc trưng Max Pooling chia theo 5 khúc văn bản
        pooled_segments = []
        for seg_id in range(1, 6): # ID từ 1 đến 5 (1: left, 2: E1, 3: md, 4: E2, 5: right)
            
            # Tạo ma trận True/False (1 / 0) lọc đúng segment đang xét
            seg_mask = (segments == seg_id).float()
            
            # Để dùng Max-Pooling chính xác: 
            # Những token KHI bị loại trừ khỏi mask phải biến thành âm vô cực (-1e9) 
            # để hàm Max bỏ qua chúng.
            inverted_mask = (1.0 - seg_mask).unsqueeze(-1) * (-1e9)
            
            # Đắp mask vô cực vào LSTM Tensor
            masked_lstm_out = lstm_out + inverted_mask 
            
            # Trích xuất Véc tơ lớn nhất dọc theo trục chiều dài câu (dim=1)
            seg_pooled, _ = torch.max(masked_lstm_out, dim=1)
            
            # Xử lý an toàn: Nếu câu hoàn toàn không có segment này (vd: Right trống vì E2 ở cuối câu)
            # Thì trả về toàn số 0 thay vì âm vô cực (-inf)
            empty_flag = (seg_mask.sum(dim=1) == 0).unsqueeze(-1)
            seg_pooled = seg_pooled.masked_fill(empty_flag, 0.0)
            
            pooled_segments.append(seg_pooled)
            
        # Nối cả 5 mảng đặc trưng lại (Concat theo dim=1)
        # Kích thước = 5 * lstm_out_dim
        pooled_out = torch.cat(pooled_segments, dim=1)
        
        # ------------------------------------------
        # 5. Phân Loại (CLASSIFICATION)
        # ------------------------------------------
        logits = self.classifier(pooled_out)
        
        return logits
