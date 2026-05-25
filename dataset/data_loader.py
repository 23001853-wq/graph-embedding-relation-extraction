import json
import torch
from torch.utils.data import Dataset, DataLoader
try:
    from .vocab import Vocab
except ImportError:
    from vocab import Vocab

class DDIDataset(Dataset):
    def __init__(self, processed_file, vocab_file=None):
        """
        Khởi tạo dataset
        """
        with open(processed_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

def collate_fn(batch):
    """
    Padding batch cho bằng chiều dài seq lớn nhất trong batch.
    - token_ids: pad bằng 0
    - pos1, pos2: pad bằng 0 
    - segments: pad bằng 0
    - adj_matrix: zero-pad ma trận vuông
    """
    max_len = max([len(b['token_ids']) for b in batch])
    batch_size = len(batch)
    
    token_ids_pad = torch.zeros(batch_size, max_len, dtype=torch.long)
    pos1_pad = torch.zeros(batch_size, max_len, dtype=torch.long)
    pos2_pad = torch.zeros(batch_size, max_len, dtype=torch.long)
    segments_pad = torch.zeros(batch_size, max_len, dtype=torch.long)
    adj_pad = torch.zeros(batch_size, max_len, max_len, dtype=torch.float) # float cho GCN
    labels = torch.zeros(batch_size, dtype=torch.long)
    
    masks = torch.zeros(batch_size, max_len, dtype=torch.float)
    
    # --- ĐÃ SỬA: Chốt cứng dời trục 120 ở đây ---
    POSITION_SHIFT = 120 
    
    for i, b in enumerate(batch):
        seq_len = len(b['token_ids'])
        
        token_ids_pad[i, :seq_len] = torch.tensor(b['token_ids'])
        
        # --- ĐÃ SỬA: Dời trục và Clamp chặt chẽ ---
        # Cộng POSITION_SHIFT để khử số âm
        p1_tensor = torch.tensor(b['pos1']) + POSITION_SHIFT
        p2_tensor = torch.tensor(b['pos2']) + POSITION_SHIFT
        
        # Ép giá trị nằm trong khoảng [1, 249]. 
        # Cấm không cho bằng 0 vì 0 đã được dùng làm số đại diện cho Zero-Padding ở trên.
        pos1_pad[i, :seq_len] = torch.clamp(p1_tensor, min=1, max=249)
        pos2_pad[i, :seq_len] = torch.clamp(p2_tensor, min=1, max=249)
        # ----------------------------------------
        
        segments_pad[i, :seq_len] = torch.tensor(b['segments'])
        
        # Ma trận kề GCN cắt theo seq_len x seq_len
        adj_pad[i, :seq_len, :seq_len] = torch.tensor(b['adj_matrix'])
        
        labels[i] = b['label']
        if 'attention_mask' in b:
            masks[i, :seq_len] = torch.tensor(b['attention_mask'], dtype=torch.float)
        else:
            masks[i, :seq_len] = 1.0 # Fallback

        
    return {
        'token_ids': token_ids_pad,
        'pos1': pos1_pad,
        'pos2': pos2_pad,
        'segments': segments_pad,
        'adj_matrix': adj_pad,
        'masks': masks,
        'labels': labels
    }

def get_dataloader(processed_file, vocab_file, batch_size=32, shuffle=True):
    dataset = DDIDataset(processed_file, vocab_file)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn
    )
    return dataloader

if __name__ == '__main__':
    import os
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Test DataLoader
    train_file = os.path.join(project_root, "data", "processed", "train_processed.json")
    vocab_file = os.path.join(project_root, "data", "processed", "vocab.json")
    
    # Require train_processed.json and vocab.json to be created first
    if os.path.exists(train_file) and os.path.exists(vocab_file):
        loader = get_dataloader(train_file, vocab_file, batch_size=4)
        print("Tạo dữ liệu kiểm tra loader thành công!")
        for batch in loader:
            print("Token IDs shape:", batch['token_ids'].shape)
            print("Adj Matrix shape:", batch['adj_matrix'].shape)
            print("Segments shape:", batch['segments'].shape)
            print("Pos1 shape (clamped):", batch['pos1'].shape)
            print("Labels:", batch['labels'])
            break
    else:
        print("Xin hãy chạy code xử lý nlp_pipeline.py và vocab.py trước khi test dataloader!")