import json
import os

class Vocab:
    def __init__(self):
        # Index 0 is PAD, Index 1 is UNK
        self.word2id = {"<PAD>": 0, "<UNK>": 1}
        self.id2word = {0: "<PAD>", 1: "<UNK>"}
        
    def add_word(self, word):
        if word not in self.word2id:
            idx = len(self.word2id)
            self.word2id[word] = idx
            self.id2word[idx] = word
            
    def get_id(self, word):
        return self.word2id.get(word, self.word2id["<UNK>"])
        
    def __len__(self):
        return len(self.word2id)
        
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.word2id, f, ensure_ascii=False, indent=2)
            
    def load(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.word2id = json.load(f)
            self.id2word = {int(v): k for k, v in self.word2id.items()}

def build_vocab_and_update(train_file, test_file, output_vocab_file):
    print(f"Building vocab from {train_file}...")
    vocab = Vocab()
    
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
        
    for sample in train_data:
        # Lowercase words before adding to generic vocab
        # Consider whether you want word embeddings to track casing or not
        for token in sample['tokens']:
            vocab.add_word(token.lower())
            
    vocab.save(output_vocab_file)
    print(f"Vocab size: {len(vocab)}")
    print(f"Saved vocab to: {output_vocab_file}")

    # Ghi đè token_ids trở lại file JSON
    import re
    print(f"Đang ghi đè token_ids và tokens (chữ thường) vào file {train_file}...")
    for sample in train_data:
        # Chuyển token_ids
        sample['token_ids'] = [vocab.get_id(t.lower()) for t in sample['tokens']]
        # Chuyển thẳng mảng tokens thành chữ thường luôn trong JSON
        sample['tokens'] = [t.lower() for t in sample['tokens']]
        
    with open(train_file, 'w', encoding='utf-8') as f:
        json_str = json.dumps(train_data, ensure_ascii=False, indent=2)
        json_str = re.sub(r'\[\s*([-\d,\s]+)\s*\]', lambda m: '[' + re.sub(r'\s+', '', m.group(1)) + ']', json_str)
        f.write(json_str)
        
    if os.path.exists(test_file):
        print(f"Đang ghi đè token_ids và tokens (chữ thường) vào file {test_file}...")
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        for sample in test_data:
            sample['token_ids'] = [vocab.get_id(t.lower()) for t in sample['tokens']]
            sample['tokens'] = [t.lower() for t in sample['tokens']]
        with open(test_file, 'w', encoding='utf-8') as f:
            json_str = json.dumps(test_data, ensure_ascii=False, indent=2)
            json_str = re.sub(r'\[\s*([-\d,\s]+)\s*\]', lambda m: '[' + re.sub(r'\s+', '', m.group(1)) + ']', json_str)
            f.write(json_str)

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_file = os.path.join(project_root, "data", "processed", "train_processed.json")
    test_file = os.path.join(project_root, "data", "processed", "test_processed.json")
    vocab_file = os.path.join(project_root, "data", "processed", "vocab.json")
    build_vocab_and_update(train_file, test_file, vocab_file)
