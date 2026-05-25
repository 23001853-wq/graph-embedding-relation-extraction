import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        """
        x: Token embeddings, shape (batch_size, seq_len, in_features)
        adj: Ma tran ke Adjacency matrix, shape (batch_size, seq_len, seq_len)
        """
        out = torch.bmm(adj, x)
        out = self.linear(out)
        out = F.relu(out)
        return out
