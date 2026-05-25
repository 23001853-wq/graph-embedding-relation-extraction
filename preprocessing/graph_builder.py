# preprocessing/graph_builder.py
import numpy as np

def build_adjacency_matrix(doc, seq_len):
    """Xây dựng Dependency Graph -> Adjacency Matrix (Undirected with Self-loops)"""
    adj = np.zeros((seq_len, seq_len), dtype=int)
    for token in doc:
        if token.head.i != token.i:
            # Undirected edge
            adj[token.i, token.head.i] = 1
            adj[token.head.i, token.i] = 1
            
    # Self-loops (A + I)
    np.fill_diagonal(adj, 1)
    return adj