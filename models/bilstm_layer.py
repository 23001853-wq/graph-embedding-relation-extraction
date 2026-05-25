import torch
import torch.nn as nn

class BiLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(BiLSTMLayer, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        
    def forward(self, x):
        """
        x: Input features (batch_size, seq_len, input_size)
        Returns: lstm_out (batch_size, seq_len, hidden_size * 2)
        """
        lstm_out, _ = self.lstm(x)
        return lstm_out
