import torch
import torch.nn as nn
import numpy as np

class SpatialGraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        # x shape: [batch, nodes, features]
        support = self.linear(x)
        # Spatial aggregation via adjacency matrix multiplication
        out = torch.matmul(adj, support)
        return torch.relu(out)

class STGCN_LSTM_Predictor(nn.Module):
    def __init__(self, node_count=8, in_features=5, hidden_dim=32):
        super().__init__()
        self.gcn = SpatialGraphConv(in_features, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1) # Predicts +12hr displacement (mm)

    def forward(self, x_seq, adj):
        # x_seq shape: [batch, seq_len, nodes, features]
        batch, seq_len, nodes, feats = x_seq.shape
        gcn_outputs = []
        
        # Spatial Graph Convolution over time
        for t in range(seq_len):
            gcn_t = self.gcn(x_seq[:, t, :, :], adj)
            gcn_outputs.append(gcn_t)
            
        gcn_seq = torch.stack(gcn_outputs, dim=1) # [batch, seq_len, nodes, hidden_dim]
        gcn_seq = gcn_seq.view(batch * nodes, seq_len, -1)
        
        # Temporal LSTM Sequence Processing
        lstm_out, _ = self.lstm(gcn_seq)
        
        # Linear projection on final time step
        preds = self.fc(lstm_out[:, -1, :])
        return preds.view(batch, nodes)

def run_dl_inference(telemetry_history, adj_matrix):
    """Generates 12-hour future subsidence predictions."""
    model = STGCN_LSTM_Predictor()
    model.eval()
    
    # Format incoming telemetry tensor: [1 batch, 24 time steps, 8 nodes, 5 features]
    x_tensor = torch.tensor(telemetry_history, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        predicted_subsidence = model(x_tensor, adj_matrix).squeeze(0).numpy()
        
    return predicted_subsidence
