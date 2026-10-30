import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GNNSurvivalModel(nn.Module):
    """
    Multi-State GNN Survival Model for CKD Stage Transitions (Stages 1 through 5):
    - GNN Encoder: 2-layer Graph Attention Network (GAT) capturing patient similarity graph context.
    - Residual Feature Skip Connection: Concatenates node clinical features directly with the GAT
      embedding to preserve individual patient risk signals and treatment effects (resolving Bug 2).
    - Multi-State Hazard Head: Predicts transition hazards conditioned on patient embedding,
      clinical biomarkers, current CKD stage, and discrete time intervals.
    """
    def __init__(self, in_features=15, hidden_dim=32, embedding_dim=16, num_intervals=3, num_heads=2, dropout=0.1):
        super(GNNSurvivalModel, self).__init__()
        self.in_features = in_features
        self.embedding_dim = embedding_dim
        self.num_intervals = num_intervals
        self.dropout = dropout
        
        # GAT Encoder
        self.conv1 = GATConv(in_features, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * num_heads, embedding_dim, heads=1, concat=False, dropout=dropout)
        
        # Residual Hazard Head: [GAT embedding + raw node features + interval one-hot]
        # Input dimension: embedding_dim + in_features + num_intervals
        total_head_input_dim = embedding_dim + in_features + num_intervals
        
        self.hazard_head = nn.Sequential(
            nn.Linear(total_head_input_dim, 32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ELU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x, edge_index, return_attention_weights=False):
        """
        Forward pass computing node embeddings and interval-specific hazards for all patients.
        """
        # 1. GAT Message Passing
        if return_attention_weights:
            x_hidden, att1 = self.conv1(x, edge_index, return_attention_weights=True)
            x_hidden = F.elu(x_hidden)
            x_hidden = F.dropout(x_hidden, p=self.dropout, training=self.training)
            
            emb, att2 = self.conv2(x_hidden, edge_index, return_attention_weights=True)
            attention = {'layer1': att1, 'layer2': att2}
        else:
            x_hidden = self.conv1(x, edge_index)
            x_hidden = F.elu(x_hidden)
            x_hidden = F.dropout(x_hidden, p=self.dropout, training=self.training)
            emb = self.conv2(x_hidden, edge_index)
            attention = None
            
        # 2. Multi-State Hazard Prediction with Residual Feature Connection
        num_nodes = x.size(0)
        hazards_list = []
        
        for step in range(self.num_intervals):
            # One-hot time interval vector: (num_nodes, num_intervals)
            t_oh = torch.zeros((num_nodes, self.num_intervals), device=x.device)
            t_oh[:, step] = 1.0
            
            # Concatenate GAT embedding + individual node features (x) + time representation
            combined_input = torch.cat([emb, x, t_oh], dim=1)
            
            logits = self.hazard_head(combined_input)
            h_step = torch.sigmoid(logits)
            hazards_list.append(h_step)
            
        hazards = torch.cat(hazards_list, dim=1)
        return hazards, emb, attention
