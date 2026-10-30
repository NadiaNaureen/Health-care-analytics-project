"""
MM-GAT-Surv Architecture: Multi-Modal Graph Attention Network with Discrete-Time Survival Modeling
Combines:
1. Clinical Tabular MLP Encoder (15-dim -> 16-dim e_tab)
2. Vision Encoder (ConvNeXt-Tiny -> 32-dim e_img)
3. Cross-Modal Gated Linear Projection Fusion (e_tab + e_img -> 48-dim z_i)
4. 2-layer GAT Encoder (48-dim -> 24-dim h_gat)
5. Residual Multi-Stage Discrete Hazard Head [h_gat + x_clinical + interval_tau -> hazard]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from .vision_encoder import ConvNeXtVisionEncoder
from .fusion import GatedLinearProjectionFusion

class TabularClinicalEncoder(nn.Module):
    """
    Encodes raw clinical biomarkers into a 16-dimensional tabular embedding (e_tab).
    """
    def __init__(self, in_features=15, out_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(32, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU()
        )
        
    def forward(self, x):
        return self.encoder(x)


class MMGATSurvModel(nn.Module):
    """
    Complete MM-GAT-Surv Model:
    - Tabular encoder: 16-dim embedding
    - Vision encoder: 32-dim embedding (with optional ablation mode)
    - Gated Linear Projection fusion: 48-dim unified node vector
    - 2-layer GAT: neighbor context propagation
    - Residual Discrete-Time Hazard Head: combines GAT embedding with raw individual
      clinical biomarkers to prevent information dilution and preserve treatment effects.
    """
    def __init__(self,
                 in_clinical_features=15,
                 tab_dim=16,
                 img_dim=32,
                 fused_dim=48,
                 gat_hidden_dim=24,
                 gat_out_dim=24,
                 num_intervals=3,
                 num_heads=2,
                 dropout=0.1,
                 vision_checkpoint="vision_stage_classifier.pt"):
        super().__init__()
        self.in_clinical_features = in_clinical_features
        self.tab_dim = tab_dim
        self.img_dim = img_dim
        self.fused_dim = fused_dim
        self.num_intervals = num_intervals
        self.dropout = dropout

        # 1. Clinical Tabular Encoder
        self.tab_encoder = TabularClinicalEncoder(in_features=in_clinical_features, out_dim=tab_dim)

        # 2. Vision Encoder (ConvNeXt-Tiny)
        self.vision_encoder = ConvNeXtVisionEncoder(embedding_dim=img_dim, checkpoint_path=vision_checkpoint)

        # 3. Gated Linear Projection Fusion
        self.fusion_layer = GatedLinearProjectionFusion(tab_dim=tab_dim, img_dim=img_dim, fused_dim=fused_dim)

        # 4. 2-layer GAT Encoder
        self.conv1 = GATConv(fused_dim, gat_hidden_dim, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(gat_hidden_dim * num_heads, gat_out_dim, heads=1, concat=False, dropout=dropout)

        # 5. Residual Multi-Stage Discrete Hazard Head
        # Head input: [h_gat (24) + x_clinical (15) + tau_interval_one_hot (3)] = 42 dim
        head_input_dim = gat_out_dim + in_clinical_features + num_intervals

        self.hazard_head = nn.Sequential(
            nn.Linear(head_input_dim, 48),
            nn.LayerNorm(48),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(48, 24),
            nn.ELU(),
            nn.Linear(24, 1)
        )

    def encode_multimodal_nodes(self, x_clinical, x_image, ablate_vision=False):
        """
        Encodes clinical and image inputs into the 48-dim fused embedding space (z_i).
        """
        e_tab = self.tab_encoder(x_clinical)
        e_img = self.vision_encoder(x_image, ablate_vision=ablate_vision)
        z, gate = self.fusion_layer(e_tab, e_img)
        return z, e_tab, e_img, gate

    def forward(self, x_fused, edge_index, x_clinical, return_attention=False):
        """
        Forward pass through GAT and residual hazard head:
        x_fused: [N, 48] fused node embeddings
        edge_index: [2, E] graph topology
        x_clinical: [N, 15] raw clinical features for skip connection
        """
        # 1. GAT Propagation
        if return_attention:
            h1, att1 = self.conv1(x_fused, edge_index, return_attention_weights=True)
            h1 = F.elu(h1)
            h1 = F.dropout(h1, p=self.dropout, training=self.training)
            h_gat, att2 = self.conv2(h1, edge_index, return_attention_weights=True)
            attention = {'layer1': att1, 'layer2': att2}
        else:
            h1 = self.conv1(x_fused, edge_index)
            h1 = F.elu(h1)
            h1 = F.dropout(h1, p=self.dropout, training=self.training)
            h_gat = self.conv2(h1, edge_index)
            attention = None

        # 2. Residual Multi-Stage Hazard Head across intervals
        num_nodes = x_fused.size(0)
        hazards_list = []
        
        for j in range(self.num_intervals):
            # One-hot encoding of discrete time interval tau_j
            interval_oh = torch.zeros(num_nodes, self.num_intervals, device=x_fused.device)
            interval_oh[:, j] = 1.0
            
            # Combine GAT context + raw clinical skip connection + interval
            head_in = torch.cat([h_gat, x_clinical, interval_oh], dim=-1)
            raw_logit = self.hazard_head(head_in)
            h_j = torch.sigmoid(raw_logit)
            hazards_list.append(h_j)
            
        hazards = torch.cat(hazards_list, dim=-1)  # [N, num_intervals]
        
        if return_attention:
            return hazards, h_gat, attention
        return hazards, h_gat

    def predict_survival_and_time(self, x_fused, edge_index, x_clinical, intervals_months=[6, 12, 24]):
        """
        Computes cumulative survival, progression probability, and expected time to progression in months.
        """
        self.eval()
        with torch.no_grad():
            hazards, _ = self.forward(x_fused, edge_index, x_clinical)
            # Cumulative survival S(t_j) = prod_{k<=j} (1 - h_k)
            surv_prob = torch.cumprod(1.0 - hazards.clamp(0.0, 0.999), dim=-1)
            risk_prob = 1.0 - surv_prob
            
            # Expected time to progression (months)
            # P(progression in interval j) = S(t_{j-1}) * h_j
            num_nodes = hazards.shape[0]
            density = torch.zeros_like(hazards)
            prev_s = torch.ones(num_nodes, device=hazards.device)
            
            for j in range(self.num_intervals):
                density[:, j] = prev_s * hazards[:, j]
                prev_s = surv_prob[:, j]
                
            interval_centers = torch.tensor([3.0, 9.0, 18.0], device=hazards.device)
            max_time = float(intervals_months[-1])  # 24 months
            
            # E[T] = sum(center_j * P(event_j)) + max_time * S(max_time)
            expected_time = torch.sum(density * interval_centers, dim=-1) + max_time * surv_prob[:, -1]
            
            return {
                "hazards": hazards.cpu().numpy(),
                "survival_probs": surv_prob.cpu().numpy(),
                "progression_risks": risk_prob.cpu().numpy(),
                "expected_time_months": expected_time.cpu().numpy()
            }
