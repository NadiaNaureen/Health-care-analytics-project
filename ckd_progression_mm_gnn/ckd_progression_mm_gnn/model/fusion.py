"""
Cross-Modal Fusion Module for MM-GAT-Surv
Combines 16-dim clinical tabular embedding (e_tab) and 32-dim visual embedding (e_img)
using a Gated Linear Projection Layer that learns modality weighting,
producing a 48-dimensional unified multi-modal patient node vector (z_i).
"""

import torch
import torch.nn as nn

class GatedLinearProjectionFusion(nn.Module):
    def __init__(self, tab_dim=16, img_dim=32, fused_dim=48):
        super().__init__()
        self.tab_dim = tab_dim
        self.img_dim = img_dim
        self.fused_dim = fused_dim
        
        # Individual modal linear projections
        self.fc_tab = nn.Linear(tab_dim, fused_dim)
        self.fc_img = nn.Linear(img_dim, fused_dim)
        
        # Cross-modal gating layer: outputs per-dimension gate weights in [0, 1]
        concat_dim = tab_dim + img_dim
        self.gate_fc = nn.Sequential(
            nn.Linear(concat_dim, fused_dim),
            nn.Sigmoid()
        )
        
        # Skip linear projection
        self.skip_proj = nn.Linear(concat_dim, fused_dim)
        
        # Layer normalization & activation for output stability
        self.ln = nn.LayerNorm(fused_dim)
        self.act = nn.ELU()

    def forward(self, e_tab, e_img):
        """
        e_tab: [N, 16] - Tabular clinical embedding
        e_img: [N, 32] - Visual scan embedding
        Returns:
        z_i: [N, 48] - Unified multimodal embedding
        gate: [N, 48] - Gate activation values for inspection/interpretability
        """
        e_cat = torch.cat([e_tab, e_img], dim=-1) # [N, 48]
        
        # Modal projections
        h_tab = self.fc_tab(e_tab)   # [N, 48]
        h_img = self.fc_img(e_img)   # [N, 48]
        h_proj = h_tab + h_img       # [N, 48]
        
        # Dynamic gate
        gate = self.gate_fc(e_cat)   # [N, 48] in [0, 1]
        
        # Gated fusion with skip connection
        h_skip = self.skip_proj(e_cat)
        z = gate * h_proj + (1.0 - gate) * h_skip
        z = self.act(self.ln(z))
        
        return z, gate
