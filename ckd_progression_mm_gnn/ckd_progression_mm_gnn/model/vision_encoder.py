"""
Vision Encoder Module for MM-GAT-Surv
Uses pretrained ConvNeXt-Tiny backbone fine-tuned on kidney scans to output
a 32-dimensional visual embedding vector (e_img) per patient scan.
"""

import os
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class ConvNeXtVisionEncoder(nn.Module):
    def __init__(self, embedding_dim=32, checkpoint_path="vision_stage_classifier.pt", freeze_backbone=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Pretrained ConvNeXt-Tiny backbone
        try:
            self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        except Exception:
            self.backbone = convnext_tiny(weights=None)
            
        self.backbone.classifier[2] = nn.Identity()  # Outputs [B, 768]
        
        # 32-dimensional visual embedding bottleneck
        self.proj = nn.Sequential(
            nn.Linear(768, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                # Map embedding_head -> proj
                mapped_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("embedding_head."):
                        new_k = k.replace("embedding_head.", "")
                        mapped_dict[new_k] = v
                self.proj.load_state_dict(mapped_dict, strict=False)
                print(f"Loaded fine-tuned vision bottleneck weights from {checkpoint_path}.")
            except Exception as e:
                print(f"Warning: could not load vision checkpoint: {e}")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x, ablate_vision=False):
        """
        x: Tensor of shape [batch_size, 3, 224, 224] OR pre-extracted features [batch_size, 768]
        Returns: Tensor of shape [batch_size, 32]
        """
        if ablate_vision:
            batch_size = x.shape[0] if x is not None else 1
            device = x.device if x is not None else torch.device("cpu")
            return torch.zeros(batch_size, self.embedding_dim, device=device)
            
        if x.dim() == 4:
            feat = self.backbone(x)  # [B, 768]
        else:
            feat = x  # already [B, 768]
            
        emb = self.proj(feat)  # [B, 32]
        return emb
