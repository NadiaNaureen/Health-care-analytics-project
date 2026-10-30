"""
Training Loop for MM-GAT-Surv
Trains the multimodal Graph Attention Network with discrete-time survival loss,
logs epoch-by-epoch loss curves and concordance indices, and saves the best model checkpoint.
"""

import os
import torch
import numpy as np
from lifelines.utils import concordance_index

def discrete_time_survival_loss(hazards, y, mask):
    """
    Computes discrete-time negative log-likelihood (NLL) survival loss:
    hazards: [N, T] - Predicted interval-specific hazards
    y: [N, T] - Ground truth event indicator per interval
    mask: [N, T] - Observation mask (0 if interval right-censored/unobserved)
    """
    eps = 1e-7
    interval_nll = - (y * torch.log(hazards + eps) + (1.0 - y) * torch.log(1.0 - hazards + eps))
    masked_nll = interval_nll * mask
    return masked_nll.sum() / (mask.sum() + eps)

def calculate_c_index(hazards_np, y_np, mask_np):
    """
    Calculates the concordance index (C-index) for discrete-time survival predictions.
    Risk score is the cumulative probability of progression: 1 - prod_{j=0}^{T-1} (1 - h_{i,j})
    """
    num_patients = hazards_np.shape[0]
    durations = np.zeros(num_patients)
    event_observed = np.zeros(num_patients)
    risk_scores = np.zeros(num_patients)
    
    for i in range(num_patients):
        durations[i] = mask_np[i].sum()
        event_observed[i] = 1.0 if (y_np[i] * mask_np[i]).sum() > 0 else 0.0
        survival_prob = np.prod(1.0 - hazards_np[i])
        risk_scores[i] = 1.0 - survival_prob
        
    try:
        if len(np.unique(event_observed)) < 2 or len(np.unique(durations)) < 2:
            return 0.5
        c_idx = concordance_index(durations, -risk_scores, event_observed)
    except Exception:
        c_idx = 0.5
        
    return c_idx

def train_mm_model(model, data, epochs=100, lr=0.005, weight_decay=1e-4, device='cpu', checkpoint_path='best_mm_model.pt'):
    """
    Trains MM-GAT-Surv model on PyG Data object containing x_fused, x_clinical, edge_index.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_c_index = -1.0
    best_epoch = -1
    
    x_fused = data.x.to(device)
    x_clinical = data.x_clinical.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    mask = data.mask.to(device)
    
    train_mask = data.train_mask
    val_mask = data.val_mask
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_c_index': [],
        'val_c_index': []
    }
    
    print("=" * 65)
    print("        STARTING MM-GAT-SURV MODEL TRAINING & CONVERGENCE")
    print("=" * 65)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<12} | {'Train C-Index':<14} | {'Val C-Index':<12}")
    print("-" * 65)
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        hazards, _ = model(x_fused, edge_index, x_clinical)
        train_loss = discrete_time_survival_loss(hazards[train_mask], y[train_mask], mask[train_mask])
        
        train_loss.backward()
        optimizer.step()
        
        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_hazards, _ = model(x_fused, edge_index, x_clinical)
            val_loss = discrete_time_survival_loss(val_hazards[val_mask], y[val_mask], mask[val_mask])
            
            # C-index calculations
            train_c = calculate_c_index(
                hazards[train_mask].cpu().numpy(),
                y[train_mask].cpu().numpy(),
                mask[train_mask].cpu().numpy()
            )
            val_c = calculate_c_index(
                val_hazards[val_mask].cpu().numpy(),
                y[val_mask].cpu().numpy(),
                mask[val_mask].cpu().numpy()
            )
            
        history['train_loss'].append(train_loss.item())
        history['val_loss'].append(val_loss.item())
        history['train_c_index'].append(train_c)
        history['val_c_index'].append(val_c)
        
        if val_c > best_val_c_index:
            best_val_c_index = val_c
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_c_index': best_val_c_index,
                'history': history
            }, checkpoint_path)
            
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"{epoch:<8} | {train_loss.item():<12.4f} | {val_loss.item():<12.4f} | {train_c:<14.4f} | {val_c:<12.4f}")
            
    print("-" * 65)
    print(f"Training Complete. Best checkpoint saved from Epoch {best_epoch} with Val C-Index: {best_val_c_index:.4f}")
    print("=" * 65)
    
    return history

train_model = train_mm_model
