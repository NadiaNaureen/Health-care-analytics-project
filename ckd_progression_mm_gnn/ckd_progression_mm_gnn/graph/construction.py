"""
Graph Construction Module for MM-GAT-Surv
Constructs a k-nearest neighbor patient similarity graph over the 48-dimensional
fused multi-modal patient embedding space (combining clinical biomarkers and kidney scans).
"""

import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import numpy as np

def build_multimodal_patient_graph(fused_embeddings,
                                  clinical_features,
                                  y,
                                  mask,
                                  train_mask,
                                  val_mask,
                                  test_mask,
                                  stage_one_hot=None,
                                  current_stages=None,
                                  image_paths=None,
                                  k_neighbors=10,
                                  metric='euclidean'):
    """
    Constructs a patient similarity graph in the 48-dim fused multi-modal space:
    - Nodes are patients.
    - Node representation x = fused_embeddings [N, 48] (Gated projection of tabular + vision).
    - Raw clinical features x_clinical [N, 15] are preserved for the residual skip connection.
    - Edges connect each patient to their k-nearest neighbors in fused multi-modal space.
    - Returns a PyG Data object.
    """
    if isinstance(fused_embeddings, torch.Tensor):
        feats_np = fused_embeddings.detach().cpu().numpy()
    else:
        feats_np = np.asarray(fused_embeddings)
        
    num_nodes = feats_np.shape[0]
    
    # Fit NearestNeighbors in 48-dim fused space
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric=metric).fit(feats_np)
    distances, indices = nbrs.kneighbors(feats_np)
    
    sources = []
    targets = []
    
    for i in range(num_nodes):
        # Filter out self-loops
        neighbors = [idx for idx in indices[i] if idx != i][:k_neighbors]
        for neighbor in neighbors:
            sources.append(neighbor)
            targets.append(i)
            
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    
    # Node tensors
    x_fused_t = torch.tensor(feats_np, dtype=torch.float32)
    
    if isinstance(clinical_features, torch.Tensor):
        x_clinical_t = clinical_features.float()
    else:
        x_clinical_t = torch.tensor(clinical_features, dtype=torch.float32)
        
    y_t = torch.tensor(y, dtype=torch.float32) if not isinstance(y, torch.Tensor) else y.float()
    mask_t = torch.tensor(mask, dtype=torch.float32) if not isinstance(mask, torch.Tensor) else mask.float()
    
    train_mask_t = torch.tensor(train_mask, dtype=torch.bool) if not isinstance(train_mask, torch.Tensor) else train_mask.bool()
    val_mask_t = torch.tensor(val_mask, dtype=torch.bool) if not isinstance(val_mask, torch.Tensor) else val_mask.bool()
    test_mask_t = torch.tensor(test_mask, dtype=torch.bool) if not isinstance(test_mask, torch.Tensor) else test_mask.bool()
    
    data = Data(
        x=x_fused_t,                     # Fused 48-dim node features
        x_clinical=x_clinical_t,         # Raw clinical features (15-dim) for skip connection
        edge_index=edge_index,           # Graph topology from fused space
        y=y_t,
        mask=mask_t,
        train_mask=train_mask_t,
        val_mask=val_mask_t,
        test_mask=test_mask_t
    )
    
    if stage_one_hot is not None:
        data.stage_oh = torch.tensor(stage_one_hot, dtype=torch.float32) if not isinstance(stage_one_hot, torch.Tensor) else stage_one_hot.float()
    if current_stages is not None:
        data.current_stage = torch.tensor(current_stages, dtype=torch.long) if not isinstance(current_stages, torch.Tensor) else current_stages.long()
        
    return data


def build_patient_similarity_graph(processed_data, k_neighbors=10, metric='euclidean'):
    """
    Constructs a tabular-only patient similarity graph for baseline evaluations and unit testing.
    """
    feats_np = processed_data['scaled_features']
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric=metric).fit(feats_np)
    distances, indices = nbrs.kneighbors(feats_np)
    
    sources = []
    targets = []
    for i in range(len(feats_np)):
        neighbors = [idx for idx in indices[i] if idx != i][:k_neighbors]
        for neighbor in neighbors:
            sources.append(neighbor)
            targets.append(i)
            
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    return Data(
        x=torch.tensor(feats_np, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(processed_data['y'], dtype=torch.float32),
        mask=torch.tensor(processed_data['mask'], dtype=torch.float32),
        train_mask=torch.tensor(processed_data['train_mask'], dtype=torch.bool),
        val_mask=torch.tensor(processed_data['val_mask'], dtype=torch.bool),
        test_mask=torch.tensor(processed_data['test_mask'], dtype=torch.bool)
    )

