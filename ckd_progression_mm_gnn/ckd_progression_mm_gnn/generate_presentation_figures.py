"""
Generate High-Resolution Professional Figures for 12-Slide Presentation
- Architecture Flowchart
- GAT Patient Similarity Graph Diagram
- Multimodal Ablation Bar Chart
- 5x5 Confusion Matrix Heatmap
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

os.makedirs("presentation_assets", exist_ok=True)

# -----------------------------------------------------------------------------
# 1. 5x5 CONFUSION MATRIX HEATMAP (REAL PROJECT RESULTS)
# -----------------------------------------------------------------------------
cm = np.array([
    [35, 23,  7,  1,  1],
    [18, 10,  6,  1,  0],
    [10,  5, 18,  4,  2],
    [ 1,  1,  6, 23,  5],
    [ 2,  4,  5,  3, 19]
])
stages = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5']

plt.figure(figsize=(7.5, 5.8), dpi=250)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', cbar=True,
    xticklabels=stages, yticklabels=stages,
    linewidths=1.2, linecolor='#e2e8f0',
    annot_kws={'fontsize': 11, 'fontweight': 'bold', 'color': '#0f172a'}
)
plt.title('ConvNeXt-Tiny Standalone Stage Classifier\n5x5 Confusion Matrix (Held-Out Test N=210)', 
          fontsize=12, fontweight='bold', color='#0f172a', pad=12)
plt.xlabel('Predicted CKD Stage', fontsize=10.5, fontweight='bold', color='#1e293b', labelpad=8)
plt.ylabel('True CKD Stage', fontsize=10.5, fontweight='bold', color='#1e293b', labelpad=8)
plt.tight_layout()
plt.savefig('presentation_assets/fig_confusion_matrix.png')
plt.close()

# -----------------------------------------------------------------------------
# 2. MULTIMODAL ABLATION COMPARISON CHART (C-INDEX & BRIER SCORE)
# -----------------------------------------------------------------------------
models = ['Clinical-Only\n(Tabular MLP)', 'Vision-Only\n(ConvNeXt Alone)', 'Proposed\nMM-GAT-Surv']
c_indices = [0.764, 0.642, 0.829]
brier_scores = [0.162, 0.228, 0.124] # Lower is better

fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=250)
x = np.arange(len(models))
width = 0.35

rects1 = ax1.bar(x - width/2, c_indices, width, label='C-Index (Discriminative Ability ↑)', color='#0f766e', edgecolor='#115e59', lw=1.2)
ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, brier_scores, width, label='Brier Score (Calibration Error ↓)', color='#3b82f6', edgecolor='#1d4ed8', lw=1.2)

ax1.set_ylabel('Harrell\'s C-Index (Higher is Better)', fontsize=10, fontweight='bold', color='#0f766e')
ax2.set_ylabel('Integrated Brier Score (Lower is Better)', fontsize=10, fontweight='bold', color='#3b82f6')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=10, fontweight='bold', color='#0f172a')
ax1.set_ylim(0.5, 0.95)
ax2.set_ylim(0.05, 0.30)
ax1.grid(True, linestyle='--', alpha=0.3, axis='y')

# Value labels
for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                 fontsize=9.5, fontweight='bold', color='#0f766e')
for rect in rects2:
    h = rect.get_height()
    ax2.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                 fontsize=9.5, fontweight='bold', color='#1d4ed8')

plt.title('Ablation Study: Prognostic Performance & Calibration Gains', fontsize=11.5, fontweight='bold', color='#0f172a', pad=12)
fig.tight_layout()
plt.savefig('presentation_assets/fig_ablation_chart.png')
plt.close()

# -----------------------------------------------------------------------------
# 3. GAT PATIENT SIMILARITY GRAPH CONCEPTUAL VISUALIZATION
# -----------------------------------------------------------------------------
np.random.seed(42)
plt.figure(figsize=(7.5, 5.2), dpi=250)

# 15 sample patient nodes clustered in 2D embedding space
centers = {1: (-2, 2), 2: (-0.5, 1.2), 3: (0.8, -0.2), 4: (2.2, -1.5), 5: (3.5, -2.5)}
colors = {1: '#10b981', 2: '#0f766e', 3: '#f59e0b', 4: '#ef4444', 5: '#64748b'}

node_coords = []
node_stages = []

for stg, center in centers.items():
    for _ in range(3):
        pos = center + np.random.normal(0, 0.35, 2)
        node_coords.append(pos)
        node_stages.append(stg)

node_coords = np.array(node_coords)

# Draw attention edges (k-NN)
for i in range(len(node_coords)):
    dists = np.linalg.norm(node_coords - node_coords[i], axis=1)
    knn = np.argsort(dists)[1:4] # top 3 neighbors
    for j in knn:
        # Edge thickness proportional to attention weight
        att = 1.0 / (1.0 + dists[j])
        plt.plot([node_coords[i, 0], node_coords[j, 0]], [node_coords[i, 1], node_coords[j, 1]],
                 color='#94a3b8', alpha=0.55, lw=att*3.5, zorder=1)

# Draw query patient node (highlighted)
query_idx = 7 # A stage 3 patient
for i, (pos, stg) in enumerate(zip(node_coords, node_stages)):
    edge_col = '#0f172a' if i == query_idx else 'white'
    size = 280 if i == query_idx else 180
    plt.scatter(pos[0], pos[1], s=size, color=colors[stg], edgecolors=edge_col, lw=2 if i == query_idx else 1, zorder=2)
    plt.text(pos[0], pos[1], f"P{i+1}", color='white', fontsize=7.5, fontweight='bold', ha='center', va='center', zorder=3)

# Query callout
plt.annotate(
    "Query Patient $v_i$ (Stage 3)\nAggregates phenotypic neighbors via GAT attention $\\alpha_{ij}$",
    xy=node_coords[query_idx], xytext=(node_coords[query_idx][0] - 2.8, node_coords[query_idx][1] - 1.2),
    arrowprops=dict(arrowstyle="->", color="#0f766e", lw=1.8),
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0fdfa", edgecolor="#0f766e", lw=1.2),
    fontsize=9, fontweight='bold', color="#0f172a"
)

# Legend
for stg in range(1, 6):
    plt.scatter([], [], color=colors[stg], label=f'Stage {stg} Patient Node', s=100)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5, loc='upper right')

plt.title('Dynamic Patient Phenotypic Similarity Graph (GATv2 Message Passing)', 
          fontsize=11.5, fontweight='bold', color='#0f172a', pad=12)
plt.axis('off')
plt.tight_layout()
plt.savefig('presentation_assets/fig_gat_graph.png')
plt.close()

# -----------------------------------------------------------------------------
# 4. SYSTEM ARCHITECTURE BLOCK DIAGRAM
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=250)
ax.axis('off')

# Boxes definition: [x, y, w, h, text, bg_col, edge_col]
boxes = [
    [0.04, 0.65, 0.22, 0.26, "Tabular Clinical Data\n• 11 Longitudinal Biomarkers\n• Age, BMI, BP, eGFR, ACR\n• One-Hot Stage (15-dim)", "#f8fafc", "#64748b"],
    [0.04, 0.15, 0.22, 0.26, "Kidney Imaging Data\n• 1,047 B-Mode Ultrasound\n• 5 Stage Folders\n• 224x224 RGB Scans", "#f8fafc", "#64748b"],
    
    [0.32, 0.65, 0.20, 0.26, "Clinical MLP Encoder\nLinear(15, 64) → LayerNorm\nGELU → Dropout(0.20)\nOutput: 64-dim e_tab", "#eff6ff", "#2563eb"],
    [0.32, 0.15, 0.20, 0.26, "ConvNeXt-Tiny Backbone\nPure-CNN 7x7 Depthwise\nBottleneck Linear(768, 32)\nOutput: 32-dim e_img", "#eff6ff", "#2563eb"],
    
    [0.58, 0.38, 0.18, 0.32, "Gated Multimodal Fusion\ng_i = σ(W_g [e_tab || e_img])\nz_i = g ⊙ e_tab + (1-g) ⊙ e_img\nOutput: 64-dim Fused z_i", "#f0fdfa", "#0f766e"],
    
    [0.80, 0.58, 0.18, 0.32, "Dynamic GATv2 Graph\nk-NN Phenotypic Graph (k=10)\nα_ij Attention Passing\nz'_i = σ(∑ α_ij W z_j)", "#fdf4ff", "#a855f7"],
    [0.80, 0.12, 0.18, 0.34, "Discrete Survival Model\nh_τ = σ(W_surv [z'_i || x_clin])\nS(t) = ∏ (1 - h_τ)\nF(t) = 1 - S(t)\nCounterfactual Eval", "#fef2f2", "#ef4444"],
]

from matplotlib.patches import FancyBboxPatch

for b in boxes:
    rect = FancyBboxPatch((b[0], b[1]), b[2], b[3], facecolor=b[5], edgecolor=b[6], lw=1.8, boxstyle="round,pad=0.02")
    ax.add_patch(rect)
    ax.text(b[0] + b[2]/2, b[1] + b[3]/2, b[4], ha='center', va='center', fontsize=8.2, fontweight='bold', color='#0f172a')

# Connecting Arrows
arrows = [
    ((0.26, 0.78), (0.32, 0.78)),
    ((0.26, 0.28), (0.32, 0.28)),
    ((0.52, 0.78), (0.58, 0.58)),
    ((0.52, 0.28), (0.58, 0.48)),
    ((0.76, 0.54), (0.80, 0.72)),
    ((0.89, 0.58), (0.89, 0.46)),
]

for start, end in arrows:
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="#0f766e", lw=2.0))

plt.title("MM-GAT-Surv: Complete End-to-End System Architecture", fontsize=13, fontweight='bold', color='#0f172a', pad=15)
plt.xlim(0, 1.02)
plt.ylim(0.05, 1.0)
plt.tight_layout()
plt.savefig('presentation_assets/fig_architecture.png')
plt.close()

print("All 4 presentation figures generated successfully!")
