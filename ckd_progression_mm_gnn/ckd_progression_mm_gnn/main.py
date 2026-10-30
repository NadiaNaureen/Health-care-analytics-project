"""
Main Entry Point for MM-GAT-Surv
Orchestrates:
1. Data ingestion & multi-stage longitudinal cohort construction
2. Stage-matched kidney scan pairing (pseudopairing)
3. Multimodal node encoding (16-dim tabular + 32-dim vision -> 48-dim fused z_i)
4. Patient similarity graph construction in 48-dim fused space
5. End-to-end MM-GAT-Surv training with discrete survival loss
6. Per-stage transition evaluation (1->2, 2->3, 3->4, 4->5)
7. Expected time to progression & intervention benefit
8. Contrasting patient profile validation & calibration plotting
"""

import os
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

from data.loader import (
    inspect_real_dataset,
    build_multi_stage_longitudinal_cohort,
    load_and_preprocess_multi_stage_data
)
from graph.construction import build_multimodal_patient_graph
from model.mm_gat_surv import MMGATSurvModel
from train.loop import train_mm_model
from eval.metrics import (
    evaluate_per_stage_transitions,
    compute_expected_time_and_gain,
    test_contrasting_patients,
    plot_calibration_curves
)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device: {device}")
    
    real_csv_path = "data/updated_ckd_dataset_with_stages.csv"
    if not os.path.exists(real_csv_path):
        real_csv_path = "C:/Users/Admin/Downloads/updated_ckd_dataset_with_stages.csv"
        
    # Step 1: Ingest Real Dataset & Build Multi-Stage Longitudinal Cohort
    print(f"\nStep 1: Loading real CKD dataset from: {real_csv_path}")
    inspect_real_dataset(real_csv_path)
    df_long = build_multi_stage_longitudinal_cohort(real_csv_path, num_patients=1200, random_seed=42)
    
    # Step 2: Preprocess Multi-Stage Transitions & Assign Stage-Matched Scans
    print("\nStep 2: Preprocessing multi-stage transitions and pairing stage-matched kidney scans...")
    processed_data = load_and_preprocess_multi_stage_data(
        df_long, time_bins=[6, 12, 24], image_base_dir=r"C:\healthcare analytics", random_seed=42
    )
    df_cohort = processed_data['df_cohort']
    
    print(f"\nTotal Eligible Baseline Patients (Stages 1-4): {len(df_cohort)}")
    print("Transition cohort distribution by starting stage:")
    for s in [1, 2, 3, 4]:
        sub = df_cohort[df_cohort['initial_stage'] == s]
        ev_count = int(sub['event'].sum())
        print(f"  Stage {s} (Transition {s}->{s+1}): {len(sub)} patients | {ev_count} progressions ({ev_count/len(sub)*100:.1f}%)")

    # Step 3: Initialize Model & Encode Multimodal Nodes
    print("\nStep 3: Initializing MM-GAT-Surv and encoding multimodal nodes...")
    model = MMGATSurvModel(
        in_clinical_features=15,
        tab_dim=16,
        img_dim=32,
        fused_dim=48,
        gat_hidden_dim=24,
        gat_out_dim=24,
        num_intervals=3,
        num_heads=2,
        dropout=0.1,
        vision_checkpoint="vision_stage_classifier.pt"
    ).to(device)

    # Encode all patients into 48-dim fused space
    x_clin_t = torch.tensor(processed_data['scaled_features'], dtype=torch.float32).to(device)
    x_img_t = processed_data['image_features'].to(device)
    
    with torch.no_grad():
        z_fused, e_tab, e_img, gates = model.encode_multimodal_nodes(x_clin_t, x_img_t)
        print(f"  Tabular Embeddings: {e_tab.shape}")
        print(f"  Visual Embeddings:  {e_img.shape}")
        print(f"  Fused Embeddings:   {z_fused.shape}")
        print(f"  Mean Gate Value:    {gates.mean().item():.4f} (Balancing Tabular & Vision)")

    # Step 4: Construct 10-NN Graph in 48-dim Fused Space
    print("\nStep 4: Constructing 10-NN graph on the 48-dim fused multi-modal patient space...")
    data_pyg = build_multimodal_patient_graph(
        fused_embeddings=z_fused,
        clinical_features=x_clin_t,
        y=processed_data['y'],
        mask=processed_data['mask'],
        train_mask=processed_data['train_mask'],
        val_mask=processed_data['val_mask'],
        test_mask=processed_data['test_mask'],
        k_neighbors=10,
        metric='euclidean'
    )
    print(f"PyG Multimodal Graph: {data_pyg}")
    print(f"  Nodes: {data_pyg.num_nodes} | Edges: {data_pyg.num_edges} | Fused Dims: {data_pyg.x.shape[1]}")

    # Step 5: Train MM-GAT-Surv End-to-End
    print("\nStep 5: Training MM-GAT-Surv model across epochs...")
    checkpoint_path = "best_mm_model.pt"
    history = train_mm_model(
        model,
        data_pyg,
        epochs=100,
        lr=0.005,
        weight_decay=1e-4,
        device=device,
        checkpoint_path=checkpoint_path
    )

    # Load best checkpoint
    print(f"\nLoading best checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Step 6: Per-Stage Transition Evaluation
    print("\nStep 6: Evaluating transition-specific C-Index and 1-Year AUROC...")
    eval_results = evaluate_per_stage_transitions(model, data_pyg, df_cohort, device=device)

    # Step 7: Expected Time to Progression
    print("\nStep 7: Computing expected progression times...")
    time_stats = compute_expected_time_and_gain(model, data_pyg, device=device)
    print(f"Mean Expected Progression Time: {time_stats['mean_expected_time']:.2f} +/- {time_stats['std_expected_time']:.2f} months")

    # Step 8: Contrasting Patient Profiles (Verification of Risk Differentiation)
    print("\nStep 8: Testing contrasting patient profiles (High-Risk vs Low-Risk)...")
    nbrs = NearestNeighbors(n_neighbors=11, metric='euclidean').fit(z_fused.cpu().numpy())
    test_contrasting_patients(
        model,
        processed_data['scaler'],
        nbrs,
        data_pyg,
        processed_data['clinical_feature_cols'],
        device=device
    )

    # Step 9: Calibration Curves
    print("\nStep 9: Generating calibration plot...")
    plot_calibration_curves(model, data_pyg, df_cohort, device=device, out_path="calibration_plot.png")

    # Step 10: Attention Extraction
    print("\nStep 10: Extracting GAT Layer 1 attention weights for Patient 0...")
    with torch.no_grad():
        _, _, att_dict = model(data_pyg.x.to(device), data_pyg.edge_index.to(device), data_pyg.x_clinical.to(device), return_attention=True)
        if att_dict and 'layer1' in att_dict:
            edge_idx_l1, alpha_l1 = att_dict['layer1']
            target_mask = (edge_idx_l1[1] == 0)
            src_nodes = edge_idx_l1[0][target_mask].cpu().numpy()
            weights = alpha_l1[target_mask].mean(dim=-1).cpu().numpy()
            
            print("\n" + "=" * 55)
            print("  MM-GAT ATTENTION NEIGHBORS FOR PATIENT 0 (Node Index: 0)")
            print("=" * 55)
            print(f"{'Neighbor Node':<16} | {'Neighbor Patient ID':<22} | {'Attention Weight'}")
            print("-" * 55)
            sorted_indices = np.argsort(-weights)[:5]
            for idx in sorted_indices:
                nbr_id = src_nodes[idx]
                pid_str = df_cohort.iloc[nbr_id]['patient_id']
                print(f"{nbr_id:<16} | {pid_str:<22} | {weights[idx]:.4f}")
            print("=" * 55 + "\n")

    # Save evaluation summary
    torch.save({
        'eval_results': eval_results,
        'history': history,
        'scaler': processed_data['scaler'],
        'clinical_feature_cols': processed_data['clinical_feature_cols']
    }, "mm_eval_summary.pt")
    print("MM-GAT-Surv pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
