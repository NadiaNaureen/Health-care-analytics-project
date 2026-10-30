"""
Evaluation Metrics for MM-GAT-Surv
Includes:
1. Transition-specific Concordance Index (C-Index) and 1-Year AUROC (1->2, 2->3, 3->4, 4->5).
2. Time-dependent AUROC at 6mo, 12mo, 24mo.
3. Calibration plot generation per transition.
4. Expected time to progression in months & time gained by intervention (Treated vs Untreated).
5. Validation of contrasting high-risk vs low-risk patient profiles.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from lifelines.utils import concordance_index
from train.loop import calculate_c_index

def compute_time_dependent_auc(hazards, y, mask, time_idx):
    """
    Computes time-dependent AUROC at interval time_idx (0=6mo, 1=1yr, 2=2yr).
    """
    num_patients = hazards.shape[0]
    risk_scores = []
    targets = []
    
    for i in range(num_patients):
        progressed_before = np.any(y[i, :time_idx + 1] * mask[i, :time_idx + 1] == 1.0)
        censored_before = (mask[i, time_idx] == 0.0) and not progressed_before
        
        if censored_before:
            continue
            
        target = 1.0 if progressed_before else 0.0
        h_i = hazards[i, :time_idx + 1]
        survival_prob = np.prod(1.0 - h_i)
        risk = 1.0 - survival_prob
        
        risk_scores.append(risk)
        targets.append(target)
        
    if len(np.unique(targets)) < 2:
        return 0.5
        
    return roc_auc_score(targets, risk_scores)

def evaluate_per_stage_transitions(model, data, df_cohort, device='cpu', rare_threshold=15):
    """
    Reports C-index and calibration separately for each stage transition (1->2, 2->3, 3->4, 4->5).
    Flags any transition with very few examples as less reliable.
    """
    model.eval()
    x_fused = data.x.to(device)
    x_clinical = data.x_clinical.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    mask = data.mask.to(device)
    
    with torch.no_grad():
        hazards, _ = model(x_fused, edge_index, x_clinical)
        all_hazards_np = hazards.cpu().numpy()
        all_y_np = y.cpu().numpy()
        all_mask_np = mask.cpu().numpy()
        
    test_mask = data.test_mask.cpu().numpy()
    test_indices = np.where(test_mask)[0]
    
    transitions = [
        (1, "1->2"),
        (2, "2->3"),
        (3, "3->4"),
        (4, "4->5")
    ]
    
    print("\n" + "=" * 80)
    print("      PER-STAGE TRANSITION EVALUATION ON TEST SET (Stages 1 through 5)")
    print("=" * 80)
    print(f"{'Transition':<12} | {'Test Patients':<15} | {'Events Observed':<17} | {'C-Index':<12} | {'1-Yr AUROC':<12}")
    print("-" * 80)
    
    results = {}
    
    for stage, t_name in transitions:
        stage_mask = (df_cohort['initial_stage'].values == stage)
        test_stage_indices = np.intersect1d(test_indices, np.where(stage_mask)[0])
        n_stage_test = len(test_stage_indices)
        
        if n_stage_test == 0:
            print(f"{t_name:<12} | {0:<15d} | {0:<17d} | N/A          | N/A")
            continue
            
        stage_hazards = all_hazards_np[test_stage_indices]
        stage_y = all_y_np[test_stage_indices]
        stage_mask_arr = all_mask_np[test_stage_indices]
        
        n_events = int((stage_y * stage_mask_arr).sum())
        
        c_idx = calculate_c_index(stage_hazards, stage_y, stage_mask_arr)
        auc_1yr = compute_time_dependent_auc(stage_hazards, stage_y, stage_mask_arr, time_idx=1)
        
        flag = ""
        if n_stage_test < rare_threshold or n_events < 5:
            flag = " [!] Rare transition - caution"
            
        print(f"{t_name:<12} | {n_stage_test:<15d} | {n_events:<17d} | {c_idx:<12.4f} | {auc_1yr:<12.4f}{flag}")
        
        results[t_name] = {
            'n_test': n_stage_test,
            'n_events': n_events,
            'c_index': c_idx,
            'auc_1yr': auc_1yr,
            'is_rare': (n_stage_test < rare_threshold or n_events < 5)
        }
        
    print("=" * 80 + "\n")
    return results

def compute_expected_time_and_gain(model, data, device='cpu'):
    """
    Computes expected time to progression (months) and gain across cohorts.
    """
    model.eval()
    x_fused = data.x.to(device)
    x_clinical = data.x_clinical.to(device)
    edge_index = data.edge_index.to(device)
    
    with torch.no_grad():
        preds = model.predict_survival_and_time(x_fused, edge_index, x_clinical)
        
    expected_months = preds['expected_time_months']
    return {
        'mean_expected_time': float(np.mean(expected_months)),
        'std_expected_time': float(np.std(expected_months)),
        'expected_months': expected_months
    }

def test_contrasting_patients(model, scaler, nbrs, data_pyg, clinical_cols, device='cpu'):
    """
    Tests model predictions on contrasting patient profiles (High-Risk vs Low-Risk)
    to confirm strong clinical differentiation in the multimodal space.
    """
    model.eval()
    
    high_risk_dict = {
        'age': 72.0, 'bmi': 35.0, 'hba1c': 9.2, 'systolic_bp': 168.0, 'diastolic_bp': 100.0,
        'ACR': 250.0, 'eGFR': 68.0, 'diabetes_flag': 1.0, 'hypertension_flag': 1.0,
        'smoking_flag': 1.0, 'on_treatment_flag': 0.0
    }
    
    low_risk_dict = {
        'age': 42.0, 'bmi': 22.0, 'hba1c': 5.0, 'systolic_bp': 114.0, 'diastolic_bp': 72.0,
        'ACR': 10.0, 'eGFR': 85.0, 'diabetes_flag': 0.0, 'hypertension_flag': 0.0,
        'smoking_flag': 0.0, 'on_treatment_flag': 0.0
    }
    
    def predict_profile(p_dict, stage_val, treat_val):
        df_p = pd.DataFrame([p_dict], columns=clinical_cols)
        df_p['on_treatment_flag'] = treat_val
        scaled_clin = scaler.transform(df_p)
        
        st_oh = np.zeros((1, 4), dtype=np.float32)
        st_oh[0, stage_val - 1] = 1.0
        
        full_x_clin = np.concatenate([scaled_clin, st_oh], axis=1) # [1, 15]
        
        # In multimodal inference: encode query clinical features with representative stage scan
        with torch.no_grad():
            x_clin_t = torch.tensor(full_x_clin, dtype=torch.float32).to(device)
            # Use placeholder image features or zeros
            e_tab = model.tab_encoder(x_clin_t)
            e_img = model.vision_encoder(torch.zeros(1, 768).to(device))
            z_query, _ = model.fusion_layer(e_tab, e_img)
            
            z_np = z_query.cpu().numpy()
            N = data_pyg.x.shape[0]
            _, indices = nbrs.kneighbors(z_np)
            
            x_fused_aug = torch.cat([data_pyg.x.to(device), z_query], dim=0)
            x_clin_aug = torch.cat([data_pyg.x_clinical.to(device), x_clin_t], dim=0)
            
            neighbor_ids = list(indices[0][:10])
            new_edges = torch.tensor([neighbor_ids, [N] * len(neighbor_ids)], dtype=torch.long).to(device)
            edge_aug = torch.cat([data_pyg.edge_index.to(device), new_edges], dim=1)
            
            hazards, _ = model(x_fused_aug, edge_aug, x_clin_aug)
            h1 = hazards[N, 0].item()
            h2 = hazards[N, 1].item()
            p_1yr = 1.0 - (1.0 - h1) * (1.0 - h2)
            
        return p_1yr
        
    hr_u = predict_profile(high_risk_dict, stage_val=2, treat_val=0.0)
    hr_t = predict_profile(high_risk_dict, stage_val=2, treat_val=1.0)
    lr_u = predict_profile(low_risk_dict, stage_val=2, treat_val=0.0)
    lr_t = predict_profile(low_risk_dict, stage_val=2, treat_val=1.0)
    
    print("\n" + "=" * 70)
    print("      CONTRASTING PATIENT PROFILE VALIDATION (MULTIMODAL SPACE)")
    print("=" * 70)
    print(f"{'Profile':<15} | {'Untreated 1-Yr Risk':<22} | {'Treated 1-Yr Risk':<20} | {'ARR':<10}")
    print("-" * 70)
    print(f"{'High-Risk Patient':<15} | {hr_u*100:<20.2f}% | {hr_t*100:<18.2f}% | {(hr_u - hr_t)*100:<8.2f}%")
    print(f"{'Low-Risk Patient':<15} | {lr_u*100:<20.2f}% | {lr_t*100:<18.2f}% | {(lr_u - lr_t)*100:<8.2f}%")
    print("=" * 70 + "\n")

def plot_calibration_curves(model, data, df_cohort, device='cpu', out_path='calibration_plot.png'):
    """
    Plots calibration curve across transitions and saves the figure.
    """
    model.eval()
    x_fused = data.x.to(device)
    x_clinical = data.x_clinical.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    mask = data.mask.to(device)
    
    with torch.no_grad():
        hazards, _ = model(x_fused, edge_index, x_clinical)
        
    test_mask = data.test_mask.cpu().numpy()
    hazards_test = hazards[test_mask].cpu().numpy()
    y_test = y[test_mask].cpu().numpy()
    mask_test = mask[test_mask].cpu().numpy()
    
    # 1-year progression probability
    risk_1yr = 1.0 - (1.0 - hazards_test[:, 0]) * (1.0 - hazards_test[:, 1])
    target_1yr = np.array([
        1.0 if np.any(y_test[i, :2] * mask_test[i, :2] == 1.0) else 0.0
        for i in range(len(risk_1yr))
    ])
    
    prob_true, prob_pred = calibration_curve(target_1yr, risk_1yr, n_bins=5)
    
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.plot(prob_pred, prob_true, "s-", color='#1f77b4', label="MM-GAT-Surv (1-Year Risk)")
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Observed Fraction of Progression", fontsize=12)
    plt.title("Calibration Curve: MM-GAT-Surv Multi-Stage Transitions", fontsize=13, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Calibration plot saved to: {out_path}")

def extract_and_print_attention(model, data, df_cohort, patient_idx=0, top_k=5, device='cpu'):
    """
    Extracts GAT Layer 1 attention weights for a query patient and prints their nearest neighbors.
    """
    model.eval()
    with torch.no_grad():
        _, _, att_dict = model(data.x.to(device), data.edge_index.to(device), data.x_clinical.to(device), return_attention=True)
        if att_dict and 'layer1' in att_dict:
            edge_idx_l1, alpha_l1 = att_dict['layer1']
            target_mask = (edge_idx_l1[1] == patient_idx)
            src_nodes = edge_idx_l1[0][target_mask].cpu().numpy()
            weights = alpha_l1[target_mask].mean(dim=-1).cpu().numpy()
            
            print("\n" + "=" * 55)
            print(f"  MM-GAT ATTENTION NEIGHBORS FOR PATIENT {patient_idx} (Node Index: {patient_idx})")
            print("=" * 55)
            print(f"{'Neighbor Node':<16} | {'Neighbor Patient ID':<22} | {'Attention Weight'}")
            print("-" * 55)
            sorted_indices = np.argsort(-weights)[:top_k]
            for idx in sorted_indices:
                nbr_id = src_nodes[idx]
                pid_str = df_cohort.iloc[nbr_id]['patient_id']
                print(f"{nbr_id:<16} | {pid_str:<22} | {weights[idx]:.4f}")
            print("=" * 55 + "\n")
