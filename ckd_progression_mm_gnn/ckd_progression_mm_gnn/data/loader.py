"""
Data Loading, Preprocessing, and Stage-Matched Multimodal Pairing Module
For Multi-Modal CKD Progression (MM-GAT-Surv) across Stages 1 through 5.

CRITICAL METHODOLOGICAL DISCLOSURE:
The clinical numeric dataset and the kidney ultrasound scan dataset are NOT linked
at the individual patient level (there are no shared patient IDs). The image folders
(images-stage0 through images-stage4) are categorized strictly by disease stage.
Therefore, this module implements STAGE-MATCHED PAIRING (PSEUDOPAIRING):
- Each patient's baseline CKD stage (Stages 1 through 5) is derived from their eGFR.
- A representative ultrasound scan is randomly assigned from the matching stage folder.
- This creates a stage-conditioned multimodal representation for research/educational
  demonstration while acknowledging this real-world pairing constraint.
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_ckd_stage(egfr):
    """
    Derives the CKD stage from eGFR (mL/min/1.73m²):
    Stage 1: eGFR >= 90
    Stage 2: eGFR 60-89
    Stage 3: eGFR 30-59
    Stage 4: eGFR 15-29
    Stage 5: eGFR < 15
    """
    if egfr >= 90.0:
        return 1
    elif egfr >= 60.0:
        return 2
    elif egfr >= 30.0:
        return 3
    elif egfr >= 15.0:
        return 4
    else:
        return 5

def inspect_real_dataset(csv_path):
    """
    Inspects and reports missing values and stage counts for the real dataset.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Real dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    missing = df.isnull().sum()
    total_missing = missing.sum()
    
    print("=" * 60)
    print(f"REAL DATASET INSPECTION: {csv_path}")
    print("=" * 60)
    print(f"Total Rows: {total_rows}, Total Columns: {len(df.columns)}")
    print(f"Total Missing Values: {total_missing}")
    if total_missing == 0:
        print("-> Clean dataset: Zero missing values detected across all columns.")
    else:
        print("Missing value breakdown per column:")
        print(missing[missing > 0])
        
    if 'ckd_stage' in df.columns:
        print("\nStage breakdown in dataset:")
        for stage, count in df['ckd_stage'].value_counts().sort_index().items():
            print(f"  Stage {stage}: {count} patients ({count/total_rows*100:.1f}%)")
    print("=" * 60)
    return df

def build_multi_stage_longitudinal_cohort(real_csv_path, num_patients=1200, random_seed=42):
    """
    Loads the real dataset and constructs a multi-stage longitudinal cohort across Stages 1 through 4.
    Simulates consecutive visits [0, 6, 12, 18, 24] months.
    Tracks transitions (1->2, 2->3, 3->4, 4->5), regressions, and stable visits.
    """
    np.random.seed(random_seed)
    df_raw = pd.read_csv(real_csv_path)
    
    # Stratified baseline sampling across Stages 1, 2, 3, 4 (300 patients each)
    stage_samples = []
    n_per_stage = num_patients // 4
    
    for s in [1, 2, 3, 4]:
        sub_df = df_raw[df_raw['ckd_stage'] == s]
        if len(sub_df) == 0:
            sub_df = df_raw.sample(n=n_per_stage, replace=True, random_state=random_seed + s)
        sampled = sub_df.sample(n=n_per_stage, replace=(len(sub_df) < n_per_stage), random_state=random_seed + s)
        stage_samples.append(sampled)
        
    df_baselines = pd.concat(stage_samples).reset_index(drop=True)
    df_baselines['patient_id'] = [f"PAT_{i:04d}" for i in range(len(df_baselines))]
    
    visit_records = []
    consecutive_transitions = []
    progression_count = 0
    stable_count = 0
    regression_count = 0
    
    visit_schedule = [0.0, 6.0, 12.0, 18.0, 24.0]
    
    for _, patient in df_baselines.iterrows():
        patient_id = patient['patient_id']
        base_gfr = float(patient['gfr'])
        initial_stage = int(patient['ckd_stage'])
        
        # Clinical risk factors
        sbp = float(patient['blood_pressure'])
        dbp = sbp * 0.65 + np.random.normal(0, 3)
        hba1c = 7.5 if patient['cluster'] in [2, 3] else 5.8
        acr = float(patient['oxalate_levels']) * 40.0 + (50.0 if patient['hematuria'] == 1 else 10.0)
        age = 50 + int(patient['months']) * 2
        bmi = 26.5 + (sbp - 120.0) * 0.1
        
        is_smoker = 1 if str(patient['smoking']).lower() == 'yes' else 0
        is_diabetic = 1 if hba1c >= 6.5 else 0
        is_hypertensive = 1 if sbp >= 135 else 0
        on_treatment = 1 if (np.random.rand() > 0.45) else 0
        
        # Monthly eGFR decline rate
        base_decline_rate = 0.45 if initial_stage >= 3 else 0.35
        if is_hypertensive: base_decline_rate += 0.35
        if acr > 150: base_decline_rate += 0.40
        if is_diabetic: base_decline_rate += 0.30
        if is_smoker: base_decline_rate += 0.20
        
        # Treatment protective effect
        if on_treatment == 1:
            base_decline_rate *= 0.45
            
        current_gfr = base_gfr
        patient_visits = []
        
        for v_idx, t_month in enumerate(visit_schedule):
            if v_idx > 0:
                elapsed = t_month - visit_schedule[v_idx - 1]
                decline = (base_decline_rate * elapsed) + np.random.normal(0, 0.4)
                current_gfr = max(5.0, current_gfr - decline)
                
            current_stage = get_ckd_stage(current_gfr)
            
            visit_record = {
                'patient_id': patient_id,
                'visit_date': t_month,
                'eGFR': round(current_gfr, 2),
                'ACR': round(acr, 2),
                'systolic_bp': round(sbp, 1),
                'diastolic_bp': round(dbp, 1),
                'hba1c': round(hba1c, 2),
                'age': age,
                'bmi': round(bmi, 1),
                'diabetes_flag': is_diabetic,
                'hypertension_flag': is_hypertensive,
                'on_treatment_flag': on_treatment,
                'smoking_flag': is_smoker,
                'initial_stage': initial_stage,
                'current_stage': current_stage
            }
            visit_records.append(visit_record)
            patient_visits.append(visit_record)
            
            if current_stage > initial_stage:
                break
                
        # Analyze consecutive transitions
        for v_idx in range(len(patient_visits) - 1):
            v_curr = patient_visits[v_idx]
            v_next = patient_visits[v_idx + 1]
            s_curr = v_curr['current_stage']
            s_next = v_next['current_stage']
            
            if s_next > s_curr:
                progression_count += 1
                transition_label = 1
            elif s_next < s_curr:
                regression_count += 1
                transition_label = -1
            else:
                stable_count += 1
                transition_label = 0
                
            consecutive_transitions.append({
                'patient_id': patient_id,
                'visit_start': v_curr['visit_date'],
                'visit_end': v_next['visit_date'],
                'stage_start': s_curr,
                'stage_end': s_next,
                'transition_label': transition_label
            })
            
    df_long = pd.DataFrame(visit_records)
    print("\n--- Longitudinal Cohort Simulation Summary ---")
    print(f"Total Patients: {len(df_baselines)}")
    print(f"Total Longitudinal Visit Records: {len(df_long)}")
    print(f"Consecutive Visit Transitions: {len(consecutive_transitions)}")
    print(f"  - Progressed to next stage down: {progression_count} ({progression_count/len(consecutive_transitions)*100:.1f}%)")
    print(f"  - Stayed in same stage:          {stable_count} ({stable_count/len(consecutive_transitions)*100:.1f}%)")
    print(f"  - Regressed (eGFR improved):     {regression_count} ({regression_count/len(consecutive_transitions)*100:.1f}%) [Tracked separately]")
    print("-" * 50)
    
    return df_long


def get_stage_image_map(image_base_dir=None):
    """
    Scans the stage folders and builds a mapping from stage number (1..5)
    to a list of valid JPEG/PNG file paths across local Dataset directories.
    Supports Stage0-4 (Stage 1) through Stage4-4 (Stage 5).
    """
    candidate_bases = []
    if image_base_dir and os.path.exists(image_base_dir):
        candidate_bases.append(image_base_dir)
        
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(curr_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    
    for c in [
        os.path.join(parent_dir, "dataset"),
        os.path.join(grandparent_dir, "dataset"),
        r"C:\NADIA\Sem7\Health care\HEALTHCARE PROJECT\ckd_progression_mm_gnn\dataset",
        r"C:\NADIA\Sem7\Health care\project\dataset",
        r"C:\healthcare analytics"
    ]:
        if os.path.exists(c) and c not in candidate_bases:
            candidate_bases.append(c)

    stage_folder_patterns = {
        1: ["Stage0-4", "images-stage0", "stage0"],
        2: ["Stage1-4", "images-stage1", "stage1"],
        3: ["Stage2-4", "images-stage2", "stage2"],
        4: ["Stage3-4", "images-stage3", "stage3"],
        5: ["Stage4-4", "images-stage4", "stage4"]
    }
    
    stage_image_map = {s: [] for s in range(1, 6)}
    
    for base in candidate_bases:
        for stage_num, patterns in stage_folder_patterns.items():
            if stage_image_map[stage_num]:
                continue
            for pattern in patterns:
                folder_path = os.path.join(base, pattern)
                if os.path.exists(folder_path):
                    files = glob.glob(os.path.join(folder_path, "**", "*.jpg"), recursive=True)
                    files += glob.glob(os.path.join(folder_path, "**", "*.jpeg"), recursive=True)
                    files += glob.glob(os.path.join(folder_path, "**", "*.png"), recursive=True)
                    if files:
                        stage_image_map[stage_num] = sorted(files)
                        break
                        
    return stage_image_map



def assign_stage_matched_scans(df_cohort, image_base_dir=r"C:\healthcare analytics",
                               feature_cache_path="cache/convnext_features.pt", random_seed=42):
    """
    Assigns each patient in the cohort a stage-matched ultrasound scan:
    - Queries stage_image_map using the patient's baseline initial_stage.
    - Randomly assigns a scan from the matching stage folder.
    - Maps the scan to its pre-extracted 768-dim ConvNeXt feature vector if cache is present.
    """
    np.random.seed(random_seed)
    stage_map = get_stage_image_map(image_base_dir)
    
    # Load feature cache if available
    path_to_feat = {}
    if feature_cache_path and os.path.exists(feature_cache_path):
        try:
            cache = torch.load(feature_cache_path, map_location="cpu", weights_only=False)
            for p, f in zip(cache['train_paths'], cache['train_feats']):
                path_to_feat[os.path.normpath(p)] = f
            for p, f in zip(cache['test_paths'], cache['test_feats']):
                path_to_feat[os.path.normpath(p)] = f
        except Exception as e:
            print(f"Warning loading feature cache: {e}")
            
    assigned_paths = []
    assigned_features = []
    
    for _, row in df_cohort.iterrows():
        s = int(row['initial_stage'])
        available_scans = stage_map.get(s, [])
        if available_scans:
            chosen_scan = np.random.choice(available_scans)
        else:
            chosen_scan = ""
            
        assigned_paths.append(chosen_scan)
        
        norm_p = os.path.normpath(chosen_scan)
        if norm_p in path_to_feat:
            assigned_features.append(path_to_feat[norm_p])
        else:
            # Fallback zero vector
            assigned_features.append(torch.zeros(768))
            
    df_cohort['assigned_scan_path'] = assigned_paths
    image_features_tensor = torch.stack(assigned_features, dim=0) # [N, 768]
    
    print(f"\nStage-Matched Image Pairing Complete (Pseudo-Pairing):")
    print(f"  Assigned {len(assigned_paths)} scans across {len(stage_map)} stage folders.")
    print(f"  Pre-extracted visual feature tensor shape: {image_features_tensor.shape}")
    print("  Note: Image and tabular data are matched by CKD stage, not individual patient ID.")
    
    return df_cohort, assigned_paths, image_features_tensor


def load_and_preprocess_multi_stage_data(df_long, time_bins=[6, 12, 24],
                                         image_base_dir=r"C:\healthcare analytics",
                                         random_seed=42):
    """
    Preprocesses longitudinal visits for multi-stage discrete survival prediction,
    and assigns stage-matched kidney scans to each patient.
    """
    df_long = df_long.sort_values(by=['patient_id', 'visit_date']).copy()
    patient_ids = df_long['patient_id'].unique()
    
    patient_cohort = []
    
    for pid in patient_ids:
        df_p = df_long[df_long['patient_id'] == pid]
        baseline = df_p.iloc[0]
        initial_stage = int(baseline['initial_stage'])
        
        if initial_stage not in [1, 2, 3, 4]:
            continue
            
        progressed = 0
        event_time = baseline['visit_date']
        
        for _, visit in df_p.iterrows():
            if visit['current_stage'] > initial_stage:
                progressed = 1
                event_time = visit['visit_date']
                break
            else:
                event_time = visit['visit_date']
                
        transition_name = f"{initial_stage}->{initial_stage+1}"
        
        patient_cohort.append({
            'patient_id': pid,
            'event': progressed,
            'time': event_time,
            'initial_stage': initial_stage,
            'transition_name': transition_name,
            'eGFR': baseline['eGFR'],
            'ACR': baseline['ACR'],
            'systolic_bp': baseline['systolic_bp'],
            'diastolic_bp': baseline['diastolic_bp'],
            'hba1c': baseline['hba1c'],
            'age': baseline['age'],
            'bmi': baseline['bmi'],
            'diabetes_flag': baseline['diabetes_flag'],
            'hypertension_flag': baseline['hypertension_flag'],
            'on_treatment_flag': baseline['on_treatment_flag'],
            'smoking_flag': baseline['smoking_flag']
        })
        
    df_cohort = pd.DataFrame(patient_cohort)
    num_intervals = len(time_bins)
    
    y_labels = []
    y_masks = []
    
    for _, row in df_cohort.iterrows():
        t = row['time']
        event = row['event']
        
        y = np.zeros(num_intervals, dtype=np.float32)
        mask = np.zeros(num_intervals, dtype=np.float32)
        
        if event == 1:
            if t <= 6.0:
                y[0] = 1.0
                mask[0] = 1.0
            elif t <= 12.0:
                y[1] = 1.0
                mask[:2] = 1.0
            else:
                y[2] = 1.0
                mask[:3] = 1.0
        else: # Censored
            if t >= 24.0:
                mask[:3] = 1.0
            elif t >= 12.0:
                mask[:2] = 1.0
            elif t >= 6.0:
                mask[0] = 1.0
                
        y_labels.append(y)
        y_masks.append(mask)
        
    df_cohort['y'] = y_labels
    df_cohort['mask'] = y_masks
    
    # Assign stage-matched ultrasound scans
    df_cohort, assigned_scans, image_features = assign_stage_matched_scans(
        df_cohort, image_base_dir=image_base_dir, random_seed=random_seed
    )
    
    # Split by patient_id: 70% Train, 15% Val, 15% Test
    p_ids_list = list(df_cohort['patient_id'])
    train_ids, test_ids = train_test_split(p_ids_list, test_size=0.30, random_state=random_seed)
    val_ids, test_ids = train_test_split(test_ids, test_size=0.50, random_state=random_seed)

    
    clinical_feature_cols = [
        'age', 'bmi', 'hba1c', 'systolic_bp', 'diastolic_bp',
        'ACR', 'eGFR', 'diabetes_flag', 'hypertension_flag', 'smoking_flag', 'on_treatment_flag'
    ]
    
    scaler = StandardScaler()
    train_idx = df_cohort['patient_id'].isin(train_ids)
    scaler.fit(df_cohort.loc[train_idx, clinical_feature_cols])
    
    scaled_clinical = scaler.transform(df_cohort[clinical_feature_cols])
    
    # Current stage one-hot encoding (Stages 1, 2, 3, 4)
    stage_one_hot = np.zeros((len(df_cohort), 4), dtype=np.float32)
    for idx, s in enumerate(df_cohort['initial_stage']):
        stage_one_hot[idx, s - 1] = 1.0
        
    # Full clinical feature matrix: [scaled_clinical (11) + stage_one_hot (4)] = 15 dim
    full_clinical_features = np.concatenate([scaled_clinical, stage_one_hot], axis=1)
    
    n_patients = len(df_cohort)
    train_mask = np.zeros(n_patients, dtype=bool)
    val_mask = np.zeros(n_patients, dtype=bool)
    test_mask = np.zeros(n_patients, dtype=bool)
    
    pid_to_idx = {pid: i for i, pid in enumerate(df_cohort['patient_id'])}
    for pid in train_ids: train_mask[pid_to_idx[pid]] = True
    for pid in val_ids: val_mask[pid_to_idx[pid]] = True
    for pid in test_ids: test_mask[pid_to_idx[pid]] = True
    
    return {
        'df_cohort': df_cohort,
        'scaled_features': full_clinical_features,
        'image_features': image_features,
        'assigned_scans': assigned_scans,
        'y': np.stack(y_labels),
        'mask': np.stack(y_masks),
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'clinical_feature_cols': clinical_feature_cols,
        'scaler': scaler
    }
