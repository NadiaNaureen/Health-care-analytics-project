# Chronic Kidney Disease (CKD) Progression Prediction using GNN + Discrete-Time Survival Modeling

A pair-programming machine learning project that predicts CKD progression from **Stage 1 (eGFR ≥ 90 mL/min/1.73m²)** to **Stage 2 (eGFR 60–89 mL/min/1.73m²)** using a **Graph Attention Network (GAT)** coupled with **discrete-time survival modeling**.

The model estimates the interval-specific hazard and cumulative 1-year risk of progression under counterfactual scenarios: **"Untreated" (No ACEi/ARB)** versus **"Treated" (On ACEi/ARB)**.

---

## Features & Methodology

1. **Longitudinal Patient Data Processing (`data/loader.py`)**:
   - Supports cohorts starting at Stage 1 (baseline eGFR ≥ 90).
   - Automatically derives progression events (first visit dropping into the 60–89 range or lower) and handles right-censored patients without negative label bias.
   - Splits patients into Train (70%), Validation (15%), and Test (15%) by `patient_id` to prevent data leakage across visits.
   - Includes fallback synthetic data generator and adapter for the user's shared dataset.

2. **Patient Similarity Graph (`graph/construction.py`)**:
   - Connects patients to their $k=10$ nearest neighbors based on standardized baseline clinical variables (eGFR, ACR, blood pressure, HbA1c, age, BMI, comorbidities).
   - Built into a PyTorch Geometric `Data` object.

3. **GAT Encoder + Discrete-Time Survival Head (`model/gnn_survival.py`)**:
   - **GNN Encoder**: 2-layer Graph Attention Network (GATConv) with multi-head attention to learn patient representations from clinical neighborhoods.
   - **Hazard Head**: Multi-Layer Perceptron (MLP) taking the patient embedding and a one-hot representation of the discrete time intervals (6 months, 12 months, 24 months).
   - **Counterfactual Inference**: Evaluates nodes twice with `on_treatment_flag=0` and `on_treatment_flag=1` to isolate the treatment effect.

4. **Discrete-Time Survival Loss (`train/loop.py`)**:
   - Computes Negative Log-Likelihood (NLL) over observed intervals:
     $$\mathcal{L} = - \frac{1}{N} \sum_{i=1}^N \sum_{j=1}^{T} \text{mask}_{i,j} \cdot \left[ y_{i,j} \log(h_{i,j}) + (1 - y_{i,j}) \log(1 - h_{i,j}) \right]$$
   - Handles censored patients by masking unobserved intervals beyond their last follow-up.

5. **Evaluation & Calibration (`eval/metrics.py`)**:
   - Concordance Index (C-Index) and time-dependent AUROC at 6mo, 1yr, and 2yr.
   - Binned calibration curve (`calibration_plot.png`).
   - GAT attention weight extraction to inspect neighbor influences.

6. **Interactive Streamlit Demo App (`app.py`)**:
   - Clean, interactive web interface to test query patients and visualize risk reductions side-by-side.

---

## Project Structure

```text
ckd_progression_gnn/
├── app.py                  # Streamlit web demo application
├── main.py                 # End-to-end pipeline (training, evaluation, attention, counterfactuals)
├── test_pipeline.py        # Automated unit test suite
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── best_model.pt           # Saved trained model checkpoint
├── calibration_plot.png    # Output calibration chart
├── data/
│   ├── __init__.py
│   ├── loader.py           # Longitudinal simulation, cohort filtering, and preprocessing
│   └── shared_dataset.csv  # User shared baseline dataset
├── graph/
│   ├── __init__.py
│   └── construction.py     # k-NN patient similarity graph builder
├── model/
│   ├── __init__.py
│   └── gnn_survival.py     # 2-layer GAT encoder + discrete hazard head
├── train/
│   ├── __init__.py
│   └── loop.py             # NLL survival loss, training and validation loop
└── eval/
    ├── __init__.py
    └── metrics.py          # C-index, time-dependent AUROC, calibration, attention & counterfactuals
```

---

## Getting Started

### 1. Prerequisites & Environment Setup
Ensure you have Python 3.10+ installed. Open a terminal (PowerShell or Command Prompt) and navigate to the project directory:

```powershell
cd "C:\Users\Admin\.gemini\antigravity\scratch\ckd_progression_gnn"
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

---

### 2. Running the Training & Evaluation Pipeline
To train the model on the cohort, evaluate metrics, and extract attention weights:

```powershell
python main.py
```

Expected outputs:
- **Test Concordance Index (C-Index)**: ~0.97
- **Time-Dependent AUROC at 6mo / 1yr / 2yr**: ~0.96 - 0.98
- **Saved Model**: `best_model.pt`
- **Saved Calibration Plot**: `calibration_plot.png`

---

### 3. Running Unit Tests
To verify all individual components (data loader, graph construction, GAT model, and survival loss):

```powershell
python -m unittest test_pipeline.py
```

---

### 4. Running the Streamlit Demo App
Launch the interactive web application:

```powershell
streamlit run app.py
```

Once started, Streamlit will provide a local URL (e.g., `http://localhost:8501`). Open it in your browser to:
1. Enter patient clinical values (eGFR, ACR, blood pressure, HbA1c, age, BMI, smoking, diabetes, hypertension) or select an example preset.
2. Click **"Run Progression Risk Prediction"**.
3. View the predicted 1-year progression probability side-by-side for **Untreated** vs. **Treated**.
4. View the bar chart comparing the two outcomes, along with the patient's top 10 clinical nearest neighbors in the reference cohort.


