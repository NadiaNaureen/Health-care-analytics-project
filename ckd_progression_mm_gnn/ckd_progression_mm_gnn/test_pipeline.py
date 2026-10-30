import unittest
import torch
import numpy as np
import pandas as pd
from data.loader import (
    get_ckd_stage,
    build_multi_stage_longitudinal_cohort,
    load_and_preprocess_multi_stage_data
)
from graph.construction import build_patient_similarity_graph
from model.gnn_survival import GNNSurvivalModel
from train.loop import discrete_time_survival_loss, calculate_c_index

class TestMultiStageCKDPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        real_csv = "data/updated_ckd_dataset_with_stages.csv"
        cls.df_long = build_multi_stage_longitudinal_cohort(real_csv, num_patients=60, random_seed=42)
        cls.processed = load_and_preprocess_multi_stage_data(cls.df_long, time_bins=[6, 12, 24], random_seed=42)
        cls.data_pyg = build_patient_similarity_graph(cls.processed, k_neighbors=3)
        
    def test_ckd_stage_definitions(self):
        # Stage 1: eGFR >= 90
        self.assertEqual(get_ckd_stage(105.0), 1)
        self.assertEqual(get_ckd_stage(90.0), 1)
        # Stage 2: 60 <= eGFR < 90
        self.assertEqual(get_ckd_stage(89.9), 2)
        self.assertEqual(get_ckd_stage(60.0), 2)
        # Stage 3: 30 <= eGFR < 60
        self.assertEqual(get_ckd_stage(59.9), 3)
        self.assertEqual(get_ckd_stage(30.0), 3)
        # Stage 4: 15 <= eGFR < 30
        self.assertEqual(get_ckd_stage(29.9), 4)
        self.assertEqual(get_ckd_stage(15.0), 4)
        # Stage 5: eGFR < 15
        self.assertEqual(get_ckd_stage(14.9), 5)
        self.assertEqual(get_ckd_stage(5.0), 5)
        
    def test_multi_stage_cohort_and_transitions(self):
        df_cohort = self.processed['df_cohort']
        self.assertGreater(len(df_cohort), 0)
        
        # Check that starting stages 1, 2, 3, 4 are present
        stages_present = set(df_cohort['initial_stage'].unique())
        self.assertTrue({1, 2, 3, 4}.issubset(stages_present))
        
        # Target shape: (N, 3)
        self.assertEqual(self.processed['y'].shape, (len(df_cohort), 3))
        self.assertEqual(self.processed['mask'].shape, (len(df_cohort), 3))
        
    def test_graph_node_features(self):
        # 11 clinical features + 4 stage one-hot features = 15 total features
        self.assertEqual(self.data_pyg.x.shape[1], 15)
        self.assertEqual(self.data_pyg.num_nodes, len(self.processed['df_cohort']))
        self.assertEqual(self.data_pyg.num_edges, len(self.processed['df_cohort']) * 3)
        
    def test_model_forward_and_bounds(self):
        in_features = self.data_pyg.x.shape[1]
        model = GNNSurvivalModel(in_features=in_features, hidden_dim=16, embedding_dim=8, num_intervals=3)
        
        hazards, emb, att = model(self.data_pyg.x, self.data_pyg.edge_index, return_attention_weights=True)
        
        self.assertEqual(hazards.shape, (self.data_pyg.num_nodes, 3))
        self.assertEqual(emb.shape, (self.data_pyg.num_nodes, 8))
        self.assertIsNotNone(att)
        
        # Probability bounds
        self.assertTrue((hazards >= 0.0).all())
        self.assertTrue((hazards <= 1.0).all())
        
    def test_discrete_survival_loss(self):
        in_features = self.data_pyg.x.shape[1]
        model = GNNSurvivalModel(in_features=in_features, hidden_dim=16, embedding_dim=8, num_intervals=3)
        hazards, _, _ = model(self.data_pyg.x, self.data_pyg.edge_index)
        
        loss = discrete_time_survival_loss(hazards, self.data_pyg.y, self.data_pyg.mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertFalse(torch.isnan(loss))
        
    def test_risk_differentiation_bug2(self):
        # Verify that different patient profiles produce different risk scores
        in_features = self.data_pyg.x.shape[1]
        model = GNNSurvivalModel(in_features=in_features, hidden_dim=16, embedding_dim=8, num_intervals=3)
        
        # High risk node (first node) vs low risk node (second node)
        x_copy = self.data_pyg.x.clone()
        # Make node 0 extreme high risk (high BP, high ACR, diabetic, smoker)
        x_copy[0, 3] = 3.0 # high systolic BP
        x_copy[0, 5] = 3.0 # high ACR
        x_copy[0, 7] = 2.0 # diabetes
        x_copy[0, 9] = 2.0 # smoker
        x_copy[0, 10] = -1.0 # untreated
        
        # Make node 1 low risk (normal BP, low ACR, no diabetes, no smoker, treated)
        x_copy[1, 3] = -1.5 # normal systolic BP
        x_copy[1, 5] = -1.5 # low ACR
        x_copy[1, 7] = -1.0 # no diabetes
        x_copy[1, 9] = -1.0 # non-smoker
        x_copy[1, 10] = 1.0 # treated
        
        hazards, _, _ = model(x_copy, self.data_pyg.edge_index)
        risk_high = 1.0 - (1.0 - hazards[0, 0].item()) * (1.0 - hazards[0, 1].item())
        risk_low = 1.0 - (1.0 - hazards[1, 0].item()) * (1.0 - hazards[1, 1].item())
        
        # They should not be identical
        self.assertNotEqual(risk_high, risk_low)

if __name__ == '__main__':
    unittest.main()
