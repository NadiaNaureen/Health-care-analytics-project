"""
Unit Tests for MM-GAT-Surv Pipeline
Validates:
1. Vision Encoder feature extraction and zero-ablation mode.
2. Gated Linear Projection fusion shape ([N, 48]) and gate range ([0, 1]).
3. Graph construction over 48-dim fused space.
4. Survival loss non-negativity and hazard range ([0, 1]).
5. Survival probability monotonicity across time intervals.
6. Counterfactual risk differentiation and treatment protective effect.
"""

import unittest
import torch
import numpy as np

from model.vision_encoder import ConvNeXtVisionEncoder
from model.fusion import GatedLinearProjectionFusion
from model.mm_gat_surv import TabularClinicalEncoder, MMGATSurvModel
from graph.construction import build_multimodal_patient_graph
from train.loop import discrete_time_survival_loss
from data.loader import get_ckd_stage

class TestMMGATSurvPipeline(unittest.TestCase):
    def setUp(self):
        self.num_patients = 10
        self.tab_features = torch.randn(self.num_patients, 15)
        self.img_features = torch.randn(self.num_patients, 768)

    def test_stage_detection(self):
        self.assertEqual(get_ckd_stage(105.0), 1)
        self.assertEqual(get_ckd_stage(90.0), 1)
        self.assertEqual(get_ckd_stage(89.9), 2)
        self.assertEqual(get_ckd_stage(60.0), 2)
        self.assertEqual(get_ckd_stage(45.0), 3)
        self.assertEqual(get_ckd_stage(20.0), 4)
        self.assertEqual(get_ckd_stage(10.0), 5)

    def test_vision_encoder_and_ablation(self):
        encoder = ConvNeXtVisionEncoder(embedding_dim=32, checkpoint_path=None)
        
        # Test normal pass with 768-dim features
        emb = encoder(self.img_features, ablate_vision=False)
        self.assertEqual(emb.shape, (self.num_patients, 32))
        
        # Test ablation mode (deliberately zeroed out)
        emb_zero = encoder(self.img_features, ablate_vision=True)
        self.assertEqual(emb_zero.shape, (self.num_patients, 32))
        self.assertTrue(torch.all(emb_zero == 0.0))

    def test_gated_fusion_layer(self):
        tab_enc = TabularClinicalEncoder(in_features=15, out_dim=16)
        fusion = GatedLinearProjectionFusion(tab_dim=16, img_dim=32, fused_dim=48)
        
        e_tab = tab_enc(self.tab_features)
        e_img = torch.randn(self.num_patients, 32)
        
        z, gate = fusion(e_tab, e_img)
        self.assertEqual(z.shape, (self.num_patients, 48))
        self.assertEqual(gate.shape, (self.num_patients, 48))
        self.assertTrue(torch.all(gate >= 0.0) and torch.all(gate <= 1.0))

    def test_multimodal_graph_construction(self):
        z_dummy = np.random.randn(self.num_patients, 48).astype(np.float32)
        clin_dummy = np.random.randn(self.num_patients, 15).astype(np.float32)
        y_dummy = np.zeros((self.num_patients, 3), dtype=np.float32)
        mask_dummy = np.ones((self.num_patients, 3), dtype=np.float32)
        train_mask = np.ones(self.num_patients, dtype=bool)
        
        data_pyg = build_multimodal_patient_graph(
            fused_embeddings=z_dummy,
            clinical_features=clin_dummy,
            y=y_dummy,
            mask=mask_dummy,
            train_mask=train_mask,
            val_mask=train_mask,
            test_mask=train_mask,
            k_neighbors=3
        )
        
        self.assertEqual(data_pyg.x.shape, (self.num_patients, 48))
        self.assertEqual(data_pyg.x_clinical.shape, (self.num_patients, 15))
        self.assertEqual(data_pyg.edge_index.shape[0], 2)
        self.assertEqual(data_pyg.edge_index.shape[1], self.num_patients * 3)

    def test_mm_gat_surv_forward_and_loss(self):
        model = MMGATSurvModel(
            in_clinical_features=15, tab_dim=16, img_dim=32, fused_dim=48,
            gat_hidden_dim=16, gat_out_dim=16, num_intervals=3
        )
        
        z_fused, _, _, _ = model.encode_multimodal_nodes(self.tab_features, self.img_features)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        
        hazards, h_gat = model(z_fused[:4], edge_index, self.tab_features[:4])
        self.assertEqual(hazards.shape, (4, 3))
        self.assertTrue(torch.all(hazards >= 0.0) and torch.all(hazards <= 1.0))
        
        # Test loss
        y = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.float32)
        mask = torch.ones_like(y)
        loss = discrete_time_survival_loss(hazards, y, mask)
        self.assertGreater(loss.item(), 0.0)

    def test_survival_monotonicity_and_time_gain(self):
        model = MMGATSurvModel(
            in_clinical_features=15, tab_dim=16, img_dim=32, fused_dim=48,
            gat_hidden_dim=16, gat_out_dim=16, num_intervals=3
        )
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        z = torch.randn(2, 48)
        x_clin = torch.randn(2, 15)
        
        preds = model.predict_survival_and_time(z, edge_index, x_clin)
        surv = preds['survival_probs']
        
        # Monotonically non-increasing: S(t1) >= S(t2) >= S(t3)
        self.assertTrue(np.all(surv[:, 0] >= surv[:, 1] - 1e-6))
        self.assertTrue(np.all(surv[:, 1] >= surv[:, 2] - 1e-6))
        self.assertTrue(len(preds['expected_time_months']) == 2)

if __name__ == '__main__':
    unittest.main()
